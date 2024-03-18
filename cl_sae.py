import torch
from nqgl.mlutils.components.component_layer import ComponentLayer
from unpythonic import box
from nqgl.mlutils.components.cache_layer import (
    CacheModule,
    CacheLayer,
    CacheProcLayer,
    CacheLayerConfig,
)
from nqgl.mlutils.components.component_layer import ComponentLayer, LayerComponent
from nqgl.mlutils.components.component_layer.resampler import (
    ResamplingMethod,
    ResamplerConfig,
    ResamplerComponent,
    QueuedResamplerConfig,
    TopKResamplingConfig,
    NoResampling,
)


import torch
import torch.nn as nn
import tqdm

from nqgl.mlutils.components.component_layer.freq_tracker import (
    CountingFreqTracker,
    CountingFreqTrackerConfig,
    FreqTracker,
    EMAFreqTracker,
    EMAFreqTrackerConfig,
)
from nqgl.mlutils.components.cache import Cache
from unpythonic import box
import wandb
from dataclasses import dataclass
from nqgl.mlutils.optimizations.norepr import fastpartial
from nqgl.mlutils.components.config import WandbDynamicConfig
from resamplers import QueuedTopkDiffResampler


@dataclass
class SAEConfig(WandbDynamicConfig):
    lr: float = 1e-3
    betas: tuple = (0.5, 0.99)
    d_data: int = 768
    d_dict: int = 768 * 1
    resampler_cfg: ResamplerConfig = ResamplerConfig()
    freq_tracker_cfg: CountingFreqTrackerConfig = CountingFreqTrackerConfig()
    l1_coeff: float = 1e-3
    device: str = "cuda"
    start_from_dead: bool = False
    architecture: str = "n/a"
    l2_loss_type: str = "squared/40"
    optim: str = "adam"
    tied_init: bool = True
    b_enc_init: float = 0.0
    bias_lr_coeff: float = 3
    l0l1_coeff: float = None
    l0l1_thresh: int = 0
    different_d_data_in: int = None
    d_in: int = None
    d_out: int = None

    def __post_init__(self):
        super().__post_init__()
        self.d_in = self.d_in or self.d_data
        self.d_out = self.d_out or self.d_data


class SAECache(Cache):
    l1: torch.NumberType = ...
    l0: torch.NumberType = ...
    l0l1: torch.NumberType
    cumulative_num_resampled = ...
    x = ...

    @staticmethod
    def process_acts(cache, acts: torch.Tensor):
        cache.l1 = acts.relu().sum(-1).mean()
        cache.l0 = acts.count_nonzero(dim=-1).to(torch.get_default_dtype()).mean()

    def __init__(self):
        super().__init__()

        self.register_write_callback("acts", self.process_acts)


class SAETrainCache(SAECache):
    l2: torch.NumberType = ...
    y_pred: torch.Tensor = ...
    l2_norm: torch.NumberType = ...
    num_queued_for_reset: int = ...

    # In the future, possibly better to replace these with a component
    @staticmethod
    def process_y(cache, y):
        sumsquares = (cache.y_pred - y).pow(2).sum(-1)
        cache.l2 = sumsquares.mean()
        cache.l2_norm = (cache.y_pred - y).norm(dim=-1).mean()

    def __init__(self):
        super().__init__()

        self.register_write_callback("y", self.process_y)


from typing import Type


class SAECacheLayer(CacheModule):
    decoder: nn.Linear
    encoder: ComponentLayer
    b_dec: nn.Parameter

    def __init__(
        self,
        cfg: SAEConfig,
        resampler_factory=NoResampling,
        freq_tracker_factory=None,
        other_encoder_components=[],
        train_cache=None,
        eval_cache=None,
        encoder_cachelayer: CacheLayer = None,
        encoder=None,
        resampler=None,
        freq_tracker=None,
    ):
        super().__init__()
        self.cfg = cfg
        # self.decoder = CacheProcLayer(
        #     CacheLayer.from_dims(
        #         d_in=cfg.d_dict, d_out=cfg.d_data, bias=False, nonlinearity=lambda x: x
        #     )
        # )
        train_cache = train_cache or SAETrainCache()
        # assert other_encoder_components == []
        self.decoder = nn.Linear(cfg.d_dict, cfg.d_out, bias=False, device=cfg.device)
        resampler = resampler or resampler_factory(
            cfg.resampler_cfg, W_next=self.decoder.weight
        )
        resampler.W_next = self.decoder.weight
        freq_tracker = freq_tracker or (freq_tracker_factory or CountingFreqTracker)(
            cfg.freq_tracker_cfg
        )
        cachelayer = encoder_cachelayer or CacheLayer.from_dims(
            d_in=cfg.different_d_data_in or cfg.d_data, d_out=cfg.d_dict
        )
        self.encoder = encoder or ComponentLayer(
            cachelayer=cachelayer,
            components=[freq_tracker, resampler] + other_encoder_components,
            train_cache=train_cache,
            eval_cache=eval_cache or train_cache.clone(),
        )
        if cfg.tied_init:
            self.decoder.weight.data[:] = self.encoder.cachelayer.W.data
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_data))  # TODO
        if cfg.l0l1_coeff is not None:
            self.encoder.train_cache_template.l0l1 = ...

            def calc_l0l1(cache, acts):
                l0_over_b = acts.count_nonzero(dim=-1).to(torch.get_default_dtype())
                l0_adj = torch.relu(l0_over_b - cfg.l0l1_thresh)
                l1_over_b = acts.relu().sum(-1)
                l0l1_over_b = l0_adj * l1_over_b
                cache.l0l1 = l0l1_over_b.mean()
                assert l0_over_b.shape == l1_over_b.shape == l0l1_over_b.shape

            self.encoder.train_cache_template.register_write_callback("acts", calc_l0l1)

    def forward(self, x, cache: SAECache):
        # cache.x = x
        cache.y_pred = (
            y_pred := (
                self.decoder(
                    self.encoder(x - self.b_dec, cache=cache["encoder"])
                )  # in the future I think I do prefer cache["encoder"]
                + self.b_dec
            )
        )
        return y_pred

    def zero(self, n=0):
        # self.decoder.weight.data[:] = 0.0 + n
        self.encoder.cachelayer.W.data[:] = 0.0 + n
        self.encoder.cachelayer.b.data[:] = -1e-9 + n


class SAEComponentLayer(ComponentLayer):
    # resampler: ResamplerComponent
    # activations: FreqTracker
    cachelayer: SAECacheLayer

    def __init__(
        self,
        cfg: SAEConfig,
        resampler_factory=NoResampling,
        freq_tracker_factory=None,
        components=[],
        sae_cachelayer=None,
        train_cache=None,
        eval_cache=None,
    ):
        # SAECacheLayer(
        #     cfg,
        #     cache_factory=SAETrainCache,
        #     resampler_factory=resampler_factory,
        #     freq_tracker_factory=freq_tracker_factory,
        #     other_encoder_components=[],
        # )
        assert sae_cachelayer is None or (
            freq_tracker_factory is None and resampler_factory is NoResampling
        )
        super().__init__(
            cachelayer=sae_cachelayer
            or SAECacheLayer(
                cfg,
                train_cache=SAETrainCache(),
                resampler_factory=resampler_factory,
                freq_tracker_factory=freq_tracker_factory or CountingFreqTracker,
            ),
            components=components,
            train_cache=train_cache or SAETrainCache(),
            eval_cache=eval_cache or SAETrainCache(),
        )


class SAETrainer:
    def __init__(
        self,
        cfg: SAEConfig,
        resampler_factory=None,
        # sae: SAEComponentLayer,
        freq_tracker_factory=None,
        other_encoder_components=[],
        sae: SAEComponentLayer = None,
        sae_cachelayer=None,
    ):
        assert sae is None or sae_cachelayer is None
        self.cfg = cfg
        if not sae:
            resampler_factory = resampler_factory.bind_named_args(
                get_optim_fn=lambda: self.optim
            )
        self.sae = (
            sae
            or SAEComponentLayer(
                cfg,
                resampler_factory=resampler_factory,
                freq_tracker_factory=freq_tracker_factory,
                sae_cachelayer=sae_cachelayer,
            )
        ).to(cfg.device)
        # ComponentLayer(
        #     cachelayer=sae_cachelayer
        #     or SAECacheLayer(
        #         cfg,
        #         train_cache=SAETrainCache(),
        #         resampler_factory=resampler_factory,
        #         freq_tracker_factory=freq_tracker_factory,
        #     ),
        #     components=[],
        #     train_cache=SAETrainCache(),
        #     eval_cache=SAETrainCache(),
        # ).to(cfg.device)
        self.t = 1
        self.extra_calls = []
        self.init_optim()

    def init_optim(self):
        if self.cfg.optim == "adam":
            self.optim = torch.optim.Adam(
                self.parameters(), lr=self.cfg.lr, betas=self.cfg.betas
            )
        elif self.cfg.optim == "radam":
            self.optim = torch.optim.RAdam(
                self.parameters(), lr=self.cfg.lr, betas=self.cfg.betas
            )
        elif self.cfg.optim == "sgd":
            self.optim = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.lr,
                momentum=self.cfg.betas[0],
                nesterov=True,
            )
        elif self.cfg.optim == "nadam":
            self.optim = torch.optim.NAdam(
                self.parameters(), lr=self.cfg.lr, betas=self.cfg.betas
            )

    def update_optim_lrs(self):
        for pg in self.optim.param_groups:
            if pg["name"] == "bias":
                pg["lr"] = self.cfg.lr * self.cfg.bias_lr_coeff
            elif pg["name"] == "weight":
                pg["lr"] = self.cfg.lr
            else:
                raise ValueError(f"param group name {pg['name']} not recognized")

    def parameters(self):
        bias_params = []
        for name, param in self.sae.named_parameters():
            if name.endswith(".b_dec") or name.endswith(".b") or name.endswith(".bias"):
                bias_params.append(param)

        weights = set(self.sae.parameters()) - set(bias_params)
        groups = [
            {
                "params": bias_params,
                "lr": self.cfg.lr * self.cfg.bias_lr_coeff,
                "name": "bias",
            },
            {"params": list(weights), "lr": self.cfg.lr, "name": "weight"},
        ]
        return groups
        biases = [self.sae.cachelayer.b_dec, self.sae.cachelayer.encoder.cachelayer.b]
        weights = [
            self.sae.cachelayer.decoder.weight,
            self.sae.cachelayer.encoder.cachelayer.W,
        ]
        return [
            {
                "params": biases,
                "lr": self.cfg.lr * self.cfg.bias_lr_coeff,
                "name": "bias",
            },
            {"params": weights, "lr": self.cfg.lr, "name": "weight"},
        ]

    def train(self, buffer):
        l1_coeff = self.cfg.l1_coeff
        if self.cfg.start_from_dead:
            with torch.no_grad():
                self.sae.cachelayer.zero()
                self.sae.cachelayer.encoder.cachelayer.W[0, 0] = torch.randn_like(
                    self.sae.cachelayer.encoder.cachelayer.W[0, 0]
                )
        self.norm_dec()
        for bn in buffer:
            try:
                x, y = bn
                assert isinstance(bn, tuple)
            except:
                y = (x := bn)
                assert isinstance(bn, torch.Tensor)
            # in the future trainer could itself be a CacheProcLayer maybe
            cache: SAETrainCache = self.sae.generate_default_cache()
            y_pred = self.sae(x, cache=cache)
            cache.y = ...
            cache.y = y
            self.step(cache=cache, x=x, y_pred=y_pred, y=x)
            self.full_log(cache)
            self.t += 1

    def step(self, cache: SAETrainCache, x, y_pred, y=None):
        loss = self.loss(cache)
        cache.loss = loss
        loss.backward()
        self.norm_dec_grads()
        resample_before_step = getattr(
            self.cfg.resampler_cfg, "resample_before_step", False
        )
        if resample_before_step:
            for c in cache.search("resample"):
                c.num_resampled = ...
                c.resample(x=x)
        self.optim.step()
        self.optim.zero_grad()
        self.norm_dec()
        if not resample_before_step:
            for c in cache.search("resample"):
                c.num_resampled = ...
                c.resample(x=x)

        # if self.t % 100 == 0:
        for call in self.extra_calls:
            call(cache)

    @torch.no_grad()
    def norm_dec_grads(self):
        grad = self.sae.cachelayer.decoder.weight.grad
        dec_normed = (
            self.sae.cachelayer.decoder.weight.data
            / self.sae.cachelayer.decoder.weight.data.norm(dim=0, keepdim=True)
        )
        grad_orth = grad - (dec_normed * grad).sum(0, keepdim=True) * dec_normed
        self.sae.cachelayer.decoder.weight.grad[:] = grad_orth

    def norm_dec(self):
        if self.t % 1000 == 0:
            print("norm dec", self.sae.cachelayer.decoder.weight.shape)
        self.sae.cachelayer.decoder.weight.data[:] = (
            self.sae.cachelayer.decoder.weight.data
            / self.sae.cachelayer.decoder.weight.data.norm(dim=0, keepdim=True)
        )

    def loss(self, cache: SAETrainCache):
        # print("l1 coeff", self.cfg.l1_coeff)
        return (
            self.get_l2_type(cache, self.cfg.l2_loss_type)
            + cache["encoder"].l1 * self.cfg.l1_coeff
            + (
                cache["encoder"].l0l1 * self.cfg.l0l1_coeff
                if self.cfg.l0l1_coeff and self.cfg.l0l1_coeff != 0
                else 0
            )
        )

    def get_l2_type(self, cache, l2_type):
        if isinstance(l2_type, str):
            if l2_type == "squared/40":
                return cache.l2 / 40
            elif l2_type == "l2_norm":
                return cache.l2_norm
            elif l2_type == "l2_root":
                return cache.l2**0.5
            elif l2_type == "l2_norm_squared/40":
                return cache.l2_norm**2 / 40
            else:
                raise ValueError(f"l2_type {l2_type} not recognized")
        else:
            v = 0
            for l2 in l2_type:
                v = v + self.get_l2_type(cache, l2)
            return v / len(l2_type)

    def full_log(self, cache: Cache):
        if self.t % 10 != 0:
            return
        d = cache.logdict(excluded=["acts", "y_pred", "x", "y", "resample"])
        # print(d)
        if wandb.run is not None:
            wandb.log(d, step=self.t)


@dataclass
class FreqTrackerCombinedConfig(CountingFreqTrackerConfig, EMAFreqTrackerConfig): ...


def main():
    # sae = SAECacheLayer(sae_cfg)
    torch.set_default_dtype(torch.float32)
    device = "cuda"
    batch_size = 1024
    sae_cfg = SAEConfig(
        lr=1e-3,
        betas=(0.0, 0.99),
        d_data=768,
        d_dict=768 * 2,
        resampler_cfg=QueuedResamplerConfig(
            dead_threshold=0.001,
            min_viable_count=500 * batch_size,
            reset_all_freqs_interval=2_000,
            reset_all_freqs_offset=1000,
            check_frequency=200,
            resample_frequency=2,
            num_to_resample=2,
        ),
        freq_tracker_cfg=CountingFreqTrackerConfig(),
        l1_coeff=1 / (6),
        device=device,
    )
    wandb.init(entity="sae_all", project="saes_2.0_testing", config=sae_cfg)

    # vecs = torch.eye(768)
    trainer = SAETrainer(
        sae_cfg,
        resampler_factory=QueuedTopkDiffResampler,
    )
    print([i for i in trainer.sae.named_modules()])

    m = torch.randn(768 // 4, 768, device=device) / 10
    m = m.repeat(4, 1)
    num_features = 768 * 2
    m = torch.randn(num_features, 768, device=device)
    m = m / m.norm(dim=1, keepdim=True)
    avg_n_features_active = 10

    def buffer():
        for i in tqdm.tqdm(range(90000)):
            rv = (
                torch.rand(batch_size, num_features, device=device)
                < (avg_n_features_active / num_features)
            ).to(torch.get_default_dtype())
            rv = (
                rv
                * 10
                * (torch.rand(batch_size, num_features, device=device) + 0.5)
                / avg_n_features_active**0.5
            )
            yield rv @ m

    print(trainer.sae.cachelayer.encoder.component_architectures())
    assert False
    trainer.sae.cachelayer.zero(1)
    trainer.train(buffer())


if __name__ == "__main__":
    main()
