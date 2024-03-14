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
    QueuedResampler,
    RandomResamplingDirections,
    DiffResamplingDirections,
    YResamplingDirections,
    TopKResampling,
    SVDResampling,
    TopKResamplingConfig,
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


class SAECache(Cache):
    l1: torch.NumberType = ...
    l0: torch.NumberType = ...
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


class SAECacheLayer(CacheModule):
    decoder: nn.Linear
    encoder: ComponentLayer
    b_dec: nn.Parameter

    def __init__(
        self,
        cfg: SAEConfig,
        resampler_factory,
        freq_tracker_factory=None,
        cache_factory=SAECache,
        other_encoder_components=[],
        eval_cache_factory=None,
    ):
        super().__init__()
        self.cfg = cfg
        # self.decoder = CacheProcLayer(
        #     CacheLayer.from_dims(
        #         d_in=cfg.d_dict, d_out=cfg.d_data, bias=False, nonlinearity=lambda x: x
        #     )
        # )
        assert other_encoder_components == []
        self.decoder = nn.Linear(cfg.d_dict, cfg.d_data, bias=False, device=cfg.device)
        resampler = resampler_factory(cfg.resampler_cfg, W_next=self.decoder.weight)
        freq_tracker = (freq_tracker_factory or CountingFreqTracker)(
            cfg.freq_tracker_cfg
        )

        cachelayer = CacheLayer.from_dims(d_in=cfg.d_data, d_out=cfg.d_dict)
        self.encoder = ComponentLayer(
            cachelayer=cachelayer,
            components=[freq_tracker, resampler] + other_encoder_components,
            train_cache=cache_factory(),
            eval_cache=(eval_cache_factory or cache_factory)(),
        )
        if cfg.tied_init:
            self.decoder.weight.data[:] = self.encoder.cachelayer.W.data
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_data))  # TODO

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
    resampler: ResamplerComponent
    activations: FreqTracker

    def __init__(self, cfg: SAEConfig, components=None):

        super().__init__(
            cachelayer=SAECacheLayer(cfg, cache_factory=SAETrainCache),
            components=components
            or [
                CountingFreqTracker(cfg.freq_tracker_cfg),
                ResamplingMethod(cfg.resampler_cfg),
            ],
            train_cache=SAETrainCache(),
            eval_cache=SAETrainCache(),
        )


class SAETrainer:
    def __init__(
        self,
        cfg: SAEConfig,
        resampler_factory,
        # sae: SAEComponentLayer,
        freq_tracker_factory=None,
        other_encoder_components=[],
    ):

        self.cfg = cfg
        resampler_factory = resampler_factory.bind_named_args(
            get_optim_fn=lambda: self.optim
        )
        self.sae = ComponentLayer(
            cachelayer=SAECacheLayer(
                cfg,
                cache_factory=SAETrainCache,
                resampler_factory=resampler_factory,
                freq_tracker_factory=freq_tracker_factory,
                other_encoder_components=[],
            ),
            components=[],
            train_cache=SAETrainCache(),
            eval_cache=SAETrainCache(),
        ).to(cfg.device)

        if cfg.optim == "adam":
            self.optim = torch.optim.Adam(self.parameters(), lr=cfg.lr, betas=cfg.betas)
        elif cfg.optim == "radam":
            self.optim = torch.optim.RAdam(
                self.parameters(), lr=cfg.lr, betas=cfg.betas
            )
        elif cfg.optim == "sgd":
            self.optim = torch.optim.SGD(
                self.parameters(), lr=cfg.lr, momentum=cfg.betas[0], nesterov=True
            )
        elif cfg.optim == "nadam":
            self.optim = torch.optim.NAdam(
                self.parameters(), lr=cfg.lr, betas=cfg.betas
            )
        self.t = 1
        self.extra_calls = []

    def parameters(self):
        biases = [self.sae.cachelayer.b_dec, self.sae.cachelayer.encoder.cachelayer.b]
        weights = [
            self.sae.cachelayer.decoder.weight,
            self.sae.cachelayer.encoder.cachelayer.W,
        ]
        return [
            {"params": biases, "lr": self.cfg.lr * self.cfg.bias_lr_coeff},
            {"params": weights},
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
            if self.t % 4000 == 1:
                for call in self.extra_calls:
                    call()
            # if self.t in (998, 1950):
            #     self.sae.cachelayer.zero(1e-9)

            # if self.cfg.start_from_dead:
            #     if self.t <= 1000:
            #         self.cfg.l1_coeff = l1_coeff * ((self.t) % (1000) + 1)
            #     else:
            #         self.cfg.l1_coeff = l1_coeff

            try:
                x, y = bn
                assert isinstance(bn, tuple)
            except:
                y = (x := bn)
                assert isinstance(bn, torch.Tensor)
            # in the future trainer could itself be a CacheProcLayer maybe
            cache: SAETrainCache = self.sae.generate_default_cache()
            y_pred = self.sae(x, cache=cache)
            cache.y = y
            self.step(cache=cache, x=x, y_pred=y_pred, y=x)
            self.full_log(cache)
            self.t += 1

    def step(self, cache: SAETrainCache, x, y_pred, y=None):
        # cache.l2 = (y_pred - y).pow(2).mean()
        loss = self.loss(cache)
        # if self.t % 100 == 0:
        #     print(loss)
        if cache["encoder"].has.resample:
            cache["encoder"].num_resampled = ...
            cache["encoder"].resample(x=x)
            # print("resampled", cache["encoder"].num_resampled)
        cache.loss = loss
        loss.backward()
        self.norm_dec_grads()
        self.optim.step()
        self.optim.zero_grad()
        self.norm_dec()

    @torch.no_grad()
    def norm_dec_grads(self):
        dec_normed = self.sae.cachelayer.decoder.weight.data

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


class QueuedRandomResampler(QueuedResampler, RandomResamplingDirections): ...


class QueuedDiffResampler(QueuedResampler, DiffResamplingDirections): ...


class QueuedTopkDiffResampler(
    QueuedResampler, TopKResampling, DiffResamplingDirections
): ...


class QueuedSVDResampler(QueuedResampler, SVDResampling): ...


class QueuedTopkSVDResampler(QueuedResampler, TopKResampling, SVDResampling): ...


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
