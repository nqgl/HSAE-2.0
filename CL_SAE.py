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
)
import torch
import torch.nn as nn

from nqgl.mlutils.components.component_layer.freq_tracker import (
    CountingFreqTracker,
    CountingFreqTrackerConfig,
    FreqTracker,
)
from nqgl.mlutils.components.cache import Cache
from unpythonic import box
import wandb
from dataclasses import dataclass


@dataclass
class SAEConfig:
    lr: float = 1e-3
    betas: tuple = (0.5, 0.99)
    d_data: int = 768
    d_dict: int = 768 * 1
    resampler_cfg: ResamplerConfig = ResamplerConfig()
    freq_tracker_cfg: CountingFreqTrackerConfig = CountingFreqTrackerConfig()
    l1_coeff: float = 1e-3
    device: str = "cuda"


class SAECache(Cache):
    l1: torch.NumberType = ...
    l0: torch.NumberType = ...

    @staticmethod
    def process_acts(cache, acts):
        cache.l1 = acts.abs().mean(dim=0).sum()
        cache.l0 = acts.abs().mean(dim=0).sum()

    def __init__(self):
        super().__init__()

        self.register_write_callback("acts", self.process_acts)


class SAETrainCache(SAECache):
    l2: torch.NumberType = ...
    y_pred: torch.Tensor = ...

    # In the future, possibly better to replace these with a component
    @staticmethod
    def process_y(cache, y):
        cache.l2 = (cache.y_pred - y).pow(2).mean()

    def __init__(self):
        super().__init__()

        self.register_write_callback("y", self.process_y)


class SAECacheLayer(CacheModule):
    cfg: dict

    def __init__(self, cfg: SAEConfig, cache_factory=SAECache, eval_cache_factory=None):
        super().__init__()
        self.cfg = cfg
        # self.decoder = CacheProcLayer(
        #     CacheLayer.from_dims(
        #         d_in=cfg.d_dict, d_out=cfg.d_data, bias=False, nonlinearity=lambda x: x
        #     )
        # )
        self.decoder = nn.Linear(cfg.d_dict, cfg.d_data, bias=False, device=cfg.device)
        cachelayer = CacheLayer.from_dims(d_in=cfg.d_data, d_out=cfg.d_dict)
        resampler = ResamplingMethod(
            cfg.resampler_cfg, W_next=self.decoder.weight.transpose(-2, -1)
        )
        freq_tracker = CountingFreqTracker(cfg.freq_tracker_cfg)
        self.encoder = ComponentLayer(
            cachelayer=cachelayer,
            components=[
                freq_tracker,
                resampler,
            ],
            train_cache=cache_factory(),
            eval_cache=(eval_cache_factory or cache_factory)(),
        )
        self.b_dec = 0  # TODO

    def forward(self, x, cache: SAECache):
        cache.y_pred = (
            y_pred := (
                self.decoder(
                    self.encoder(x - self.b_dec, cache=cache["encoder"])
                )  # in the future I think I do prefer cache["encoder"]
                + self.b_dec
            )
        )
        return y_pred


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
        )


class SAETrainer:
    def __init__(self, cfg: SAEConfig):
        self.cfg = cfg
        self.sae = ComponentLayer(
            cachelayer=SAECacheLayer(cfg, cache_factory=SAETrainCache),
            components=[],
            train_cache=SAETrainCache(),
            eval_cache=SAETrainCache(),
        ).to(cfg.device)
        self.optim = torch.optim.Adam(self.sae.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.t = 1

    def train(self, buffer):
        for bn in buffer:
            try:
                x, y = bn
                assert isinstance(bn, tuple)
            except:
                y = (x := bn)
                assert isinstance(bn, torch.Tensor)
            # in the future trainer could itself be a CacheProcLayer maybe
            cache = self.sae.generate_default_cache()
            y_pred = self.sae(x, cache=cache)
            cache.y = y
            self.step(cache=cache, x=x, y_pred=y_pred, y=x)
            self.log(cache)
            self.t += 1

    def step(self, cache: SAETrainCache, x, y_pred, y=None):
        y = x if y is None else y
        # cache.l2 = (y_pred - y).pow(2).mean()
        loss = self.loss(cache)
        if self.t % 100 == 0:
            print(loss)
        if cache["encoder"].has.resample:
            cache["encoder"].num_reset = ...
            cache["encoder"].resample(x=x)
            print("resampled", cache["encoder"].num_reset)

        loss.backward()
        self.norm_dec_grads()
        self.optim.step()
        self.optim.zero_grad()
        self.norm_dec()

    def norm_dec_grads(self):
        pass  # TODO

    def norm_dec(self):
        pass  # TODO

    def loss(self, cache: SAETrainCache):
        return cache.l2 + cache["encoder"].l1 * self.cfg.l1_coeff

    def log(self, cache):
        d = {}
        if cache.has.l2:
            d["l2"] = cache.l2
        if cache.has.l1:
            d["l1"] = cache.l1
        if cache.has.l0:
            d["l0"] = cache.l0

        # wandb.log({})


def main():
    # sae = SAECacheLayer(sae_cfg)
    device = "cuda"

    sae_cfg = SAEConfig(
        lr=1e-3,
        betas=(0.5, 0.99),
        d_data=768,
        d_dict=768 * 1,
        resampler_cfg=ResamplerConfig(
            dead_threshold=0.001,
            min_viable_count=500,
            reset_all_freqs_interval=2_000,
            reset_all_freqs_offset=1000,
        ),
        freq_tracker_cfg=CountingFreqTrackerConfig(),
        l1_coeff=1e-3,
        device=device,
    )
    # sae = SAEComponentLayer(sae_cfg)
    sae = ComponentLayer(
        cachelayer=SAECacheLayer(sae_cfg),
        components=[],
    )  # TODO

    vecs = torch.eye(768)
    # for i in range(10):
    #     x = torch.randn(80, sae_cfg.d_data)
    #     y = sae(x)
    #     print(y.shape)
    #     # print(sae.activations.freqs)
    #     print(sae.cachelayer.encoder.activations.freqs)
    m = torch.randn(768, 768, device=device) / 10
    trainer = SAETrainer(sae_cfg)
    m = torch.randn(768 // 4, 768, device=device) / 10
    m = m.repeat(4, 1)

    def buffer():
        for i in range(100000):
            rv = torch.randn(80, 768, device=device) / 10
            yield rv @ m

    trainer.train(buffer())


if __name__ == "__main__":
    main()
