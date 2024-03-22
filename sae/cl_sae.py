import torch
from nqgl.mlutils.components.component_layer import ComponentLayer
from unpythonic import box
from nqgl.mlutils.components.component_layer import ComponentLayer, LayerComponent
from nqgl.mlutils.components.component_layer.resampler import (
    QueuedResamplerConfig,
    NoResampling,
)


import torch
import tqdm

from nqgl.mlutils.components.component_layer.freq_tracker import (
    CountingFreqTracker,
    CountingFreqTrackerConfig,
    EMAFreqTrackerConfig,
)
from unpythonic import box
import wandb
from dataclasses import dataclass
from nqgl.mlutils.optimizations.norepr import fastpartial
from components.resamplers import QueuedTopkDiffResampler


from typing import Type

from sae.config import SAEConfig, SAETrainCache
from sae.sae_cachemodule import SAECacheLayer


class SAEComponentLayer(ComponentLayer):
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


@dataclass
class FreqTrackerCombinedConfig(CountingFreqTrackerConfig, EMAFreqTrackerConfig): ...


def main():
    from sae.trainer import SAETrainer

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
    trainer.sae.cachelayer.zero(1)
    trainer.train(buffer())


if __name__ == "__main__":
    main()
