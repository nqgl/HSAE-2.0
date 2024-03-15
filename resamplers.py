from dataclasses import dataclass
from nqgl.mlutils.components.component_layer.resampler import (
    QueuedResamplerConfig,
    DiffResamplingDirections,
    QueuedResampler,
    RandomResamplingDirections,
    SVDResampling,
    TopKResampling,
)
from nqgl.mlutils.components.component_layer.resampler.methods.orth_resampler import (
    OrthDiffResampling,
    OrthDiffResamplingConfig,
    OrthRankedTopkResampling,
)
from nqgl.mlutils.components.component_layer.resampler.resampler import (
    DiffDecYEncResamplingDirections,
)


@dataclass
class QKOrthResampleConfig(QueuedResamplerConfig, OrthDiffResamplingConfig):
    resample_top_k: int = 256


class QueuedOrthDiffResampler(QueuedResampler, OrthDiffResampling): ...


class QueuedTopkDiffDecYEncResampler(
    TopKResampling, DiffDecYEncResamplingDirections, QueuedResampler
): ...


class QueuedOrthTopkDiffDecYEncResampler(
    OrthRankedTopkResampling, DiffDecYEncResamplingDirections, QueuedResampler
): ...


class QueuedRandomResampler(QueuedResampler, RandomResamplingDirections): ...


class QueuedDiffResampler(QueuedResampler, DiffResamplingDirections): ...


class QueuedTopkDiffResampler(
    QueuedResampler, TopKResampling, DiffResamplingDirections
): ...


class QueuedSVDResampler(QueuedResampler, SVDResampling): ...


class QueuedTopkSVDResampler(QueuedResampler, TopKResampling, SVDResampling): ...
