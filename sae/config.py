from nqgl.mlutils.components.cache import Cache
from nqgl.mlutils.components.component_layer.freq_tracker import (
    CountingFreqTrackerConfig,
)
from nqgl.mlutils.components.component_layer.resampler import ResamplerConfig
from nqgl.mlutils.components.config import WandbDynamicConfig

import torch
from dataclasses import dataclass


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
    batch_size: int = None

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
