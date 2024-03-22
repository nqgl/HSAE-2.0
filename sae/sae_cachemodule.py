from nqgl.mlutils.components.cache_layer import CacheLayer, CacheModule
from nqgl.mlutils.components.component_layer import ComponentLayer
from nqgl.mlutils.components.component_layer.freq_tracker import CountingFreqTracker
from nqgl.mlutils.components.component_layer.resampler import NoResampling
from sae.config import SAECache, SAEConfig, SAETrainCache


import torch
import torch.nn as nn


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
        train_cache = train_cache or SAETrainCache()
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
                self.decoder(self.encoder(x - self.b_dec, cache=cache["encoder"]))
                + self.b_dec
            )
        )
        return y_pred
