from cl_sae import (
    SAETrainer,
    SAECacheLayer,
    SAEComponentLayer,
    SAEConfig,
    SAETrainCache,
)
from nqgl.mlutils.components import (
    ComponentLayer,
    CacheModule,
    LayerComponent,
    CacheLayer,
)
from nqgl.mlutils.components.cache import Cache
from nqgl.mlutils.components.component_layer.resampler import (
    ResamplingMethod,
    NoResampling,
)
from resamplers import QueuedSVDResampler
from nqgl.mlutils.components.component_layer.freq_tracker import EMAFreqTracker
from sae_seq import SequentialCacheLayer, CatSeqCacheLayer

resampler = NoResampling
freq_tracker = EMAFreqTracker
import torch.nn as nn


class L1L0Reader(LayerComponent):
    _default_component_name = "cache_rewriter"

    def _update_from_cache(self, cache: Cache, **kwargs):
        # if cache._subcache_index + 1 in cache._parent._subcaches:
        #     return
        last_id = max([i for i in cache._subcaches if isinstance(i, int)])
        cache.l0 = cache[last_id].l0
        cache.l1 = cache[last_id].l1
        cache.l0l1 = cache[last_id].l0l1

        # cache += cache[)]


from cl_on_data import sae_cfg as cfg


# trainer = SAETrainer(
#     cfg=cfg,
#     sae=SAEComponentLayer(
#         cfg=cfg,
#         sae_cachelayer=SAECacheLayer(
#             cfg=cfg,
#             encoder_cachelayer=ComponentLayer(
#                 cachelayer=SequentialCacheLayer(
#                     CacheLayer.from_dims(d_in=cfg.d_data, d_out=cfg.d_data * 2),
#                     CacheLayer.from_dims(d_in=cfg.d_data * 2, d_out=cfg.d_data * 2),
#                     CacheLayer.from_dims(d_in=cfg.d_data * 2, d_out=cfg.d_dict),
#                 ),
#                 components=[L1L0Reader(), freq_tracker, resampler],
#                 train_cache=SAETrainCache(),
#                 eval_cache=SAETrainCache(),
#             ),
#         ),
#         components=[],
#     ),
# )
class SeqForSAE(SequentialCacheLayer):
    @property
    def W(self):
        return self._sequence[-1].W

    @property
    def b(self):
        return self._sequence[-1].b

    # def s


# cfg.d_in = cfg.d_data * 2
cfg.tied_init = False
trainer = SAETrainer(
    cfg=cfg,
    sae=SAEComponentLayer(
        cfg=cfg,
        sae_cachelayer=SAECacheLayer(
            cfg=cfg,
            encoder_cachelayer=CatSeqCacheLayer(
                CacheLayer.from_dims(
                    d_in=cfg.d_data, d_out=cfg.d_data, nonlinearity=nn.ReLU()
                ),
                # CacheLayer.from_dims(
                #     d_in=cfg.d_data * 2,
                #     d_out=cfg.d_data,
                #     nonlinearity=nn.ReLU(),
                # ),
                CacheLayer.from_dims(d_in=cfg.d_data * 2, d_out=cfg.d_dict),
            ),
            # encoder_cachelayer=ComponentLayer(
            #     cachelayer=CacheLayer.from_dims(
            #         d_in=cfg.d_data * 2, d_out=cfg.d_dict
            #     ),
            # ),
            other_encoder_components=[L1L0Reader()],
            freq_tracker_factory=freq_tracker,
            resampler_factory=resampler,
        ),
    ),
    # components=[],
)


# SequentialCacheLayer(
#     CacheLayer.from_dims(d_in=cfg.d_data, d_out=cfg.d_data * 2),
#     CacheLayer.from_dims(d_in=cfg.d_data * 2, d_out=cfg.d_data * 2),
#     ComponentLayer(
#         cachelayer=cachelayer,
#         components=[freq_tracker, resampler] + other_encoder_components,
#         train_cache=train_cache,
#         eval_cache=eval_cache or train_cache.clone(),
#     ),
# )
