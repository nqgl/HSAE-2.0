from sae.cl_sae import (
    SAEComponentLayer,
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
from components.resamplers import QueuedSVDResampler
from nqgl.mlutils.components.component_layer.freq_tracker import EMAFreqTracker
from sae.config import SAEConfig, SAETrainCache
from sae.sae_cachemodule import SAECacheLayer
from sae.trainer import SAETrainer
from deep_sae.sae_seq import SequentialCacheLayer, CatSeqCacheLayer
from components.resamplers import (
    QueuedTopkDiffDecYEncResampler,
    QueuedOrthTopkDiffDecYEncResampler,
)
from unpythonic import box
import torch.nn as nn


class L1L0Reader(LayerComponent):
    _default_component_name = "cache_rewriter"

    def _register_parent_layer(self, layer: ComponentLayer):
        super()._register_parent_layer(layer)

        def handle_acts(cache: Cache, acts):
            n = len(self._layer.cachelayer._sequence)
            if cache._subcache_index == n:
                cache._parent.acts = acts

        layer.train_cache_template.register_write_callback("acts", handle_acts)

    def _update_from_cache(self, cache: Cache, **kwargs):
        # if cache._subcache_index + 1 in cache._parent._subcaches:
        #     return
        last_id = max([i for i in cache._subcaches if isinstance(i, int)])
        cache.l0 = cache[last_id].l0
        cache.l1 = cache[last_id].l1
        if cache[last_id].has.l0l1:
            cache.l0l1 = cache[last_id].l0l1

        # cache += cache[)]


from training.cl_on_data import sae_cfg as cfg


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
class SeqForSAEAdapter(CacheModule):
    def __init__(self, *modules):
        super().__init__(*modules)

    @property
    def W(self):
        return self._sequence[-1].cachelayer.W

    @property
    def b(self):
        return self._sequence[-1].cachelayer.b


class SeqForSAE(SeqForSAEAdapter, SequentialCacheLayer): ...


class CatSeqForSAE(SeqForSAEAdapter, CatSeqCacheLayer): ...


# def s


fibox = box()

# resampler = QueuedTopkDiffDecYEncResampler(
resampler = QueuedOrthTopkDiffDecYEncResampler(
    cfg=cfg.resampler_cfg, get_optim_fn=lambda: fibox.x.optim
)
freq_tracker = EMAFreqTracker(cfg=cfg.freq_tracker_cfg)
MULT_IN = 3
resampled_layer = ComponentLayer(
    cachelayer=CacheLayer.from_dims(d_in=cfg.d_data * MULT_IN, d_out=cfg.d_dict),
    components=[freq_tracker, resampler],
    train_cache=SAETrainCache(),
    eval_cache=SAETrainCache(),
)


# cfg.d_in = cfg.d_data * 2
cfg.tied_init = False
trainer = SAETrainer(
    cfg=cfg,
    sae=SAEComponentLayer(
        cfg=cfg,
        sae_cachelayer=SAECacheLayer(
            cfg=cfg,
            encoder=ComponentLayer(
                cachelayer=CatSeqForSAE(
                    # CacheLayer.from_dims(
                    #     d_in=cfg.d_data, d_out=cfg.d_data, nonlinearity=nn.LeakyReLU()
                    # ),
                    CacheLayer.from_dims(
                        d_in=cfg.d_data,
                        d_out=cfg.d_data * 2,
                        nonlinearity=nn.LeakyReLU(),
                    ),
                    CacheLayer.from_dims(
                        d_in=cfg.d_data * 3,
                        d_out=cfg.d_data * (MULT_IN - 1),
                        nonlinearity=nn.LeakyReLU(),
                    ),
                    resampled_layer,
                ),
                components=[L1L0Reader()],
            ),
            # encoder_cachelayer=ComponentLayer(
            #     cachelayer=CacheLayer.from_dims(
            #         d_in=cfg.d_data * 2, d_out=cfg.d_dict
            #     ),
            # ),
            # other_encoder_components=[L1L0Reader()],
            # freq_tracker_factory=freq_tracker,
            # resampler_factory=resampler,
            resampler=resampler,
            freq_tracker=freq_tracker,
        ),
    ),
    # components=[],
)
fibox << trainer

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
