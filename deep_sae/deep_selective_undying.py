from sae.cl_sae import (
    SAETrainer,
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

# from resamplers import QueuedSVDResampler
from nqgl.mlutils.components.component_layer.freq_tracker import EMAFreqTracker
from sae.config import SAEConfig, SAETrainCache
from sae.sae_cachemodule import SAECacheLayer
from deep_sae.sae_seq import SequentialCacheLayer, CatSeqCacheLayer

from nqgl.mlutils.components.component_layer.resampler.methods.selective_undying import (
    SelectiveUndyingResamplerConfig,
    SelectiveUndyingResampler,
    SerializableNonlinearity,
)
from unpythonic import box
import torch.nn as nn
from dataclasses import dataclass, asdict


class RewriteLastLayerOfSeqToParent(LayerComponent):
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


@dataclass
class SelectiveMergedCfg(
    SelectiveUndyingResamplerConfig, cfg.resampler_cfg.__class__
): ...


undying_coeff_mult = 2
cfg.resampler_cfg = SelectiveMergedCfg(
    **asdict(cfg.resampler_cfg),
    undying_relu=SerializableNonlinearity(
        "undying_relu",
        {
            "k": 1,
            "l": 0.01 * undying_coeff_mult,
            "l_mid_neg": 0.005 * undying_coeff_mult,
            "l_low_pos": 0.005 * undying_coeff_mult,
            "l_low_neg": 0.002 * undying_coeff_mult,
            # "l": cfg.resampler_cfg.dead_threshold,
            # "l_mid_neg": cfg.resampler_cfg.dead_threshold / 2,
            # "l_low_pos": cfg.resampler_cfg.dead_threshold / 2,
            # "l_low_neg": cfg.resampler_cfg.dead_threshold / 4,
        },
    ),
    # bias_decay=0.995,
    grad_for_bias=True,
    # weight_decay=0.997,
    grad_to_max_act_decs=False,
    alive_thresh_mul=4,
    resample_before_step=True,
    add_to_max_acts=0.0001,
    # max_undying=100,
    wait_to_check_dead=0,
    # add_softmax_to_acts=True,
    # add_to_max_acts=0.001,
    reset_new_dead=True,
    reset_undied=True,
    undying_sq_ema_reset_ratio=3e1,
    undying_bias_sq_ema_reset_ratio=3e1,
    l1_grads_to_undying=0.5,
)
# cfg.freq_tracker_cfg.decay = 0.997
# cfg.resampler_cfg.resample_before_step = True
# if cfg.resampler_cfg.add_to_max_acts is not None:
#     cfg.resampler_cfg.alive_thresh_mul += (
#         1 / cfg.batch_size
#     ) / cfg.resampler_cfg.dead_threshold


fibox = box()


def selective_resampled_layer(
    d_in, d_out, cfg, W_next=None, get_optim_fn=lambda: fibox.x.optim
):
    resampler = SelectiveUndyingResampler(
        cfg=cfg.resampler_cfg, get_optim_fn=get_optim_fn, W_next=W_next
    )
    freq_tracker = EMAFreqTracker(cfg=cfg.freq_tracker_cfg)
    resampled_layer = ComponentLayer(
        cachelayer=CacheLayer.from_dims(
            d_in=d_in,
            d_out=d_out,
            nonlinearity=resampler.nonlinearity,
        ),
        components=[freq_tracker, resampler],
        train_cache=SAETrainCache(),
        eval_cache=SAETrainCache(),
    )
    return resampled_layer, resampler, freq_tracker


def selective_undying_layers(
    cfg, d_in, d_out, middle_dims, get_optim_fn=lambda: fibox.x.optim, d_in_extra=0
):
    dims = list(reversed([d_in] + middle_dims))
    last_layer, resampler, freq_tracker = selective_resampled_layer(
        cfg=cfg,
        d_in=dims[0] + d_in_extra,
        d_out=d_out,
        W_next=None,
        get_optim_fn=get_optim_fn,
    )
    layers = [last_layer]

    print("out", d_out)
    print("in", dims[0])
    for i, (d_lout, d_lin) in enumerate(zip(dims[:-1], dims[1:])):
        d_lin += d_in_extra if i != len(dims) - 2 else 0
        layer, res, freq = selective_resampled_layer(
            cfg=cfg,
            d_in=d_lin,
            d_out=d_lout,
            # W_next=layers[-1].cachelayer.W,
            get_optim_fn=get_optim_fn,
        )
        print("out", d_lout)
        print("in", d_lin)
        layers.append(layer)

    return reversed(layers), resampler, freq_tracker


# cfg.d_in = cfg.d_data * 2
cfg.tied_init = False
MULT_IN = 3
# resampled_layer, last_resampler, last_freq_tracker = selective_resampled_layer(
#     d_in=cfg.d_data * MULT_IN, d_out=cfg.d_dict, cfg=cfg
# )

layers, last_resampler, last_freq_tracker = selective_undying_layers(
    cfg=cfg,
    d_in=cfg.d_data,
    d_out=cfg.d_dict,
    middle_dims=[cfg.d_data * 2, cfg.d_data * 3],
    d_in_extra=cfg.d_data,
)
trainer = SAETrainer(
    cfg=cfg,
    sae=SAEComponentLayer(
        cfg=cfg,
        sae_cachelayer=SAECacheLayer(
            cfg=cfg,
            encoder=ComponentLayer(
                cachelayer=CatSeqForSAE(
                    *layers,
                ),
                #     CacheLayer.from_dims(d_in=cfg.d_data, d_out=cfg.d_data * 2),
                #     CacheLayer.from_dims(
                #         d_in=cfg.d_data * 3, d_out=cfg.d_data * (MULT_IN - 1)
                #     ),
                #     resampled_layer,
                # ),
                components=[RewriteLastLayerOfSeqToParent()],
            ),
            resampler=last_resampler,
            freq_tracker=last_freq_tracker,
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
