from cl_sae import SAECache, SAEConfig, SAECacheLayer
from nqgl.mlutils.components import ComponentLayer, CacheModule
import torch
import torch.nn as nn
from typing import List
from dataclasses import dataclass, asdict

from nqgl.mlutils.components.cache import Cache


@dataclass
class SeqSAECatPrev(SAEConfig):
    cache_site: str = "acts"


class SeqSAEWithPrevActs(SAECacheLayer):
    cfg: SeqSAECatPrev

    def forward(self, x, cache: SAECache):
        x = torch.cat((x, getattr(cache._prev_cache, self.cfg.cache_site)), dim=-1)
        return super().forward(x, cache)


class SeqSAEsCL(CacheModule):
    def __init__(self, cfg: SAEConfig, saes: SAECacheLayer = None, n_saes=None):
        self.cfg = cfg
        self.saes = nn.ModuleList(saes)
        if saes is None:
            next_cfg = cfg
            d_in = cfg.d_data
            self.saes = [SAECacheLayer(cfg)]
            d_in += cfg.d_dict if cfg.site not in ("y_pred",) else cfg.d_data
            for i in range(1, n_saes):
                next_cfg = self.cfg.__class__(**{**asdict(self.cfg), "d_data": d_in})
                if i == 0:
                    self.saes.append(SAECacheLayer(cfg))

    def forward(self, x, cache):
        y = 0
        for i, layer in enumerate(self.sae):
            if isinstance(layer, CacheModule):
                y = y + layer(x, cache[i])
            else:
                y = y + layer(x)
        return x


class SequentialCacheLayer(CacheModule):
    def __init__(self, *modules):
        super().__init__()
        self._sequence = nn.ModuleList(modules)

    def forward(self, x, cache: Cache = None, **kwargs):
        for i, module in enumerate(self._sequence):
            if isinstance(module, CacheModule):
                x = module(x, cache=cache[i], **kwargs)
            else:
                x = module(x)
        return x


class CatSeqCacheLayer(CacheModule):
    def __init__(self, *modules):
        super().__init__()
        self._sequence = nn.ModuleList(modules)

    def forward(self, x, cache: Cache = None, **kwargs):
        nx = None
        for i, module in enumerate(self._sequence):
            if isinstance(module, CacheModule):
                nx = module(
                    torch.cat((x, nx), dim=-1) if nx is not None else x,
                    cache=cache[i],
                    **kwargs
                )
            else:
                nx = module(torch.cat((x, nx), dim=-1) if nx is not None else x)
        return nx


class SumSeqCacheLayer(CacheModule):
    def __init__(self, *modules):
        self._sequence = nn.ModuleList(modules)

    def forward(self, *x, cache: Cache = None, **kwargs):
        y_pred = 0
        for i, module in enumerate(self._sequence):
            y_pred = y_pred + module(x, cache=cache[i], **kwargs)
        # cache.y_pred = y_pred
        return y_pred


def main():
    from cl_on_data import sae_cfg

    SeqSAEsCL(
        SAEConfig(10, 20),
        [
            SAECacheLayer(sae_cfg),
            SeqSAEWithPrevActs(),
            SeqSAEWithPrevActs(),
        ],
    )
