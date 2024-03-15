from nqgl.mlutils.components.cache_layer import (
    CacheLayer,
    CacheProcLayer,
    CacheLayerConfig,
)
from nqgl.mlutils.components.component_layer import ComponentLayer, LayerComponent
from nqgl.mlutils.components.old_component_model.act_freq_layer import (
    ActFreqLayer,
)
from nqgl.mlutils.components.old_component_model.freq_component import (
    CountingFreqActComponent,
)
import torch

from nqgl.mlutils.components.component_layer.freq_tracker import ActFreqCLayer
from unpythonic import box

d_in = 110
d_out = 64
inst = (2, 3)
cfg = CacheLayerConfig(d_in=d_in, d_out=d_out, inst=inst)


class ActCountFreqLayer(ActFreqLayer):
    def __init__(self, d_in, d_out, b_in=None, inst=tuple()):
        W = torch.randn(*inst, d_in, d_out)
        b_out = torch.zeros(*inst, d_out)
        b_in = torch.zeros(*inst, d_in)
        cl = CacheLayer(W, b_out, b_in)
        super().__init__({}, cl, CountingFreqActComponent)


# cf = ActCountFreqLayer(d_in, d_out)
ccf = ActFreqCLayer(cfg)

cl = CacheLayer(
    ccf.cachelayer.W,
    ccf.cachelayer.b,
    ccf.cachelayer.b_pre,
)
cf = ActFreqLayer({}, cl, CountingFreqActComponent)

for i in range(100):
    x = torch.randn(80, d_in)
    out = cf(x)
    # out = ccf()
    # print("out:", out)
    # print(out.shape)
    print(out - ccf(x))
    print(cf.activations.freqs - ccf.activations.freqs)
    # print("freq:", cf.activations.freqs)
    # print(cf.activations.freqs.shape)
    # print(cf.activations.freqs.mean())


class SAEConfig:
    lr = 1e-3
    betas = (0.5, 0.99)
    d_data = 768
    d_dict = 768 * 1


class SAE:
    cfg: dict

    def __init__(self, cfg):
        self.cfg = cfg
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.encoder = ComponentLayer(
            cfg,
            cachelayer=CacheProcLayer(CacheLayer.from_dims(cfg.d_data, cfg.d_dict)),
            components=[...],
        )
        self.decoder = torch.nn.Linear(cfg.d_dict, cfg.d_data, bias=False)
        self.b_dec = 0

    def forward(self, x, cache=None):
        return self.decoder(self.encoder(x - self.b_dec, cache=cache)) + self.b_dec

    def train(self, buffer, n_steps):
        for _ in n_steps:
            x = buffer.next()
            b = box()
            y_pred = self(x, b)
            self.step(cache=b.x, x=x, y_pred=y_pred, y=x)

    def step(self, cache, x, y_pred, y=None):
        y = y or x
        loss = self.loss(y_pred, y) + self.penalties(cache)
        loss.backward()
        self.norm_dec_grads()
        self.optim.step()
        if cache.has.resample:
            cache.resample(y)
            # if self.cfg.restep:
            #     self.restep_resample(cache)
        self.norm_dec()

    def log(self):
        pass
