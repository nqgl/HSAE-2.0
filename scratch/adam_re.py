# %%

from nqgl.mlutils.components.component_layer.resampler.adam_resetter import AdamResetter
import torch


class OOO: ...


o = OOO()

p = torch.nn.Parameter(torch.randn(10, 10))
b = torch.nn.Parameter(torch.randn(10, 10))
o.p = p
o.b = b
adam = torch.optim.NAdam([p, b], lr=1e-3)


# %%
l = (p * b).sum()
l.backward()
adam.step()
adam.state

# %%
alive = torch.tensor([0, 1, 2, 3, 9])
alive = torch.zeros_like(p, dtype=torch.bool).scatter_(0, alive.unsqueeze(-1), 1)
adamresetter = AdamResetter(o)

adamresetter.p[torch.tensor([4, 5, 6])](adam, alive_indices=alive)
adam.state

# %%
