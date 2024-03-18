# %%

from cl_on_data import *

weight_params = []
bias_params = []
for name, param in trainer.sae.named_parameters():
    if name.endswith(".b_dec") or name.endswith(".b"):
        bias_params.append(param)
    else:
        print(f"Skipping {name}")


# %%
bias_params
# %%
p = trainer.sae.parameters()
sp = set(p)
# %%
sb = set(bias_params)

len(sp)

# %%
len(sb)
# %%
len(sp - sb)
# %%
from deep_sae import trainer

# %%
p = trainer.sae.parameters()

# %%
len(set(p))
# %%
