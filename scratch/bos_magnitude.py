# %%
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import torch

# %%
g = HookedTransformer.from_pretrained("gpt2")
tokenizer = g.tokenizer

h = tokenizer("hello world my name is a", return_tensors="pt")
# %%
t = torch.tensor([tokenizer.bos_token_id, 31373, 995], dtype=torch.long).unsqueeze(0)
# %%
g(t)
# %%
logits, cache = g.run_with_cache(t)
# %%
acts = cache[get_act_name("resid_pre", 6)]
acts = acts.squeeze(0)
acts.norm(dim=-1)

# %%

tokenizer.bos_token_id

# %%
t2 = torch.tensor([tokenizer.bos_token_id, 31373, 995], dtype=torch.long).unsqueeze(0)
logits, cache = g.run_with_cache(t2)
acts = cache[get_act_name("resid_pre", 6)]
acts = acts.squeeze(0)
acts.norm(dim=-1)

acts @ acts.t()
# %%
p1 = torch.nn.Parameter(torch.tensor([1.0]))
p2 = torch.nn.Parameter(torch.tensor([1.0]))

optim = torch.optim.Adam(
    [{"params": p1, "lr": 0.01, "name": "p"}, {"params": p2, "lr": 0.02, "name": "p"}]
)

# %%
l = p1 + p2
l.backward()
optim.step()

# %%
t = tokenizer(
    "The President of the United States of America", return_tensors="pt"
).input_ids
# t = torch.cat((t, torch.randint(0, 40000, t.shape, dtype=torch.int64)), dim=0)
t = torch.cat(
    (
        torch.tensor([tokenizer.bos_token_id], dtype=torch.int64)
        .unsqueeze(0)
        .expand(1, 1),
        t,
    ),
    dim=1,
)
# t = t.


loss = g(t, return_type="loss")
logits = g(t)
loss
losses = g(t, return_type="loss", loss_per_token=True)
losses.shape
# loss


# %%
loss
# %%
losses
# %%b
tokenizer.bos_token_id
# %%
