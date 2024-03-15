import torch
from functools import partial
import tqdm
import einops
from nqgl.sae.training.setup_utils import DTYPES


@torch.no_grad()
def get_recons_loss(
    model, encoder, buffer, num_batches=5, local_encoder=None, cfg=None
):
    cfg = cfg or encoder.cfg
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        tokens = buffer.all_tokens[
            torch.randperm(len(buffer.all_tokens))[: max(cfg.model_batch_size // 16, 1)]
        ]
        assert torch.all(50256 == tokens[:, 0])
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[
                (
                    cfg.act_name,
                    partial(replacement_hook, encoder=local_encoder, cfg=cfg),
                )
            ],
        )
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg.act_name, mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(cfg.act_name, zero_ablate_hook)],
        )
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss


def replacement_hook(acts, hook, encoder, cfg):
    acts_shape = acts.shape
    acts_re = acts.reshape(-1, cfg.act_size)
    mlp_post_reconstr = encoder(acts_re.reshape(-1, cfg.act_size))

    mlp_post_reconstr = mlp_post_reconstr.reshape(acts_shape)
    seq_len = acts_shape[1]
    assert seq_len == 128
    return torch.cat((acts[:, :1], mlp_post_reconstr[:, 1:]), dim=1)


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:, :] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:, :] = 0.0
    return mlp_post
