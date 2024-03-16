from nqgl.sae.sae.config import AutoEncoderConfig
from nqgl.sae.training.setup_utils import DTYPES
from transformer_lens import HookedTransformer

import einops
import torch


import time

# I might come back to this and think about changing refresh ratio up
# also is there a pipelining efficiency we could add?
# is it bad to have like 2 gb of tokens in memory?


class Buffer:
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder.
    It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg: AutoEncoderConfig, tokens, model, dont_shuffle=False):
        self.buffer = torch.zeros(
            (cfg.buffer_size, cfg.d_data),
            dtype=DTYPES[cfg.buffer_dtype],
            requires_grad=False,
            device=cfg.device,
        )
        self.cfg: AutoEncoderConfig = cfg
        self.token_pointer = 0
        self.first = True
        self.all_tokens = tokens
        self.model: HookedTransformer = model
        self.time_shuffling = 0
        self.dont_shuffle = dont_shuffle
        self.perm = torch.randperm(self.buffer.shape[0])
        self.prevperm = None
        self.perm_i = 0
        self.refresh()

    def end_refresh(self):
        self.perm_i += int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio)
        if (
            self.perm_i
            + (
                int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio)
                - self.cfg.batch_size
            )
            >= self.buffer.shape[0]
        ):
            print("Resetting the perm")
            self.perm = torch.randperm(self.buffer.shape[0])
            self.perm_i = 0

    def getperm(self, i, size):
        return self.perm[self.perm_i + i : self.perm_i + i + size]

    def nextperm(self, size):
        perm = self.getperm(self.pointer, size)
        self.pointer += size
        return perm

    @torch.inference_mode()
    def refresh(self):
        """
        Refreshes the buffer by populating it with new activations, then shuffles it.

        Note: This method assumes that the necessary attributes and configurations are already set.
        """
        t0 = time.time()
        self.pointer = 0
        with torch.autocast("cuda", DTYPES[self.cfg.buffer_dtype]):
            if self.first:
                num_batches = self.cfg.buffer_batches
            else:
                num_batches = (
                    int(self.cfg.buffer_batches * self.cfg.buffer_refresh_ratio) + 1
                )
            self.first = False
            for _ in range(0, num_batches + 1, self.cfg.model_batch_size):
                tokens = self.all_tokens[
                    self.token_pointer : self.token_pointer + self.cfg.model_batch_size
                ]
                # _, cache = self.model.run_with_cache(
                #     tokens, stop_at_layer=self.cfg.layer + 1
                # )

                l = []

                def hook_fn(acts, hook):
                    l.append(acts)

                self.model.run_with_hooks(
                    tokens,
                    stop_at_layer=self.cfg.layer + 1,
                    fwd_hooks=[(self.cfg.act_name, hook_fn)],
                )
                assert len(l) == 1

                if self.cfg.flatten_heads:
                    acts = einops.rearrange(
                        l[0],
                        "batch seq_pos n_head d_head -> (batch seq_pos) (n_head d_head)",
                    )
                else:
                    acts_no_re = l[0][:, 1:]
                    assert torch.all(l[0][:, 0, :] - l[0][0, 0, :] < 1e-5), (
                        l[0][:, 0, :] - l[0][0, 0, :]
                    ).max()
                    acts = einops.rearrange(
                        acts_no_re,
                        "batch seq_pos d_act -> (batch seq_pos) d_act",
                    )
                assert acts.shape[-1] == self.cfg.d_data
                self.buffer[self.nextperm(acts.shape[0])] = acts
                self.token_pointer += self.cfg.model_batch_size
            assert self.pointer + self.perm_i > int(
                self.buffer.shape[0] * self.cfg.buffer_refresh_ratio
            ), f"Pointer: {self.pointer}, buffer shape: {self.buffer.shape[0]}, buffer refresh ratio: {self.cfg.buffer_refresh_ratio}"
        self.pointer = 0
        assert (
            not self.dont_shuffle
            and not self.cfg.sus_cheap_shuffle
            and not self.cfg.subshuffle
        )
        self.end_refresh()
        assert (
            self.token_pointer < self.all_tokens.shape[0]
        ), f"Ran out of tokens! token pointer: {self.token_pointer}, all tokens: {self.all_tokens.shape[0]}"
        self.time_shuffling += time.time() - t0
        # torch.cuda.empty_cache()

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.nextperm(self.cfg.batch_size)]
        if (
            self.pointer
            > int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio)
            - self.cfg.batch_size
        ):
            print("Refreshing the buffer!")
            self.refresh()

        return out

    @torch.no_grad()
    def next_tokens(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg.batch_size]
        return out

    @torch.no_grad()
    def freshen_buffer(self, fresh_factor=1, half_first=True):
        """
        Refreshes the buffer by moving the pointer and calling the refresh method.
        Warning: this burns data

        Args:
            fresh_factor (int): The factor by which the buffer should be refreshed.
            half_first (bool): Whether to refresh half of the buffer first.

        Returns:
            None
        """
        if half_first:
            n = (0.5 * self.cfg.buffer_size) // self.cfg.batch_size
            self.pointer += n * self.cfg.batch_size
            self.refresh()
        n = (
            (self.cfg.buffer_refresh_ratio) * self.cfg.buffer_size
        ) // self.cfg.batch_size
        for _ in range(1 + int(fresh_factor / (self.cfg.buffer_refresh_ratio))):
            self.pointer += (n + 1) * self.cfg.batch_size
            self.refresh()

    @torch.no_grad()
    def skip_first_tokens_ratio(self, skip_percent):
        """
        Fast-forwards through skip_percent proportion of the data
        """
        self.token_pointer += int(self.all_tokens.shape[0] * skip_percent)
        self.first = True
        self.refresh()
