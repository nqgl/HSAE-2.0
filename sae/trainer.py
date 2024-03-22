from nqgl.mlutils.components.cache import Cache
from sae.cl_sae import SAEComponentLayer
from sae.config import SAEConfig, SAETrainCache


import torch
import wandb


class SAETrainer:
    def __init__(
        self,
        cfg: SAEConfig,
        resampler_factory=None,
        # sae: SAEComponentLayer,
        freq_tracker_factory=None,
        other_encoder_components=[],
        sae: SAEComponentLayer = None,
        sae_cachelayer=None,
    ):
        assert sae is None or sae_cachelayer is None
        self.cfg = cfg
        if not sae:
            resampler_factory = resampler_factory.bind_named_args(
                get_optim_fn=lambda: self.optim
            )
        self.sae = (
            sae
            or SAEComponentLayer(
                cfg,
                resampler_factory=resampler_factory,
                freq_tracker_factory=freq_tracker_factory,
                sae_cachelayer=sae_cachelayer,
            )
        ).to(cfg.device)
        self.t = 1
        self.extra_calls = []
        self.init_optim()

    def init_optim(self):
        if self.cfg.optim == "adam":
            self.optim = torch.optim.Adam(
                self.parameters(), lr=self.cfg.lr, betas=self.cfg.betas
            )
        elif self.cfg.optim == "radam":
            self.optim = torch.optim.RAdam(
                self.parameters(), lr=self.cfg.lr, betas=self.cfg.betas
            )
        elif self.cfg.optim == "sgd":
            self.optim = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.lr,
                momentum=self.cfg.betas[0],
                nesterov=True,
            )
        elif self.cfg.optim == "nadam":
            self.optim = torch.optim.NAdam(
                self.parameters(), lr=self.cfg.lr, betas=self.cfg.betas
            )

    def update_optim_lrs(self):
        for pg in self.optim.param_groups:
            if pg["name"] == "bias":
                pg["lr"] = self.cfg.lr * self.cfg.bias_lr_coeff
            elif pg["name"] == "weight":
                pg["lr"] = self.cfg.lr
            else:
                raise ValueError(f"param group name {pg['name']} not recognized")

    def parameters(self):
        bias_params = []
        for name, param in self.sae.named_parameters():
            if name.endswith(".b_dec") or name.endswith(".b") or name.endswith(".bias"):
                bias_params.append(param)

        weights = set(self.sae.parameters()) - set(bias_params)
        groups = [
            {
                "params": bias_params,
                "lr": self.cfg.lr * self.cfg.bias_lr_coeff,
                "name": "bias",
            },
            {"params": list(weights), "lr": self.cfg.lr, "name": "weight"},
        ]
        return groups

    def train(self, buffer):
        l1_coeff = self.cfg.l1_coeff
        if self.cfg.start_from_dead:
            with torch.no_grad():
                self.sae.cachelayer.zero()
                self.sae.cachelayer.encoder.cachelayer.W[0, 0] = torch.randn_like(
                    self.sae.cachelayer.encoder.cachelayer.W[0, 0]
                )
        self.norm_dec()
        for bn in buffer:
            try:
                x, y = bn
                assert isinstance(bn, tuple)
            except:
                y = (x := bn)
                assert isinstance(bn, torch.Tensor)
            # in the future trainer could itself be a CacheProcLayer maybe
            cache: SAETrainCache = self.sae.generate_default_cache()
            y_pred = self.sae(x, cache=cache)
            cache.y = ...
            cache.y = y
            self.step(cache=cache, x=x, y_pred=y_pred, y=x)
            self.full_log(cache)
            self.t += 1

    def step(self, cache: SAETrainCache, x, y_pred, y=None):
        loss = self.loss(cache)
        cache.loss = loss
        loss.backward()
        resample_before_step = getattr(
            self.cfg.resampler_cfg, "resample_before_step", False
        )
        if resample_before_step:
            for c in cache.search("resample"):
                c.num_resampled = ...
                c.resample(x=x)
        self.norm_dec_grads()
        self.optim.step()
        self.optim.zero_grad()
        self.norm_dec()
        if not resample_before_step:
            for c in cache.search("resample"):
                c.num_resampled = ...
                c.resample(x=x)
        for call in self.extra_calls:
            call(cache)

    @torch.no_grad()
    def norm_dec_grads(self):
        grad = self.sae.cachelayer.decoder.weight.grad
        dec_normed = (
            self.sae.cachelayer.decoder.weight.data
            / self.sae.cachelayer.decoder.weight.data.norm(dim=0, keepdim=True)
        )
        grad_orth = grad - (dec_normed * grad).sum(0, keepdim=True) * dec_normed
        self.sae.cachelayer.decoder.weight.grad[:] = grad_orth

    def norm_dec(self):
        if self.t % 1000 == 0:
            wandb.log(
                {
                    "dec_norms": self.sae.cachelayer.decoder.weight.norm(dim=0)
                    .max()
                    .item()
                },
                step=self.t,
            )

        self.sae.cachelayer.decoder.weight.data[:] = (
            self.sae.cachelayer.decoder.weight.data
            / self.sae.cachelayer.decoder.weight.data.norm(dim=0, keepdim=True)
        )

    def loss(self, cache: SAETrainCache):
        return (
            self.get_l2_type(cache, self.cfg.l2_loss_type)
            + cache["encoder"].l1 * self.cfg.l1_coeff
            + (
                cache["encoder"].l0l1 * self.cfg.l0l1_coeff
                if self.cfg.l0l1_coeff and self.cfg.l0l1_coeff != 0
                else 0
            )
        )

    def get_l2_type(self, cache, l2_type):
        if isinstance(l2_type, str):
            if l2_type == "squared/40":
                return cache.l2 / 40
            elif l2_type == "l2_norm":
                return cache.l2_norm
            elif l2_type == "l2_root":
                return cache.l2**0.5
            elif l2_type == "l2_norm_squared/40":
                return cache.l2_norm**2 / 40
            else:
                raise ValueError(f"l2_type {l2_type} not recognized")
        else:
            v = 0
            for l2 in l2_type:
                v = v + self.get_l2_type(cache, l2)
            return v / len(l2_type)

    def full_log(self, cache: Cache):
        if self.t % 10 != 0:
            return
        d = cache.logdict(
            excluded=[
                "acts",
                "y_pred",
                "x",
                "y",
                "resample",
                "nonlinear_argsmaxed",
                "acts_spoof",
            ]
        )
        if wandb.run is not None:
            wandb.log(d, step=self.t)
