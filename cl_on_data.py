from sae.cl_sae import *
from nqgl.sae.scripts.train_hsae import (
    # Buffer,
    load_data,
    get_model,
    load_data,
    HierarchicalAutoEncoderConfig,
    SerializableNonlinearity,
    ForwardOptions,
)
from recons_modified import get_recons_loss
from data.buffer2 import Buffer
from nqgl.mlutils.components.component_layer.resampler.methods.orth_resampler import (
    OrthRankedTopkResamplingConfig,
)
from components.resamplers import (
    QKOrthResampleConfig,
    QueuedSVDResampler,
    QueuedOrthDiffResampler,
    QueuedTopkDiffDecYEncResampler,
    QueuedOrthTopkDiffDecYEncResampler,
)
from sae.config import SAEConfig
from sae.trainer import SAETrainer
from nqgl.mlutils.components.component_layer.freq_tracker import EMAFreqTracker

fp32 = False
torch.set_default_dtype(torch.float32 if fp32 else torch.bfloat16)

legacy_cfg = HierarchicalAutoEncoderConfig(
    site="resid_pre",
    d_data=768,
    model_name="gpt2-small",
    layer=6,
    gram_shmidt_trail=512,
    batch_size=1024,
    buffer_mult=2048,
    buffer_refresh_ratio=0.48,
    flatten_heads=False,
    buffer_dtype="bf16" if not fp32 else "fp32",
    enc_dtype="bf16" if not fp32 else "fp32",
    device="cuda",
)


# Actual configs
device = "cuda"
batch_size = legacy_cfg.batch_size
dict_mult = 16
sae_cfg = SAEConfig(
    lr=1e-3,
    betas=(0.8, 0.99),
    d_data=768,
    d_dict=int(768 * dict_mult),
    batch_size=batch_size,
    resampler_cfg=QKOrthResampleConfig(
        dead_threshold=1e-6,
        # dead_threshold=1 / (768 * dict_mult) / 300,
        min_viable_count=4_000 * batch_size,
        reset_all_freqs_interval=10_000,
        reset_all_freqs_offset=5_000,
        check_frequency=500,
        resample_frequency=64,
        num_to_resample=16,
        resample_top_k=32,
        normalized_encoder_multiplier=0.003,
        resampling_cycle=(10000, 20000),
        append_to_queue=False,
        # gram_schmidt_trail=,
        negative_bias_multiplier=20,
        sq_ema_reset_ratio=1,
        bias_sq_ema_reset_ratio=1,
    ),
    freq_tracker_cfg=FreqTrackerCombinedConfig(decay=0.99, initial_freq_value=3e-5),
    device=device,
    optim="nadam",
    b_enc_init=-3,
    start_from_dead=False,
    bias_lr_coeff=3,
    # l2_loss_type="l2_norm",
    # l2_loss_type=["l2_norm_squared/40", "l2_norm", "l2_root"],
    # l2_loss_type=["l2_norm_squared/40", "squared/40", "l2_norm"],
    l2_loss_type="squared/40",
    # l2_loss_type="l2_norm_squared/40",
    # l1_coeff=1 / (10),
    l1_coeff=1 / (10),
    l0l1_coeff=1 / 10000,
    l0l1_thresh=20,
)

trainer = SAETrainer(
    sae_cfg,
    # resampler_factory=QueuedTopkDiffResampler,
    resampler_factory=QueuedTopkDiffResampler,
    # resampler_factory=QueuedOrthDiffResampler,
    # resampler_factory=QueuedSVDResampler,
    # resampler_factory=QueuedTopkDiffDecYEncResampler,
    # resampler_factory=QueuedOrthTopkDiffDecYEncResampler,
    freq_tracker_factory=EMAFreqTracker,
)

dataset = "apollo-research/sae-Skylion007-openwebtext-tokenizer-gpt2"
dataset = "monology/pile-uncopyrighted"
dataset = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
seq_mul = 2
# Data
from typing import Tuple


@dataclass
class TrainConfig:
    dataset_name: str
    train_split: Tuple[int, int]
    val_split: Tuple[int, int]
    test_split: Tuple[int, int]


model = get_model(legacy_cfg)
train_percent = 5
train_start = 5
all_tokens = load_data(
    model,
    dataset=dataset,
    name=legacy_cfg.model_name,
    split=f"train[{train_start}%:{train_start+train_percent}%]",
    front_only=False,
    seq_len=128,
    seq_mul=seq_mul,
)  # .cuda()
val_tokens = load_data(
    model,
    dataset=dataset,
    name=legacy_cfg.model_name,
    split=f"train[90%:91%]",
    front_only=False,
    seq_len=128,
    seq_mul=seq_mul,
)  # .cuda()


def get_buffer():
    buf = Buffer(legacy_cfg, all_tokens, model=model)
    buf.freshen_buffer(4, True)
    return buf


def train(trainer: SAETrainer, data_source):

    sae_cfg.architecture = trainer.sae.cachelayer.encoder.component_architectures()

    wandb.init(entity="sae_all", project="2.0_gpt2++", config=sae_cfg)
    # trainer.sae.cachelayer.encoder.cachelayer.W.data[:] = (
    #     trainer.sae.cachelayer.encoder.cachelayer.W.data[:] / 3
    # )

    def end_resampling(cache):
        if trainer.t == 70000 * train_percent * 1024 // batch_size:
            trainer.cfg.resampler_cfg.resample_frequency = 100000
        # if trainer.t == 1001:
        #     trainer.sae.cachelayer.zero(0)
        # if trainer.t == 2001:
        #     trainer.sae.cachelayer.zero(1e-9)

    def warmup(params=["lr"], T=10_000, m=100, exp=1, delay=0):
        def warmup(cache):
            v = (T // m) ** 2
            t = trainer.t - delay
            for param in params:
                if trainer.t == 1:
                    setattr(trainer.cfg, param, getattr(trainer.cfg, param) / (v**exp))
                    if param == "lr":
                        trainer.update_optim_lrs()
                elif t < T and t % m == 1 and trainer.t > delay:
                    i = t // m
                    mul = ((i + 1) / i) ** 2
                    setattr(
                        trainer.cfg, param, getattr(trainer.cfg, param) * (mul**exp)
                    )
                    if param == "lr":
                        trainer.update_optim_lrs()

        return warmup

    def schedule_lr(cache):
        # return
        if (trainer.t + 7000) % 35000 == 0 and trainer.t > 9000:
            trainer.cfg.lr = max(trainer.cfg.lr * (1 / 2), 3e-5)
            # trainer.init_optim()
            trainer.update_optim_lrs()

    def target_l0_with_l1(target_l0, margin=1, start_t=10_000, up=1.0001, down=0.9998):
        def callback(cache):
            if trainer.t >= start_t and trainer.t % 100 == 0:
                if cache["encoder"].l0 > target_l0 + margin:
                    trainer.cfg.l1_coeff = (
                        trainer.cfg.l1_coeff
                        * up ** (cache["encoder"].l0 - target_l0).item()
                    )
                    if trainer.cfg.l0l1_coeff is not None:
                        trainer.cfg.l0l1_coeff = trainer.cfg.l0l1_coeff * up ** (
                            cache["encoder"].l0 - target_l0
                        )
                elif cache["encoder"].l0 < target_l0 - margin:
                    trainer.cfg.l1_coeff = trainer.cfg.l1_coeff * down ** (
                        target_l0 - cache["encoder"].l0
                    )
                    if trainer.cfg.l0l1_coeff is not None:
                        trainer.cfg.l0l1_coeff = trainer.cfg.l0l1_coeff * down ** (
                            target_l0 - cache["encoder"].l0
                        )

        return callback

    def schedule_l0l1(cache):
        # return
        if (
            trainer.t >= 4000
            and trainer.cfg.l0l1_coeff is not None
            and trainer.t % 100 == 0
        ):
            trainer.cfg.l0l1_coeff = 1 / 100
            i = trainer.t // 400
            if trainer.t >= 55000:
                i = 55000 // 400 + (trainer.t - 55000) / 2000
            threshold = max(32, 200 - i)
            trainer.cfg.l0l1_thresh = threshold
            if threshold <= 70:
                trainer.cfg.l1_coeff = trainer.cfg.l1_coeff * 0.98

        if (trainer.t + 2000) % 5000 == 0:
            trainer.cfg.lr = trainer.cfg.lr * 2 / 3
            trainer.update_optim_lrs()
        # if trainer.t == 75000:
        #     trainer.cfg.lr = 3e-5
        #     trainer.init_optim()

    def extra_logging(cache):
        if trainer.t % 1000 == 0:
            wandb.log(
                {
                    **{
                        "num_steps": trainer.t,
                    },
                    **{
                        ("recons/" + k): v
                        for k, v in get_recons_loss(
                            model,
                            trainer.sae,
                            buffer=None,
                            all_tokens=val_tokens,
                            cfg=legacy_cfg,
                        ).items()
                    },
                },
                step=trainer.t,
            )
        if trainer.t % 5000 == 0:
            wandb.log(
                {
                    ("recons/with_proc_bos/" + k): v
                    for k, v in get_recons_loss(
                        model,
                        trainer.sae,
                        buffer=None,
                        all_tokens=val_tokens,
                        cfg=legacy_cfg,
                        bos_processed_with_hook=True,
                    ).items()
                },
                step=trainer.t,
            )

            # y_norms = cache.y.norm(dim=-1)
            # x_norms = cache["encoder"].x.norm(dim=-1)
            # wandb.log(
            #     {
            #         "norms/y_norms": {
            #             "mean": y_norms.mean(),
            #             "max": y_norms.max(),
            #             "min": y_norms.min(),
            #         },
            #         "norms/x_norms": {
            #             "mean": x_norms.mean(),
            #             "max": x_norms.max(),
            #             "min": x_norms.min(),
            #         },
            #     },
            #     step=trainer.t,
            # )

    trainer.extra_calls += [
        extra_logging,
        end_resampling,
        # schedule_l0l1,
        schedule_lr,
        target_l0_with_l1(30, margin=0),
        warmup(["lr"], 10_000, 100, 1),
        warmup(["l1_coeff"], 4000, 100, 0.5),
        warmup(["l0l1_coeff"], 5_000, 100, 1),
    ]

    trainer.train(data_source)


def main():
    torch.set_default_dtype(torch.float32 if fp32 else torch.bfloat16)
    buffer = get_buffer()

    def train_buffer():
        # for i in tqdm.tqdm(range(12000)):
        #     yield buffer.next()

        for i in tqdm.tqdm(range(90000 * train_percent * 1024 // batch_size)):
            yield buffer.next()

    # trainer.sae.cachelayer.zero(1)
    # from stored_acts_buffer import ac

    # trainer.train(ac.read_as_iter(1024))
    # with torch.cuda.amp.autocast():
    train(trainer, train_buffer())


if __name__ == "__main__":
    main()
