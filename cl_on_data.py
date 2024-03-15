from cl_sae import *
from resamplers import QueuedOrthTopkDiffDecYEncResampler
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
from buffer2 import Buffer
from nqgl.mlutils.components.component_layer.resampler.methods.orth_resampler import (
    OrthRankedTopkResamplingConfig,
)
from resamplers import (
    QKOrthResampleConfig,
    QueuedSVDResampler,
    QueuedOrthDiffResampler,
    QueuedTopkDiffDecYEncResampler,
)

fp32 = True

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
dict_mult = 8
sae_cfg = SAEConfig(
    lr=3e-4,
    betas=(0.8, 0.998),
    d_data=768,
    d_dict=int(768 * dict_mult),
    resampler_cfg=QKOrthResampleConfig(
        dead_threshold=3e-6,
        # dead_threshold=1 / (768 * dict_mult) / 300,
        min_viable_count=4_000 * batch_size,
        reset_all_freqs_interval=10_000,
        reset_all_freqs_offset=5_000,
        check_frequency=1000,
        resample_frequency=5,
        num_to_resample=1,
        resample_top_k=128,
        normalized_encoder_multiplier=0.1,
        resampling_cycle=(500, 2000),
        append_to_queue=False,
        # gram_schmidt_trail=,
        negative_bias_multiplier=28,
        sq_ema_reset_ratio=1e-1,
        bias_sq_ema_reset_ratio=1,
    ),
    freq_tracker_cfg=FreqTrackerCombinedConfig(decay=0.9994, initial_freq_value=1e-5),
    device=device,
    optim="adam",
    b_enc_init=-14,
    # start_from_dead=True,
    bias_lr_coeff=1,
    # l2_loss_type="l2_norm",
    # l2_loss_type=["l2_norm_squared/40", "l2_norm", "l2_root"],
    l2_loss_type=["l2_norm_squared/40", "squared/40", "l2_norm"],
    # l1_coeff=1 / (10),
    l1_coeff=1 / (11),
    l0l1_coeff=0.0,
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

# Data
model = get_model(legacy_cfg)
train_percent = 5
all_tokens = load_data(
    model,
    dataset="apollo-research/sae-Skylion007-openwebtext-tokenizer-gpt2",
    name=legacy_cfg.model_name,
    split=f"train[5%:{5+train_percent}%]",
    front_only=False,
    seq_len=128,
)  # .cuda()
buffer = Buffer(legacy_cfg, all_tokens, model=model)


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

    def warmup(cache):
        T = 9000
        m = 100
        v = T // m
        if trainer.t == 1:
            trainer.cfg.l1_coeff /= v**0.5
            trainer.cfg.lr /= v
            if trainer.cfg.l0l1_coeff is not None:
                trainer.cfg.l0l1_coeff /= v
        elif trainer.t < T and trainer.t % m == 1:
            i = trainer.t // m
            mul = (i + 1) / i
            trainer.cfg.l1_coeff *= mul**0.5
            trainer.cfg.lr *= mul
            if trainer.cfg.l0l1_coeff is not None:
                trainer.cfg.l0l1_coeff *= mul

    def schedule_lr(cache):
        # return
        if (trainer.t + 5000) % 20000 == 0 and trainer.t > 10000:
            trainer.cfg.lr = max(trainer.cfg.lr * (1 / 3), 1e-5)
            # trainer.init_optim()
            trainer.update_optim_lrs()

    def target_l0_with_l1(target_l0, margin=1):
        def callback(cache):
            if trainer.t >= 10000 and trainer.t % 100 == 0:
                if cache["encoder"].l0 > target_l0 + margin:
                    trainer.cfg.l1_coeff = trainer.cfg.l1_coeff * 1.0002
                elif cache["encoder"].l0 < target_l0 - margin:
                    trainer.cfg.l1_coeff = trainer.cfg.l1_coeff * 0.9998

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
                    "num_steps": trainer.t,
                    "recons_score": get_recons_loss(
                        model, trainer.sae, buffer, cfg=legacy_cfg
                    )[0],
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
        target_l0_with_l1(40),
        warmup,
    ]

    trainer.train(data_source)


def main():
    torch.set_default_dtype(torch.float32 if fp32 else torch.bfloat16)

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
