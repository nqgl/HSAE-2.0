from cl_sae import *

from nqgl.sae.scripts.train_hsae import (
    Buffer,
    load_data,
    get_model,
    load_data,
    HierarchicalAutoEncoderConfig,
    SerializableNonlinearity,
    ForwardOptions,
    get_recons_loss,
)

fp32 = True

cfg = HierarchicalAutoEncoderConfig(
    site="resid_pre",
    # d_data=512,
    # layer=1,
    d_data=768,
    model_name="gpt2-small",
    layer=6,
    gram_shmidt_trail=512,
    # num_to_resample=4,
    # l1_coeff=3e-1,
    # dict_mult=1,
    batch_size=1024,
    buffer_mult=1024,
    buffer_refresh_ratio=0.5,
    # beta1=0.0,
    # beta2=0.99,
    # nonlinearity=SerializableNonlinearity("relu"),
    flatten_heads=False,
    # layer_cfg_params={
    #     "l1_coeff": 1e-1,
    #     "data_rescale": 1,
    #     "use_b_dec": False,
    #     "forward_options": ForwardOptions(
    #         sub_b_dec=False,
    #         scale_acts=True,
    #         use_b_enc_const=True,
    #         cache_before_scaling=False,
    #     ),
    # },
    buffer_dtype="bf16" if not fp32 else "fp32",
    enc_dtype="bf16" if not fp32 else "fp32",
    device="cuda",
    # subshuffle=4,
    # buffer_batch_divisor=2,
)


@dataclass
class QKResampleConfig(QueuedResamplerConfig):
    resample_top_k: int = 256


def main():
    torch.set_default_dtype(torch.float32 if fp32 else torch.bfloat16)
    model = get_model(cfg)
    train_percent = 5
    all_tokens = load_data(
        model,
        dataset="apollo-research/sae-Skylion007-openwebtext-tokenizer-gpt2",
        # dataset="stas/openwebtext-10k",
        name=cfg.model_name,
        split=f"train[5%:{5+train_percent}%]",
        front_only=False,
    ).cuda()
    buffer = Buffer(cfg, all_tokens, model=model)
    # sae = SAECacheLayer(sae_cfg)
    device = "cuda"
    batch_size = cfg.batch_size
    dict_mult = 8
    sae_cfg = SAEConfig(
        lr=1e-3,
        betas=(0.9, 0.9994),
        d_data=768,
        d_dict=768 * dict_mult,
        resampler_cfg=QKResampleConfig(
            dead_threshold=3e-6,
            # dead_threshold=1 / (768 * dict_mult) / 300,
            min_viable_count=4_000 * batch_size,
            reset_all_freqs_interval=10_000,
            reset_all_freqs_offset=5_000,
            check_frequency=500,
            resample_frequency=5,
            num_to_resample=1,
            resample_top_k=128,
            normalized_encoder_multiplier=0.1,
            resampling_cycle=(40_000, 50_000),
            append_to_queue=False,
        ),
        freq_tracker_cfg=FreqTrackerCombinedConfig(decay=0.9993),
        l1_coeff=1 / (20),
        device=device,
    )
    wandb.init(entity="sae_all", project="saes_2.0_gpt2", config=sae_cfg)

    # vecs = torch.eye(768)
    trainer = SAETrainer(
        sae_cfg,
        # resampler_factory=QueuedTopkDiffResampler,
        resampler_factory=QueuedTopkDiffResampler,
        freq_tracker_factory=EMAFreqTracker,
    )
    print([i for i in trainer.sae.named_modules()])
    trainer.sae.cachelayer.encoder.cachelayer.W.data[:] = (
        trainer.sae.cachelayer.encoder.cachelayer.W.data[:] / 3
    )

    def end_resampling():
        if trainer.t > 70000 * train_percent * 1024 // batch_size:
            trainer.cfg.resampler_cfg.resample_frequency = 100000
        # if trainer.t == 1001:
        #     trainer.sae.cachelayer.zero(0)
        # if trainer.t == 2001:
        #     trainer.sae.cachelayer.zero(1e-9)

    trainer.extra_calls += [
        lambda: wandb.log(
            {"recons_score": get_recons_loss(model, trainer.sae, buffer, cfg=cfg)[0]},
            step=trainer.t,
        ),
        end_resampling,
    ]

    def train_buffer():
        for i in tqdm.tqdm(range(90000 * train_percent * 1024 // batch_size)):
            yield buffer.next()

    # trainer.sae.cachelayer.zero(1)
    trainer.train(train_buffer())


if __name__ == "__main__":
    main()
