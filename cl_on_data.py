from cl_sae import *

from nqgl.sae.scripts.train_hsae import cfg, Buffer, load_data, load_model, 

def main():
    # sae = SAECacheLayer(sae_cfg)
    device = "cuda"
    wandb.init(entity="sae_all", project="saes_2.0_testing")
    batch_size = 1024
    sae_cfg = SAEConfig(
        lr=3e-3,
        betas=(0.5, 0.99),
        d_data=768,
        d_dict=768 * 1,
        resampler_cfg=QueuedResamplerConfig(
            dead_threshold=0.001,
            min_viable_count=500 * batch_size,
            reset_all_freqs_interval=2_000,
            reset_all_freqs_offset=1000,
            check_frequency=2000,
            resample_frequency=5,
            num_to_resample=1,
        ),
        freq_tracker_cfg=CountingFreqTrackerConfig(),
        l1_coeff=3e-3,
        device=device,
    )

    vecs = torch.eye(768)
    trainer = SAETrainer(
        sae_cfg,
        resampler_factory=QueuedTopkDiffResampler,
    )
    print([i for i in trainer.sae.named_modules()])

    m = torch.randn(768 // 4, 768, device=device) / 10
    m = m.repeat(4, 1)
    num_features = 768 * 4
    m = torch.randn(num_features, 768, device=device)
    m = m / m.norm(dim=1, keepdim=True)
    avg_n_features_active = 5

    def buffer():
        for i in range(50000):
            rv = (
                torch.rand(batch_size, num_features, device=device)
                < (avg_n_features_active / num_features)
            ).to(torch.get_default_dtype())
            rv = (
                rv
                * 10
                * (torch.rand(batch_size, num_features, device=device) + 0.5)
                / avg_n_features_active**0.5
            )
            yield rv @ m

    trainer.train(buffer())


if __name__ == "__main__":
    main()
