from training.cl_on_data import *
from deep_sae.deep_sae import trainer


def main():
    torch.set_default_dtype(torch.float32)

    # trainer.sae.cachelayer.zero(1)
    from data.stored_acts_buffer import ac

    train(trainer, ac.read_as_iter_no_bos(1024))
    # with torch.cuda.amp.autocast():
    #     trainer.train(train_buffer())


if __name__ == "__main__":
    main()
