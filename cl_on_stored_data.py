from cl_on_data import *


def main():
    torch.set_default_dtype(torch.float32)

    # trainer.sae.cachelayer.zero(1)
    from stored_acts_buffer import ac

    trainer.train(ac.read_as_iter(1024))
    # with torch.cuda.amp.autocast():
    #     trainer.train(train_buffer())


if __name__ == "__main__":
    main()
