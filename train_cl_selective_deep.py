from cl_on_data import *
from deep_selective_undying import trainer
from stored_acts_buffer import ac_small, ac_mid

# torch.set_default_dtype(torch.float32 if fp32 else torch.bfloat16)


def train_buffer():
    # for i in tqdm.tqdm(range(12000)):
    #     yield buffer.next()
    buffer = get_buffer()
    for i in tqdm.tqdm(range(90000 * train_percent * 1024 // batch_size)):
        yield buffer.next()


# trainer.sae.cachelayer.zero(1)
# from stored_acts_buffer import ac

# trainer.train(ac.read_as_iter(1024))
# with torch.cuda.amp.autocast():
# train(trainer, train_buffer())
f = True
buf = ac_mid.read_as_iter(legacy_cfg.batch_size)
train(trainer, buf)
try:
    next(buf)
    f = False
    try:
        pass
    except:
        pass
    print("switching to non-stored data")
    trainer.train(train_buffer())
except:
    assert f
    print("using non-stored data")
    train(trainer, train_buffer())
# try:
#     train(trainer, ac_small.read_as_iter(legacy_cfg.batch_size))
#     print("switching to next stored data")

#     trainer.train(ac_mid.read_as_iter(legacy_cfg.batch_size))
#     print("switching to non-stored data")

#     trainer.train(train_buffer())
# except:
#     try:
#         print("switching to non-stored data")
#         trainer.train(train_buffer())
#     except:
#         print("switching to non-stored data")
