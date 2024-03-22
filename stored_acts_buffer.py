from nqgl.sae.training.setup_utils import load_data, get_model
from nqgl.sae.hsae.hsae import HierarchicalAutoEncoderConfig, HierarchicalAutoEncoder
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import torch
import pathlib

fp32 = True
DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass
class ActsConfig:
    start_percent: Optional[int]
    end_percent: Optional[int]
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    model_name: str = "gpt2-small"
    layer_num: int = 6
    site_name: str = "resid_pre"
    dtype: str = "fp32"
    front_only: bool = False
    buffer_refresh_ratio: float = 0.5
    d_data: int = 768
    max_chunk_size_mb: int = 512
    exclude_bos_acts: bool = True

    def chunk_name(self, chunk_num: int):
        return self.folder_name() + f"/chunk{chunk_num}.pt"

    def folder_name(self):
        return (
            f"saved_acts/{self.dataset.replace('/', '_')}/{self.model_name}"
            + f"/layer{self.layer_num}/{self.site_name}"
            + f"/{self.start_percent}-{self.end_percent}/{self.dtype}"
            + f"excl_bos_{self.exclude_bos_acts}"
        )

    def path(self, chunk_num: Optional[int] = None):
        p = Path.home() / "workspace" / self.folder_name()
        if chunk_num is not None:
            p /= f"chunk{chunk_num}.pt"
        return p

    def cfg_path(self):
        return self.path() / "config.json"

    def save_chunk(self, chunk, chunk_num: int):
        folder = self.path()
        folder.mkdir(parents=True, exist_ok=True)
        if self.cfg_path().exists():
            assert self.cfg_path().read_text() == str(self)
        else:
            self.cfg_path().write_text(str(self))
        path = self.path(chunk_num)
        assert not path.exists()
        torch.save(chunk, path)

    def read_chunk(self, chunk) -> torch.Tensor:
        assert self.cfg_path().exists()
        assert self.cfg_path().read_text() == str(self)
        return torch.load(self.path(chunk), map_location="cuda")

    def get_tensor_len(self):
        bytes_per_element = 4 if self.dtype == "fp32" else 2
        bytes_per_act = self.d_data * bytes_per_element
        max_chunk_bytes = 1024**2 * self.max_chunk_size_mb
        return max_chunk_bytes // bytes_per_act

    def get_tensor_to_fill(self, batch_size: int, device="cuda"):
        tlen = self.get_tensor_len() // batch_size * batch_size
        return torch.zeros(tlen, self.d_data, dtype=DTYPES[self.dtype], device=device)

    def read_as_iter(self, batch_size):
        chunk_num = 0
        next_chunk = self.read_chunk(chunk_num)
        tqdm_iter = tqdm.trange(5000 * len(next_chunk) // batch_size)
        while True:
            for i in range(0, len(next_chunk), batch_size):
                tqdm_iter.update(1)
                yield next_chunk[i : i + batch_size]
            chunk_num += 1
            try:
                print("loading chunk", chunk_num)
                next_chunk = self.read_chunk(chunk_num)
                print("loaded chunk", chunk_num, next_chunk.shape)
            except FileNotFoundError:
                break

    @torch.no_grad()
    def read_as_iter_no_bos(self, batch_size):
        chunk_num = 0
        bos_example = None
        next_chunk = self.read_chunk(chunk_num)
        tqdm_iter = tqdm.trange(5000 * len(next_chunk) // batch_size)
        while True:
            if bos_example is None:
                norms = next_chunk.norm(dim=-1)
                i = norms.argmax()
                mm = norms[i]
                if 3100 > mm > 3000:
                    bos_example = next_chunk[i]
                    assert bos_example.ndim == 1
            cheq_okay = (
                next_chunk[:, :200] - bos_example[:200].unsqueeze(0) > 1e-5
            ).any(dim=-1)
            next_chunk = next_chunk[cheq_okay]
            for i in range(0, len(next_chunk), batch_size):
                tqdm_iter.update(1)
                if i + batch_size > len(next_chunk):
                    break
                yield next_chunk[i : i + batch_size]
            chunk_num += 1
            try:
                # print("loading chunk", chunk_num)
                next_chunk = self.read_chunk(chunk_num)
                # print("loaded chunk", chunk_num, next_chunk.shape)
            except FileNotFoundError:
                break


import tqdm


def store_acts(ac: ActsConfig, batch_size=2048, buffer_mult=2048):
    hcfg = HierarchicalAutoEncoderConfig(
        site=ac.site_name,
        d_data=ac.d_data,
        model_name=ac.model_name,
        layer=ac.layer_num,
        # gram_shmidt_trail=512,
        batch_size=batch_size,
        buffer_mult=buffer_mult,
        buffer_refresh_ratio=ac.buffer_refresh_ratio,
        flatten_heads=False,
        buffer_dtype=ac.dtype,
        enc_dtype=ac.dtype,
        device="cuda",
        # buffer_batch_divisor=4,
    )
    model = get_model(hcfg)
    all_tokens = load_data(
        model,
        dataset=ac.dataset,
        # dataset="stas/openwebtext-10k",
        name=hcfg.model_name,
        split=f"train[{ac.start_percent}%:{ac.end_percent}%]",
        front_only=False,
        seq_mul=2,
    )  # .cuda()
    if ac.exclude_bos_acts:
        from data.buffer2 import Buffer
    else:
        from nqgl.sae.training.buffer import Buffer
    buffer = Buffer(hcfg, all_tokens, model)
    buffer.freshen_buffer(4, True)
    # chunk_num = 0
    num_chunks = all_tokens.numel() // ac.get_tensor_len()
    overflow_lost = all_tokens.numel() % ac.get_tensor_len()
    print(f"num_chunks={num_chunks}, overflow_lost={overflow_lost}")
    print(f"\n Will use {num_chunks * ac.max_chunk_size_mb // 1024}GB of disk space.")
    print("chunk_len", ac.get_tensor_len())
    for chunk_num in tqdm.trange(num_chunks + 1):
        chunk = ac.get_tensor_to_fill(batch_size)
        for i in range(0, len(chunk), batch_size):
            # print(i, len(chunk))
            b = buffer.next()
            chunk[i : i + batch_size] = b

        ac.save_chunk(chunk, chunk_num)
        chunk_num += 1
        # del chunk


ac = ActsConfig(
    start_percent=1, end_percent=6, max_chunk_size_mb=1024, buffer_refresh_ratio=0.7
)


ac_small = ActsConfig(
    start_percent=20, end_percent=21, max_chunk_size_mb=1024, buffer_refresh_ratio=0.48
)
ac_mid = ActsConfig(
    start_percent=21, end_percent=23, max_chunk_size_mb=1024, buffer_refresh_ratio=0.24
)
ac_mid_bf16 = ActsConfig(
    start_percent=21,
    end_percent=23,
    max_chunk_size_mb=1024,
    buffer_refresh_ratio=0.24,
    dtype="bf16",
)


def main():
    # ac = ActsConfig(start_percent=0, end_percent=1, max_chunk_size_mb=128)
    store_acts(ac_mid_bf16)


if __name__ == "__main__":
    main()
