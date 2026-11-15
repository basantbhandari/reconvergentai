"""BilingualDataset and masking utilities."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, Sampler

__all__ = ["BilingualDataset", "LengthBucketSampler", "causal_mask", "collate_trim"]


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds            = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang      = src_lang
        self.tgt_lang      = tgt_lang
        self.seq_len       = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pair     = self.ds[idx]
        src_text = pair["translation"][self.src_lang]
        tgt_text = pair["translation"][self.tgt_lang]

        enc_ids = self.tokenizer_src.encode(src_text).ids
        dec_ids = self.tokenizer_tgt.encode(tgt_text).ids

        enc_padding = self.seq_len - len(enc_ids) - 2
        dec_padding = self.seq_len - len(dec_ids) - 1

        if enc_padding < 0 or dec_padding < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_ids, dtype=torch.int64),
            self.eos_token,
            self.pad_token.expand(enc_padding),
        ])
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_ids, dtype=torch.int64),
            self.pad_token.expand(dec_padding),
        ])
        label = torch.cat([
            torch.tensor(dec_ids, dtype=torch.int64),
            self.eos_token,
            self.pad_token.expand(dec_padding),
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len),
            "label":        label,
            "src_text":     src_text,
            "tgt_text":     tgt_text,
        }


def causal_mask(size: int) -> torch.Tensor:
    return torch.triu(torch.ones(1, size, size), diagonal=1).int() == 0


def collate_trim(items: list[dict]) -> dict:
    """Stack items and drop columns that are padding for every row in the batch.

    BilingualDataset pads to `seq_len` (350) but the median sentence is ~22 tokens.
    Trimming to the batch's longest sentence leaves every real token's position and
    mask unchanged, so the computation is identical, just without the dead padding.
    """
    batch = {k: [it[k] for it in items] for k in items[0]}
    for k in ("encoder_input", "decoder_input", "encoder_mask", "decoder_mask", "label"):
        batch[k] = torch.stack(batch[k])

    # encoder_mask: (B,1,1,S) is 1 on non-pad tokens.
    # decoder_mask: (B,1,T,T) diagonal is 1 exactly on non-pad decoder tokens.
    src_len = int(batch["encoder_mask"].sum(dim=-1).max())
    tgt_len = int(batch["decoder_mask"][:, 0].diagonal(dim1=-2, dim2=-1).sum(dim=-1).max())

    batch["encoder_input"] = batch["encoder_input"][:, :src_len]
    batch["encoder_mask"]  = batch["encoder_mask"][..., :src_len]
    batch["decoder_input"] = batch["decoder_input"][:, :tgt_len]
    batch["label"]         = batch["label"][:, :tgt_len]
    batch["decoder_mask"]  = batch["decoder_mask"][..., :tgt_len, :tgt_len]
    return batch


class LengthBucketSampler(Sampler[list[int]]):
    """Yield batches of indices whose sentences have similar length.

    Indices are shuffled, split into pools of `batch_size * pool_factor`, sorted by
    length inside each pool, cut into batches, and the batch order is shuffled.

    A batch holds at most `batch_size` sentences and at most `max_tokens` padded
    tokens (sentences x longest length), like the paper's token-based batching.
    The token cap keeps memory flat: a batch of 300-token sentences gets ~6 rows
    instead of 32, so peak memory doesn't spike on the rare long sentences.
    """

    def __init__(self, lengths: list[int], batch_size: int, max_tokens: int | None = None,
                 shuffle: bool = True, pool_factor: int = 50, seed: int = 0) -> None:
        self.lengths    = lengths
        self.batch_size = batch_size
        self.max_tokens = max_tokens or batch_size * max(lengths)
        self.shuffle    = shuffle
        self.pool_size  = batch_size * pool_factor
        self.seed       = seed
        self.epoch      = 0
        self._batches   = self._make_batches(list(range(len(lengths))))  # for __len__

    def _make_batches(self, idx: list[int]) -> list[list[int]]:
        batches = []
        for p in range(0, len(idx), self.pool_size):
            batch, longest = [], 0
            for i in sorted(idx[p:p + self.pool_size], key=lambda i: self.lengths[i]):
                longest_if_added = max(longest, self.lengths[i] + 2)  # + [SOS]/[EOS]
                if batch and (len(batch) == self.batch_size or (len(batch) + 1) * longest_if_added > self.max_tokens):
                    batches.append(batch)
                    batch, longest_if_added = [], self.lengths[i] + 2
                batch.append(i)
                longest = longest_if_added
            if batch:
                batches.append(batch)
        return batches

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        n   = len(self.lengths)
        idx = torch.randperm(n, generator=g).tolist() if self.shuffle else list(range(n))

        batches = self._make_batches(idx)
        order   = torch.randperm(len(batches), generator=g).tolist() if self.shuffle else range(len(batches))
        for b in order:
            yield batches[b]
