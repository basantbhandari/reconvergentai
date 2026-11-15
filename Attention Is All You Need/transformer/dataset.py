"""BilingualDataset and masking utilities."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

__all__ = ["BilingualDataset", "causal_mask"]


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
