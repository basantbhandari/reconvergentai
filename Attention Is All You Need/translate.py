"""Inference entry point — translate English sentences (or dataset indices) to Italian.

    uv run python translate.py "The cat is on the table." "Where is the station?"
    uv run python translate.py 42          # sentence #42 of opus_books, shown with its reference
"""
from __future__ import annotations

import logging
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from transformer import best_weights_file_path, build_transformer, get_config
from train import get_device, greedy_decode_batch, setup_logging

log = logging.getLogger("translate")


@lru_cache(maxsize=1)
def load_model():
    """Load best.pt and the tokenizers. Returns (model, tokenizer_src, tokenizer_tgt, config, device)."""
    config = get_config()
    device = get_device()
    tokenizer_src = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_src"]))
    tokenizer_tgt = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_tgt"]))

    ckpt = best_weights_file_path(config)
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"{ckpt} not found. Train the model first: uv run python train.py")
    state = torch.load(ckpt, map_location=device)
    model = build_transformer(
        state["src_vocab_size"], state["tgt_vocab_size"],
        state["seq_len"], state["seq_len"], d_model=state["d_model"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    log.info("loaded %s (epoch %d, val_loss=%.4f) on %s", ckpt, state["epoch"], state["metrics"]["val_loss"], device)
    return model, tokenizer_src, tokenizer_tgt, config, device


def translate_batch(sentences: list[str]) -> list[str]:
    model, tokenizer_src, tokenizer_tgt, config, device = load_model()
    sos, eos, pad = (tokenizer_src.token_to_id(t) for t in ("[SOS]", "[EOS]", "[PAD]"))

    encoded = [[sos, *tokenizer_src.encode(s).ids[: config["seq_len"] - 2], eos] for s in sentences]
    width   = max(map(len, encoded))
    source  = torch.tensor([e + [pad] * (width - len(e)) for e in encoded], dtype=torch.int64, device=device)
    mask    = (source != pad).unsqueeze(1).unsqueeze(1).int()

    ids = greedy_decode_batch(model, source, mask, tokenizer_tgt, min(width + 50, config["seq_len"]), device)
    return [tokenizer_tgt.decode(r) for r in ids]


def translate(sentence: str) -> str:
    return translate_batch([sentence])[0]


def main(args: list[str]) -> None:
    args = args or ["I am not a very good student."]
    config = get_config()
    references: dict[int, str] = {}
    sentences = []
    if any(a.isdigit() for a in args):
        ds = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    for i, a in enumerate(args):
        if a.isdigit():
            pair = ds[int(a)]["translation"]
            sentences.append(pair[config["lang_src"]])
            references[i] = pair[config["lang_tgt"]]
        else:
            sentences.append(a)

    for i, (src, pred) in enumerate(zip(sentences, translate_batch(sentences))):
        print(f"SOURCE    : {src}")
        if i in references:
            print(f"TARGET    : {references[i]}")
        print(f"PREDICTED : {pred}\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logging("INFO")
    main(sys.argv[1:])
