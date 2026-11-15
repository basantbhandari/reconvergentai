"""Inference entry point — translate a sentence or dataset index."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from transformer import BilingualDataset, build_transformer, causal_mask, get_config, latest_weights_file_path
from train import setup_logging

log = logging.getLogger("translate")


def translate(sentence: str) -> str:
    config = get_config()
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    log.info("device=%s", device)

    tokenizer_src = Tokenizer.from_file(str(Path(config["tokenizer_file"].format(config["lang_src"]))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config["tokenizer_file"].format(config["lang_tgt"]))))

    model = build_transformer(
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
        config["seq_len"], config["seq_len"], d_model=config["d_model"],
    ).to(device)

    ckpt = latest_weights_file_path(config)
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found. Train the model first.")
    log.info("loading checkpoint %s", ckpt)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        idx = int(sentence)
        ds  = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="all")
        ds  = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
        label    = ds[idx]["tgt_text"]
        sentence = ds[idx]["src_text"]

    seq_len = config["seq_len"]
    model.eval()
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id("[PAD]")] * (seq_len - len(source.ids) - 2), dtype=torch.int64),
        ]).to(device)
        source_mask = (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(device)

        encoder_output = model.encode(source, source_mask)
        decoder_input  = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id("[SOS]")).type_as(source).to(device)

        log.info("SOURCE    : %s", sentence)
        if label:
            log.info("TARGET    : %s", label)

        tokens = []
        while decoder_input.size(1) < seq_len:
            decoder_mask = torch.triu(
                torch.ones(1, decoder_input.size(1), decoder_input.size(1)), diagonal=1
            ).int().type_as(source_mask).to(device)
            out          = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob         = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
                dim=1,
            )
            token = tokenizer_tgt.decode([next_word.item()])
            tokens.append(token)
            if next_word == tokenizer_tgt.token_to_id("[EOS]"):
                break

        result = " ".join(tokens)
        log.info("PREDICTED : %s", result)
        return result


if __name__ == "__main__":
    setup_logging("INFO")
    translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good student.")
