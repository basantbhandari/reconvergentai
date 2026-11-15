"""Training entry point — TensorBoard logging."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchmetrics

from transformer import (
    BilingualDataset,
    build_transformer,
    causal_mask,
    get_config,
    get_weights_file_path,
    latest_weights_file_path,
)

log = logging.getLogger("train")


def setup_logging(level: str = "INFO") -> None:
    lvl     = getattr(logging, level.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(name)-15s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    for name in ("train", "transformer"):
        logger = logging.getLogger(name)
        logger.setLevel(lvl)
        logger.addHandler(handler)
        logger.propagate = False


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config: dict, ds, lang: str) -> Tokenizer:
    path = Path(config["tokenizer_file"].format(lang))
    if path.exists():
        return Tokenizer.from_file(str(path))
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
    )
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(path))
    log.info("tokenizer saved → %s", path)
    return tokenizer


def get_ds(config: dict):
    ds_raw = load_dataset(
        config["datasource"],
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    train_size = int(0.9 * len(ds_raw))
    train_raw, val_raw = random_split(ds_raw, [train_size, len(ds_raw) - train_size])

    train_ds = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds   = BilingualDataset(val_raw,   tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_src = max(len(tokenizer_src.encode(item["translation"][config["lang_src"]]).ids) for item in ds_raw)
    max_tgt = max(len(tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids) for item in ds_raw)
    log.info("max src len=%d  max tgt len=%d", max_src, max_tgt)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,                    shuffle=True)
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input  = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out          = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob         = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1,
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, max_len, device, writer, global_step, num_examples=2):
    model.eval()
    source_texts, expected, predicted = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_examples:
                break
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask  = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1

            out_ids   = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            src_text  = batch["src_text"][0]
            tgt_text  = batch["tgt_text"][0]
            pred_text = tokenizer_tgt.decode(out_ids.detach().cpu().numpy())

            source_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(pred_text)

            log.info("SOURCE    : %s", src_text)
            log.info("TARGET    : %s", tgt_text)
            log.info("PREDICTED : %s", pred_text)

    cer  = torchmetrics.CharErrorRate()(predicted, expected)
    wer  = torchmetrics.WordErrorRate()(predicted, expected)
    bleu = torchmetrics.BLEUScore()(predicted, [[e] for e in expected])

    log.info("val  CER=%.4f  WER=%.4f  BLEU=%.4f", cer, wer, bleu)

    if writer:
        writer.add_scalar("val/cer",  cer,  global_step)
        writer.add_scalar("val/wer",  wer,  global_step)
        writer.add_scalar("val/bleu", bleu, global_step)
        writer.flush()


def train_model(config: dict) -> None:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    log.info("device=%s", device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = build_transformer(
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
        config["seq_len"], config["seq_len"], d_model=config["d_model"],
    ).to(device)

    writer    = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step   = 0
    preload       = config["preload"]
    ckpt_path     = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None

    if ckpt_path:
        log.info("loading checkpoint %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step   = state["global_step"]
    else:
        log.info("no checkpoint found — training from scratch")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        total_loss  = 0.0
        batch_iter  = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iter:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask  = batch["encoder_mask"].to(device)
            decoder_mask  = batch["decoder_mask"].to(device)
            label         = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output    = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iter.set_postfix({"loss": f"{loss.item():.3f}"})
            total_loss += loss.item()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        avg_loss = total_loss / len(train_loader)
        log.info("epoch %02d  avg_loss=%.4f  global_step=%d", epoch, avg_loss, global_step)

        run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, writer, global_step)

        ckpt = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step":         global_step,
        }, ckpt)
        log.info("checkpoint saved → %s", ckpt)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logging("INFO")
    train_model(get_config())
