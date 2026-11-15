"""Training entry point — logs to output/metrics.json (and W&B with --wandb), saves checkpoints to opus_books_weights/."""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from transformer import (
    BilingualDataset,
    LengthBucketSampler,
    best_weights_file_path,
    build_transformer,
    causal_mask,
    collate_trim,
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
    for name in ("train", "evaluate", "translate", "transformer"):
        logger = logging.getLogger(name)
        logger.setLevel(lvl)
        logger.addHandler(handler)
        logger.propagate = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- data

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config: dict, ds, lang: str) -> Tokenizer:
    path = Path(config["tokenizer_file"].format(lang))
    if path.exists():
        return Tokenizer.from_file(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
    )
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(path))
    log.info("tokenizer saved → %s", path)
    return tokenizer


def get_splits(config: dict):
    """Load opus_books, build/load tokenizers, and make a seeded train/val split.

    opus_books only ships a `train` split, so we hold out `val_fraction` of it.
    The split is seeded, so train.py and evaluate.py see exactly the same val set.
    """
    ds_raw = load_dataset(
        config["datasource"],
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    val_size   = int(config["val_fraction"] * len(ds_raw))
    train_size = len(ds_raw) - val_size
    gen        = torch.Generator().manual_seed(config["seed"])
    train_raw, val_raw = random_split(ds_raw, [train_size, val_size], generator=gen)
    log.info("dataset %s  train=%d  val=%d", config["datasource"], train_size, val_size)
    return train_raw, val_raw, tokenizer_src, tokenizer_tgt


def make_loader(raw, tokenizer_src, tokenizer_tgt, config: dict, shuffle: bool) -> DataLoader:
    ds = BilingualDataset(raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    src = [p["translation"][config["lang_src"]] for p in raw]
    tgt = [p["translation"][config["lang_tgt"]] for p in raw]
    lengths = [
        max(len(a.ids), len(b.ids))
        for a, b in zip(tokenizer_src.encode_batch(src), tokenizer_tgt.encode_batch(tgt))
    ]
    sampler = LengthBucketSampler(lengths, config["batch_size"], max_tokens=config["max_tokens"],
                                  shuffle=shuffle, seed=config["seed"])
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_trim)


def get_ds(config: dict):
    train_raw, val_raw, tokenizer_src, tokenizer_tgt = get_splits(config)
    train_loader = make_loader(train_raw, tokenizer_src, tokenizer_tgt, config, shuffle=True)
    val_loader   = make_loader(val_raw,   tokenizer_src, tokenizer_tgt, config, shuffle=False)
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


# ---------------------------------------------------------------- decoding

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """Greedy decode a single sentence (batch size 1)."""
    return greedy_decode_batch(model, source, source_mask, tokenizer_tgt, max_len, device)[0]


@torch.no_grad()
def greedy_decode_batch(model, source, source_mask, tokenizer_tgt, max_len, device) -> list[list[int]]:
    """Greedy decode a batch. Returns generated token ids per row, without [SOS]/[EOS]."""
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    B = source.size(0)

    encoder_output = model.encode(source, source_mask)
    decoder_input  = torch.full((B, 1), sos_idx, dtype=source.dtype, device=device)
    finished       = torch.zeros(B, dtype=torch.bool, device=device)

    while decoder_input.size(1) < max_len and not finished.all():
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out          = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        next_word    = model.project(out[:, -1]).argmax(dim=-1)
        next_word    = torch.where(finished, torch.full_like(next_word, eos_idx), next_word)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)
        finished     |= next_word == eos_idx

    results = []
    for row in decoder_input[:, 1:].tolist():
        results.append(row[: row.index(eos_idx)] if eos_idx in row else row)
    return results


def bleu_tokens(text: str, tokenizer: Tokenizer) -> str:
    """Split text the same way the tokenizer does (words and punctuation), for BLEU."""
    return " ".join(w for w, _ in tokenizer.pre_tokenizer.pre_tokenize_str(text))


def translate_loader(model, loader, tokenizer_tgt, device, limit: int | None = None):
    """Greedy-translate every batch in `loader`. Returns (sources, references, predictions)."""
    model.eval()
    sources, references, predictions = [], [], []
    for batch in loader:
        enc  = batch["encoder_input"].to(device)
        mask = batch["encoder_mask"].to(device)
        ids  = greedy_decode_batch(model, enc, mask, tokenizer_tgt, enc.size(1) + 50, device)
        sources     += batch["src_text"]
        references  += batch["tgt_text"]
        predictions += [tokenizer_tgt.decode(r) for r in ids]
        if limit and len(sources) >= limit:
            break
    return sources[:limit], references[:limit], predictions[:limit]


def text_metrics(references: list[str], predictions: list[str], tokenizer_tgt: Tokenizer) -> dict:
    refs  = [bleu_tokens(r, tokenizer_tgt) for r in references]
    preds = [bleu_tokens(p, tokenizer_tgt) for p in predictions]
    return {
        "bleu": float(torchmetrics.BLEUScore()(preds, [[r] for r in refs])) * 100,
        "cer":  float(torchmetrics.CharErrorRate()(preds, refs)),
        "wer":  float(torchmetrics.WordErrorRate()(preds, refs)),
    }


# ---------------------------------------------------------------- training

@torch.no_grad()
def compute_loss(model, loader, loss_fn, vocab_size, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        enc   = model.encode(batch["encoder_input"].to(device), batch["encoder_mask"].to(device))
        dec   = model.decode(enc, batch["encoder_mask"].to(device), batch["decoder_input"].to(device), batch["decoder_mask"].to(device))
        loss  = loss_fn(model.project(dec).view(-1, vocab_size), batch["label"].to(device).view(-1))
        total += loss.item()
        n     += 1
    return total / n


def run_validation(model, val_loader, loss_fn, tokenizer_tgt, device, num_bleu: int = 500) -> dict:
    val_loss = compute_loss(model, val_loader, loss_fn, tokenizer_tgt.get_vocab_size(), device)
    src, ref, pred = translate_loader(model, val_loader, tokenizer_tgt, device, limit=num_bleu)
    metrics = {"val_loss": val_loss, **text_metrics(ref, pred, tokenizer_tgt)}
    if device.type == "mps":
        torch.mps.empty_cache()
    for s, r, p in list(zip(src, ref, pred))[:2]:
        log.info("SOURCE    : %s", s)
        log.info("TARGET    : %s", r)
        log.info("PREDICTED : %s", p)
    log.info("val  loss=%.4f  BLEU=%.2f  CER=%.4f  WER=%.4f  (BLEU on %d sentences)",
             metrics["val_loss"], metrics["bleu"], metrics["cer"], metrics["wer"], len(src))
    return metrics


def save_best(model, config, tokenizer_src, tokenizer_tgt, epoch, metrics) -> None:
    path = best_weights_file_path(config)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch":            epoch,
        "metrics":          metrics,
        "src_vocab_size":   tokenizer_src.get_vocab_size(),
        "tgt_vocab_size":   tokenizer_tgt.get_vocab_size(),
        "seq_len":          config["seq_len"],
        "d_model":          config["d_model"],
    }, path)
    log.info("new best (val_loss=%.4f) → %s", metrics["val_loss"], path)


def train_model(config: dict) -> None:
    torch.manual_seed(config["seed"])
    device = get_device()
    log.info("device=%s", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    out_dir = Path(config["output_folder"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    vocab_size = tokenizer_tgt.get_vocab_size()
    model = build_transformer(
        tokenizer_src.get_vocab_size(), vocab_size,
        config["seq_len"], config["seq_len"], d_model=config["d_model"],
    ).to(device)
    log.info("params=%.1fM  src_vocab=%d  tgt_vocab=%d",
             sum(p.numel() for p in model.parameters()) / 1e6, tokenizer_src.get_vocab_size(), vocab_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch, global_step, history = 0, 0, []
    preload   = config["preload"]
    ckpt_path = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None

    if ckpt_path:
        log.info("loading checkpoint %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step   = state["global_step"]
        if metrics_path.exists():
            history = [h for h in json.loads(metrics_path.read_text())["epochs"] if h["epoch"] < initial_epoch]
        train_loader.batch_sampler.epoch = initial_epoch
    else:
        log.info("no checkpoint found — training from scratch")

    run = None
    if config.get("wandb"):
        import wandb  # optional; only needed with --wandb

        run = wandb.init(project=config["wandb_project"], config=config,
                         resume="allow" if initial_epoch else None)
        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="global_step")
        run.define_metric("val/*",   step_metric="global_step")

    best_val = min((h["val_loss"] for h in history), default=float("inf"))
    loss_fn  = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        total_loss, t0 = 0.0, time.time()
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iter:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask  = batch["encoder_mask"].to(device)
            decoder_mask  = batch["decoder_mask"].to(device)
            label         = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output    = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            batch_iter.set_postfix({"loss": f"{loss.item():.3f}"})
            total_loss += loss.item()
            if run:
                run.log({"train/loss": loss.item(), "global_step": global_step})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        train_loss = total_loss / len(train_loader)
        train_time = time.time() - t0
        log.info("epoch %02d  train_loss=%.4f  global_step=%d  time=%.0fs", epoch, train_loss, global_step, train_time)

        metrics = run_validation(model, val_loader, loss_fn, tokenizer_tgt, device)
        history.append({"epoch": epoch, "global_step": global_step, "train_loss": train_loss,
                        "train_time_s": train_time, **metrics})
        metrics_path.write_text(json.dumps({"config": config, "epochs": history}, indent=2))
        if run:
            run.log({"val/" + k: v for k, v in metrics.items()}
                    | {"train/epoch_loss": train_loss, "epoch": epoch, "global_step": global_step})

        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            save_best(model, config, tokenizer_src, tokenizer_tgt, epoch, metrics)

        # Keep only the newest full checkpoint (~900 MB each with Adam state) for resuming.
        ckpt = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step":          global_step,
        }, ckpt)
        for old in Path(config["model_folder"]).glob(f"{config['model_basename']}*.pt"):
            if str(old) != ckpt:
                old.unlink()
        log.info("checkpoint saved → %s", ckpt)

    if run:
        run.finish()


def parse_args(config: dict) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs",     type=int, default=config["num_epochs"])
    p.add_argument("--batch-size", type=int, default=config["batch_size"])
    p.add_argument("--no-resume",  action="store_true", help="ignore existing checkpoints")
    p.add_argument("--wandb",      action="store_true", help="also log to Weights & Biases (run `wandb login` first)")
    p.add_argument("--wandb-project", default="pytorch-transformer")
    a = p.parse_args()
    config["num_epochs"]    = a.epochs
    config["batch_size"]    = a.batch_size
    config["wandb"]         = a.wandb
    config["wandb_project"] = a.wandb_project
    if a.no_resume:
        config["preload"] = None
    return config


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logging("INFO")
    train_model(parse_args(get_config()))
