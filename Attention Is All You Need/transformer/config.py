"""Training configuration and checkpoint path helpers."""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "get_config",
    "get_weights_file_path",
    "latest_weights_file_path",
    "best_weights_file_path",
]


def get_config() -> dict:
    return {
        "batch_size":      32,           # max sentences per batch
        "max_tokens":      2048,         # max padded tokens per batch (paper used ~25k on 8 GPUs)
        "num_epochs":      20,
        "lr":              1e-4,
        "seq_len":         350,          # max positions; batches are padded only to their longest sentence
        "d_model":         512,
        "seed":            42,
        "val_fraction":    0.1,
        "datasource":      "Helsinki-NLP/opus_books",
        "lang_src":        "en",
        "lang_tgt":        "it",
        "model_folder":    "opus_books_weights",
        "model_basename":  "tmodel_",
        "preload":         "latest",
        "tokenizer_file":  "opus_books_weights/tokenizer_{0}.json",
        "output_folder":   "output",
    }


def get_weights_file_path(config: dict, epoch: str) -> str:
    """Full training checkpoint (model + optimizer) for one epoch, used to resume."""
    filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(config["model_folder"]) / filename)


def latest_weights_file_path(config: dict) -> str | None:
    pattern = f"{config['model_basename']}*.pt"
    files   = sorted(Path(config["model_folder"]).glob(pattern))
    return str(files[-1]) if files else None


def best_weights_file_path(config: dict) -> str:
    """Inference-only weights from the epoch with the lowest validation loss."""
    return str(Path(config["model_folder"]) / "best.pt")
