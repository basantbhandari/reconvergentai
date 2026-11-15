"""Training configuration and checkpoint path helpers."""
from __future__ import annotations

from pathlib import Path

__all__ = ["get_config", "get_weights_file_path", "latest_weights_file_path"]


def get_config() -> dict:
    return {
        "batch_size":      8,
        "num_epochs":      20,
        "lr":              1e-4,
        "seq_len":         350,
        "d_model":         512,
        "datasource":      "opus_books",
        "lang_src":        "en",
        "lang_tgt":        "it",
        "model_folder":    "weights",
        "model_basename":  "tmodel_",
        "preload":         "latest",
        "tokenizer_file":  "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config: dict, epoch: str) -> str:
    folder   = f"{config['datasource']}_{config['model_folder']}"
    filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(folder) / filename)


def latest_weights_file_path(config: dict) -> str | None:
    folder  = f"{config['datasource']}_{config['model_folder']}"
    pattern = f"{config['model_basename']}*.pt"
    files   = sorted(Path(folder).glob(pattern))
    return str(files[-1]) if files else None
