"""Evaluate best.pt on the held-out opus_books split and write the README results.

    uv run python evaluate.py

Writes (committed, small):
    results/metrics.json          per-epoch training history + final test-set scores
    results/translations.md       sample translations (held-out + unseen sentences)
    results/training_curves.svg   loss and BLEU per epoch
"""
from __future__ import annotations

import json
import logging
import random
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from train import get_splits, make_loader, setup_logging, text_metrics, translate_loader
from translate import load_model, translate_batch

log = logging.getLogger("evaluate")

RESULTS = Path("results")

# Sentences written for this demo, not taken from opus_books.
UNSEEN = [
    "The cat is on the table.",
    "I love you.",
    "Where is the station?",
    "She opened the window and looked at the garden.",
    "I don't know what to do.",
    "The old man walked slowly towards the house.",
    "We will leave tomorrow morning.",
    "He was very tired after the long journey.",
]

# Chart tokens (validated categorical slots 1–2, light surface).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def plot_curves(history: list[dict], best_epoch: int, path: Path) -> None:
    epochs = [h["epoch"] + 1 for h in history]
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(best_epoch + 1, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlabel("epoch")
        ax.set_xlim(0.5, max(epochs) + 2.5)

    for key, color, name in (("train_loss", BLUE, "train"), ("val_loss", ORANGE, "validation")):
        ys = [h[key] for h in history]
        ax1.plot(epochs, ys, color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax1.annotate(name, (epochs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", color=INK_2)
    ax1.set_title("Cross-entropy loss (label smoothing 0.1)", loc="left", fontsize=11)
    ax1.legend(frameon=False, loc="upper right")
    ax1.annotate("best checkpoint", (best_epoch + 1, ax1.get_ylim()[1]), xytext=(4, -12),
                 textcoords="offset points", color=INK_2, fontsize=9)

    bleu = [h["bleu"] for h in history]
    ax2.plot(epochs, bleu, color=BLUE, linewidth=2, marker="o", markersize=4)
    ax2.annotate(f"{bleu[best_epoch]:.1f}", (best_epoch + 1, bleu[best_epoch]), xytext=(0, 8),
                 textcoords="offset points", ha="center", color=INK)
    ax2.set_title("Validation BLEU (500-sentence sample, greedy)", loc="left", fontsize=11)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    model, tokenizer_src, tokenizer_tgt, config, device = load_model()
    config["batch_size"] = 64

    train_raw, val_raw, _, _ = get_splits(config)

    t0 = time.time()
    val_loader = make_loader(val_raw, tokenizer_src, tokenizer_tgt, config, shuffle=False)
    src, ref, pred = translate_loader(model, val_loader, tokenizer_tgt, device)
    val_scores = text_metrics(ref, pred, tokenizer_tgt) | {"sentences": len(src)}
    log.info("held-out: %s  (%.0fs)", val_scores, time.time() - t0)

    # Same metric on a slice of the training set, to show the generalisation gap.
    train_slice  = Subset(train_raw, range(len(val_raw)))
    train_loader = make_loader(train_slice, tokenizer_src, tokenizer_tgt, config, shuffle=False)
    _, tref, tpred = translate_loader(model, train_loader, tokenizer_tgt, device)
    train_scores = text_metrics(tref, tpred, tokenizer_tgt) | {"sentences": len(tref)}
    log.info("train slice: %s", train_scores)

    unseen_pred = translate_batch(UNSEEN)

    history = json.loads(Path(config["output_folder"], "metrics.json").read_text())["epochs"]
    best    = min(history, key=lambda h: h["val_loss"])
    plot_curves(history, best["epoch"], RESULTS / "training_curves.svg")

    (RESULTS / "metrics.json").write_text(json.dumps({
        "best_epoch":        best["epoch"],
        "heldout":           val_scores,
        "train_slice":       train_scores,
        "total_train_time_min": sum(h["train_time_s"] for h in history) / 60,
        "device":            str(device),
        "epochs":            history,
    }, indent=2))

    rng     = random.Random(config["seed"])
    samples = rng.sample(range(len(src)), 15)
    lines   = [
        "# Sample translations (EN → IT)",
        "",
        f"Greedy decoding with `best.pt` (epoch {best['epoch'] + 1}).",
        "",
        "## Held-out opus_books sentences",
        "",
        "These sentences were never seen in training.",
        "",
        "| English (source) | Italian (reference) | Model output |",
        "|---|---|---|",
    ]
    esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
    lines += [f"| {esc(src[i])} | {esc(ref[i])} | {esc(pred[i])} |" for i in samples]
    lines += [
        "",
        "## New sentences (not from the dataset)",
        "",
        "| English | Model output |",
        "|---|---|",
    ]
    lines += [f"| {esc(s)} | {esc(p)} |" for s, p in zip(UNSEEN, unseen_pred)]
    (RESULTS / "translations.md").write_text("\n".join(lines) + "\n")

    log.info("wrote %s", ", ".join(str(p) for p in sorted(RESULTS.iterdir())))
    print(json.dumps({"best_epoch": best["epoch"], "heldout": val_scores, "train_slice": train_scores}, indent=2))
    for s, p in zip(UNSEEN, unseen_pred):
        print(f"{s}  →  {p}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logging("INFO")
    main()
