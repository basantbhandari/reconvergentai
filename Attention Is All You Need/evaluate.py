"""Evaluate the trained checkpoints on the held-out opus_books split and write the README results.

    uv run python evaluate.py

Two checkpoints are compared on the full held-out split:
    best.pt          the epoch with the lowest validation loss (what training keeps)
    tmodel_NN.pt     the final epoch
With label smoothing, validation loss can rise while translations still improve, so
the lowest-loss epoch is not necessarily the best translator. The winner is the one
with the higher held-out BLEU.

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

from train import get_device, get_splits, make_loader, setup_logging, text_metrics, translate_loader
from translate import load_model, translate_batch
from transformer import best_weights_file_path, get_config, latest_weights_file_path

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


def plot_curves(history: list[dict], best_epoch: int, winner_epoch: int, path: Path) -> None:
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
    ax1.annotate("lowest val loss", (best_epoch + 1, ax1.get_ylim()[1]), xytext=(4, -12),
                 textcoords="offset points", color=INK_2, fontsize=9)

    bleu = [h["bleu"] for h in history]
    ax2.plot(epochs, bleu, color=BLUE, linewidth=2, marker="o", markersize=4)
    for e in sorted({best_epoch, winner_epoch}):
        ax2.annotate(f"{bleu[e]:.1f}", (e + 1, bleu[e]), xytext=(0, 8),
                     textcoords="offset points", ha="center", color=INK)
    ax2.annotate(f"max {max(bleu):.1f}", (bleu.index(max(bleu)) + 1, max(bleu)), xytext=(0, -16),
                 textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    ax2.set_title("Validation BLEU (500-sentence sample, greedy)", loc="left", fontsize=11)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def evaluate_checkpoint(ckpt: str, val_raw, train_raw) -> dict:
    model, tokenizer_src, tokenizer_tgt, config, device = load_model(ckpt)
    config = config | {"batch_size": 64}

    t0 = time.time()
    val_loader = make_loader(val_raw, tokenizer_src, tokenizer_tgt, config, shuffle=False)
    src, ref, pred = translate_loader(model, val_loader, tokenizer_tgt, device)
    heldout = text_metrics(ref, pred, tokenizer_tgt) | {"sentences": len(src)}
    log.info("%s  held-out: %s  (%.0fs)", ckpt, heldout, time.time() - t0)

    # Same metric on an equal-sized slice of the training set, to show the generalisation gap.
    train_slice  = Subset(train_raw, range(len(val_raw)))
    train_loader = make_loader(train_slice, tokenizer_src, tokenizer_tgt, config, shuffle=False)
    _, tref, tpred = translate_loader(model, train_loader, tokenizer_tgt, device)
    train_scores = text_metrics(tref, tpred, tokenizer_tgt) | {"sentences": len(tref)}
    log.info("%s  train slice: %s", ckpt, train_scores)

    return {
        "heldout": heldout, "train_slice": train_scores,
        "sources": src, "references": ref, "predictions": pred,
        "unseen": translate_batch(UNSEEN, ckpt),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    config  = get_config()
    history = json.loads(Path(config["output_folder"], "metrics.json").read_text())["epochs"]
    best_epoch  = min(history, key=lambda h: h["val_loss"])["epoch"]
    final_epoch = history[-1]["epoch"]

    candidates = {"lowest val loss": (best_weights_file_path(config), best_epoch)}
    final_ckpt = latest_weights_file_path(config)
    if final_ckpt and final_epoch != best_epoch:
        candidates["final epoch"] = (final_ckpt, final_epoch)

    # Tokenizers are loaded from disk, so get_splits reuses the ones the model was trained with.
    train_raw, val_raw, _, _ = get_splits(config)
    runs = {name: evaluate_checkpoint(ckpt, val_raw, train_raw) | {"checkpoint": ckpt, "epoch": epoch}
            for name, (ckpt, epoch) in candidates.items()}
    winner = max(runs, key=lambda n: runs[n]["heldout"]["bleu"])
    log.info("winner by held-out BLEU: %s (epoch %d)", winner, runs[winner]["epoch"] + 1)

    plot_curves(history, best_epoch, runs[winner]["epoch"], RESULTS / "training_curves.svg")

    (RESULTS / "metrics.json").write_text(json.dumps({
        "winner": winner,
        "checkpoints": {
            name: {k: r[k] for k in ("checkpoint", "epoch", "heldout", "train_slice")}
            for name, r in runs.items()
        },
        "total_train_time_min": sum(h["train_time_s"] for h in history) / 60,
        "device": str(get_device()),
        "epochs": history,
    }, indent=2))

    names  = list(runs)
    labels = [f"{n} (epoch {runs[n]['epoch'] + 1})" for n in names]
    ref_run = runs[names[0]]
    rng     = random.Random(config["seed"])
    samples = rng.sample(range(len(ref_run["sources"])), 15)
    esc     = lambda s: s.replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Sample translations (EN → IT)",
        "",
        "Greedy decoding. Checkpoints compared: " + ", ".join(f"**{l}**" for l in labels)
        + f". Higher held-out BLEU: **{winner}**.",
        "",
        "## Held-out opus_books sentences",
        "",
        "These sentences were never seen in training (random sample, seed 42).",
        "",
        "| English (source) | Italian (reference) | " + " | ".join(labels) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for i in samples:
        cells = [esc(ref_run["sources"][i]), esc(ref_run["references"][i])] + [esc(runs[n]["predictions"][i]) for n in names]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## New sentences (not from the dataset)",
        "",
        "| English | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(names),
    ]
    for j, s in enumerate(UNSEEN):
        lines.append("| " + " | ".join([esc(s)] + [esc(runs[n]["unseen"][j]) for n in names]) + " |")
    (RESULTS / "translations.md").write_text("\n".join(lines) + "\n")

    log.info("wrote %s", ", ".join(str(p) for p in sorted(RESULTS.iterdir())))
    for name, r in runs.items():
        print(f"{name:16s} epoch {r['epoch'] + 1:2d}  held-out {r['heldout']}  train-slice {r['train_slice']}")
    print(f"winner: {winner}")
    for j, s in enumerate(UNSEEN):
        print(f"{s}  →  " + "  |  ".join(runs[n]["unseen"][j] for n in names))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logging("INFO")
    main()
