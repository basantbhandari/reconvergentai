# Attention Is All You Need

PyTorch reproduction of **Attention Is All You Need**  
([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))

The full encoder–decoder Transformer (the paper's *base* model), trained from scratch for
English → Italian translation on the real
[`Helsinki-NLP/opus_books`](https://huggingface.co/datasets/Helsinki-NLP/opus_books) corpus.
The dataset is downloaded automatically from the Hugging Face Hub.

## Results

> **Pending:** the 20-epoch training run started on 2026-11-15 is still in progress.
> This section will be filled in with the measured scores from `evaluate.py`:
> held-out BLEU / CER / WER, training curves (`results/training_curves.svg`) and
> sample translations (`results/translations.md`).

## Dataset

| | |
|---|---|
| Source | `Helsinki-NLP/opus_books`, `en-it` config (sentence-aligned public-domain books) |
| Pairs | 32,332 |
| Split | 90 / 10 train / held-out, seeded (`seed=42`): **29,099 train**, **3,233 held-out** |
| Tokenizer | word-level, trained on the corpus (`min_frequency=2`): 15,698 EN / 22,463 IT tokens |
| Sentence length | median ~22 tokens, max 309 |

`opus_books` only ships a `train` split, so a fixed fraction is held out. Because the split is
seeded, `train.py` and `evaluate.py` always use the same held-out sentences.

## Architecture

| Component | Detail |
|---|---|
| Layers (N) | 6 encoder + 6 decoder |
| d_model | 512 |
| Attention heads (h) | 8 |
| d_ff | 2048 |
| Dropout | 0.1 |
| Positional encoding | Sinusoidal |
| Label smoothing | 0.1 |
| Parameters | 75.2 M |
| Optimizer | Adam (lr=1e-4 constant, ε=1e-9) |
| Batching | ≤ 32 sentences and ≤ 2,048 padded tokens per batch, length-bucketed |
| Decoding | Greedy |

### Differences from the paper

The model matches the paper's base configuration. The training setup is scaled down to
one laptop GPU and a small dataset, so absolute scores are **not** comparable to the paper's
27.3 BLEU (WMT'14 EN→DE).

| | Paper | This repo |
|---|---|---|
| Data | WMT'14, 4.5 M pairs | opus_books EN→IT, 32 k pairs |
| Tokenization | shared BPE, ~37 k | separate word-level vocabularies |
| LR schedule | warmup 4,000 steps, then ∝ step⁻⁰·⁵; Adam β₂=0.98 | constant 1e-4; Adam β₂=0.999 |
| Batch | ~25 k source + 25 k target tokens, 8 GPUs | ≤ 2,048 tokens, 1 Apple-silicon GPU (MPS) |
| Layer norm | after each sub-layer (post-LN) | before each sub-layer (pre-LN) |
| Decoding | beam search (size 4), checkpoint averaging | greedy, single best checkpoint |

### Efficient batching

`BilingualDataset` pads every sentence to 350 tokens, but the median sentence is ~22 tokens.
`collate_trim` cuts each batch down to its longest sentence, and `LengthBucketSampler` groups
sentences of similar length and caps each batch at 2,048 tokens (as the paper batched by token
count). This changes nothing numerically: the logits for real tokens are identical to the
fully padded forward pass (max difference 0.0, checked). On an Apple M5 it brings an epoch from
~44 min to ~7 min and keeps memory flat on the rare very long sentences.

## Project Structure

```
Attention Is All You Need/
├── transformer/
│   ├── __init__.py          public API
│   ├── model.py             Transformer architecture
│   ├── dataset.py           BilingualDataset, causal_mask, collate_trim, LengthBucketSampler
│   └── config.py            training config + checkpoint paths
├── train.py                 training entry point (metrics JSON, optional W&B)
├── evaluate.py              held-out BLEU/CER/WER, sample translations, training-curve plot
├── translate.py             inference entry point
├── results/                 committed evaluation outputs (written by evaluate.py)
├── output/                  git-ignored: train.log, metrics.json
├── opus_books_weights/      git-ignored: trained model + tokenizers
└── pyproject.toml
```

## Installation

Requires Python ≥ 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Training

```bash
uv run python train.py                 # 20 epochs, resumes from the last checkpoint if present
uv run python train.py --no-resume     # start from scratch
uv run python train.py --epochs 5      # fewer epochs
```

Each epoch prints the training loss, validation loss, and BLEU/CER/WER on 500 held-out
sentences, with two example translations. Per-epoch metrics are written to
`output/metrics.json`.

To also log to [Weights & Biases](https://wandb.ai), run `wandb login` once, then:

```bash
uv run python train.py --wandb
uv run python train.py --wandb --wandb-project my-project
```

This logs the training loss every step, the validation metrics every epoch, and the config.

## The trained model

Training writes to `opus_books_weights/`:

| File | Size | Purpose |
|---|---|---|
| `best.pt` | ~300 MB | weights from the epoch with the lowest validation loss; used by `translate.py` and `evaluate.py` |
| `tmodel_NN.pt` | ~900 MB | latest full checkpoint (weights + optimizer state), used only to resume; replaced every epoch |
| `tokenizer_en.json`, `tokenizer_it.json` | < 1 MB | vocabularies the model was trained with; must stay with `best.pt` |

The folder is git-ignored because of its size. Recreate it with `uv run python train.py`.

## Evaluation

```bash
uv run python evaluate.py
```

Loads `best.pt`, greedy-translates the full held-out split (3,233 sentences) and an equal-sized
slice of the training set, and writes:

- `results/metrics.json`: per-epoch history and final scores
- `results/translations.md`: sample held-out translations and translations of new sentences
- `results/training_curves.svg`: loss and BLEU per epoch

BLEU is corpus BLEU-4 (`torchmetrics.BLEUScore`, ×100), computed after splitting references and
outputs into words and punctuation the same way the tokenizer does, so `tavolo.` and `tavolo .`
count as the same tokens.

## Inference

```bash
uv run python translate.py "The cat is on the table." "Where is the station?"
```

Pass an integer to translate sentence *N* of opus_books and show its reference translation
(the index refers to the whole dataset, so the sentence may be from the training split):

```bash
uv run python translate.py 42
```

From Python:

```python
from translate import translate, translate_batch

translate("I love you.")
translate_batch(["Good morning.", "See you tomorrow."])
```

## Logging

All components use Python's `logging` module scoped to `train`, `evaluate`, `translate`, and
`transformer`, so third-party libraries stay quiet.

## Reference

```bibtex
@article{vaswani2017attention,
  title   = {Attention Is All You Need},
  author  = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2017}
}
```
