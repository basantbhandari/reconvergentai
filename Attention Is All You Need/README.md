# Attention Is All You Need

PyTorch reproduction of **Attention Is All You Need**  
([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))

EN→IT translation using the `opus_books` dataset.

## Architecture

The full encoder-decoder Transformer from the paper:

| Component | Detail |
|---|---|
| Layers (N) | 6 encoder + 6 decoder |
| d_model | 512 |
| Attention heads (h) | 8 |
| d_ff | 2048 |
| Dropout | 0.1 |
| Positional encoding | Sinusoidal |
| Label smoothing | 0.1 |
| Optimizer | Adam (lr=1e-4, ε=1e-9) |

## Project Structure

```
Attention Is All You Need/
├── transformer/
│   ├── __init__.py      public API
│   ├── model.py         transformer architecture
│   ├── dataset.py       BilingualDataset + causal_mask
│   └── config.py        training config + checkpoint paths
├── train.py             training entry point (TensorBoard)
├── train_wb.py          training entry point (Weights & Biases)
├── translate.py         inference entry point
└── pyproject.toml
```

## Quick Start

```python
from transformer import build_transformer

model = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    src_seq_len=350,
    tgt_seq_len=350,
)
```

## Training

```bash
uv run python train.py
```

For Weights & Biases logging:

```bash
uv run python train_wb.py
```

Checkpoints are saved to `opus_books_weights/tmodel_<epoch>.pt` after each epoch. Training resumes automatically from the latest checkpoint.

## Inference

```bash
uv run python translate.py "The cat is on the table."
```

Pass an integer to translate a sentence from the dataset by index:

```bash
uv run python translate.py 42
```

## Logging

All components use Python's `logging` module scoped to `train`, `train_wb`, `translate`, and `transformer` — third-party libraries are silent.

| Level | What you see |
|---|---|
| `INFO` (default) | Epoch loss, validation CER/WER/BLEU, checkpoint saves |
| `DEBUG` | All of the above plus tokenizer builds and batch details |

## Installation

Requires Python ≥ 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Reference

```bibtex
@article{vaswani2017attention,
  title   = {Attention Is All You Need},
  author  = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2017}
}
```
