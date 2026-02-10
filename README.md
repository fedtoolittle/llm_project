# b_lm_project

Small char-level language-model preprocessing & data utilities.

## Project files
- [train.py](train.py) — preprocessing & dataset creation (reads [`train.text`](train.py), builds [`train.chars`](train.py), [`train.vocab_size`](train.py), [`train.char_to_idx`](train.py), [`train.idx_to_char`](train.py), [`train.encoded_text`](train.py), [`train.sequence_length`](train.py), [`train.inputs`](train.py), [`train.targets`](train.py), [`train.dataset`](train.py), [`train.loader`](train.py))
- [generate.py](generate.py) — (placeholder for sampling / generation)
- [test.py](test.py) — quick import check / examples
- [data.txt](data.txt) — raw training text

## Overview
`train.py`:
- Loads raw text from [data.txt](data.txt) into [`train.text`](train.py).
- Builds a character vocabulary [`train.chars`](train.py) and mappings [`train.char_to_idx`](train.py], [`train.idx_to_char`](train.py).
- Encodes text to indices [`train.encoded_text`](train.py].
- Creates input/target sequences of length [`train.sequence_length`](train.py] and packs them into a `TensorDataset` ([`train.dataset`](train.py]) and `DataLoader` ([`train.loader`](train.py]).

## Quick start
1. Install dependencies:
```sh
pip install torch numpy
