# b_lm_project

Small experimental language model project evolving from a character-level RNN to a mini Transformer architecture with plans for BPE tokenization and structured Q&A training.

The goal is to progressively build a minimal GPT-style input → response pipeline from scratch.

Note: I have no clue what the project name stands for

## Current status
- Transformer-based autoregressive language model
- Train/validation split
- Perplexity reporting
- Gradient clipping
- Checkpoint saving (model + optimizer + metadata)
- Text generation CLI (temperature sampling)
- Interactive generation loop
- DirectML (Windows AMD) compatibility fallback
- Manual optimizer experimentation (SGD, Adam, custom Adam variant)
- Training step timing utilities for performance benchmarking

## Project files
- `train.py` — dataset building, model training, checkpoint saving
- `generate.py` — one-shot CLI generation
- `interactive_generate.py` — multi-turn interactive inference
- `transformer.py` — Transformer model definition
- `data.tx`t — raw training corpus
- `train_tokenizer.py` (planned) — BPE tokenizer training
## Overview
`train.py`:
- Loads raw text from `data.txt`.
- Builds a character vocabulary and mappings (`char_to_idx`, `idx_to_char`).
- Encodes text to indices and creates input/target sequences.
- Trains `CharRNN` and saves `checkpoint.pth` including vocab and hyperparams.

`generate.py`:
- Exposes reusable generation utilities and a one-shot CLI.
- Loads a checkpoint and generates autoregressive character text with temperature.

`interactive_generate.py`:
- Loads a checkpoint once at startup and runs a multi-turn `input()` loop.
- Supports quit commands (`quit`, `exit`, `q`, `:q`) and empty-prompt handling.

## Quick start
1. Install dependencies:

```sh
# CPU-only PyTorch (example):
pip install torch numpy

# Optional (Windows DirectML backend):
# pip install torch-directml

```

2. Train (quick run):

```sh
python train.py
```

3. Generate from saved checkpoint:

```sh
python generate.py --ckpt checkpoint.pth --start "Sing, " --length 300 --temp 0.8

# equivalent checkpoint flag:
python generate.py --checkpoint checkpoint.pth --start "Sing, " --length 300 --temp 0.8
```

4. Interactive generation (checkpoint loaded once):

```sh
python interactive_generate.py --ckpt checkpoint.pth --length 300 --temp 0.8
```

## Next Development Direction

Short-term:
- Implement BPE tokenizer
- Build structured Q&A dataset
- Retrain Transformer
- Improve sampling (top-k / nucleus)
- Build minimal conversational interface

Long-term:
- Instruction tuning
- Larger dataset
- Deeper transformer
- CUDA/ROCm-native acceleration
- Evaluation metrics (perplexity + qualitative scoring)

## License
This project is an educational implementation and experimentation sandbox for understanding how modern autoregressive language models are constructed from first principles.
