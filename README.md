# llm_project

A minimal GPT-style language model sandbox for learning neural networks, evolving from character-level RNNs to a Transformer with BPE tokenization.

The goal currently is to progressively build a input → response pipeline from scratch.

The project is currently on indefinite hiatus due to insufficient resource

## Current status
- Transformer-based autoregressive language model
- Train/validation split
- Perplexity tracking
- Gradient clipping
- Checkpoint saving (model + optimizer + metadata)
- Text generation CLI (bpegenerate)
- DirectML (Windows AMD) compatibility fallback
- Manual optimizer experimentation (SGD, Adam, , Lion, custom Adam variant)
- Tokenization pipeline with BPE

## Project files
- `bpetrain.py` — model training, checkpoint saving
- `bpegenerate.py` — one-shot CLI generation
- `transformer.py` — Transformer model definition
- `data.txt` — raw training corpus
- `train_tokenizer.py` — BPE tokenizer training
- `pretokenize.py` - data preprocessing using trained tokenizer

## Quick start
1. Install dependencies:

```sh
# CPU-only PyTorch (example):
pip install torch numpy

# Optional (Windows DirectML backend):
# pip install torch-directml

```

2. Train tokenizer:

```sh
python train_tokenizer.py --data data.txt
```

3. Preprocess text:

```sh
python pretokenize.py --input data.txt --output pretokenized.txt
```

4. Train model:

```sh
python bpetrain.py --data pretokenized.txt --ckpt checkpoint.pth
```

5. Generate text:

```sh
python bpegenerate.py --ckpt checkpoint.pth --start "Once upon a time" --length 200 --temp 0.8
```

## Next Development Direction

Short-term:
- Improve BPE tokenization and pre-tokenization scripts
- Add top-k and nucleus (top-p) sampling options
- Enable quantized inference for faster generation
- Expand datasets for structured Q&A and instruction tuning
- Add automated evaluation metrics and benchmarks

Long-term:
- Instruction tuning
- Larger dataset
- Deeper transformer
- CUDA/ROCm-native acceleration
- Evaluation metrics (perplexity + qualitative scoring)
