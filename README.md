# b_lm_project

Small char-level language-model preprocessing & data utilities.

## Project files
- `train.py` — preprocessing & dataset creation (reads `data.txt`, builds char vocab and sequences)
- `generate.py` — sample/generate text from saved checkpoints (one-shot CLI + reusable generator)
- `interactive_generate.py` — interactive prompt loop that reuses a loaded checkpoint
- `model.py` — model definition (`CharRNN`)
- `data.txt` — raw training text

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
pip install torch numpy torchvision

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

## Notes
- `generate.py` now robustly coerces saved mappings and uses safe sampling to avoid KeyErrors.
- If you use an AMD GPU on Windows, `torch-directml` can be used as a backend; install it separately.

## License
This project is a small example and may be used for experimentation.
