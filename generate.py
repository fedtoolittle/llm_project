import argparse
import pickle
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

#from model import CharRNN
from transformer import TransformerModel


REQUIRED_CKPT_KEYS = {"model_state", "vocab_size"}


def _load_checkpoint(checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError:
        # PyTorch >=2.6 defaults to weights_only=True, which can reject
        # checkpoints containing Python objects (e.g., vocab dictionaries).
        # Re-load with weights_only=False for trusted local checkpoints.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dictionary.")

    missing = REQUIRED_CKPT_KEYS.difference(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {sorted(missing)}")

    ckpt_version = ckpt.get("checkpoint_version", 1)
    if ckpt_version not in (1, 2):
        raise ValueError(f"Unsupported checkpoint_version={ckpt_version}")

    return ckpt


def _coerce_mappings(ckpt):
    # Robustly construct char_to_idx (str->int) and idx_to_char (int->str)
    cti = ckpt.get("char_to_idx")
    iti = ckpt.get("idx_to_char")
    if isinstance(cti, dict) and all(isinstance(k, str) for k in cti.keys()):
        # normal: char -> int
        char_to_idx = cti
        idx_to_char = {int(v): k for k, v in cti.items()}
        return char_to_idx, idx_to_char
    if isinstance(iti, dict):
        # if idx_to_char uses string keys, coerce to int keys
        try:
            idx_to_char = {int(k): v for k, v in iti.items()}
            char_to_idx = {v: int(k) for k, v in iti.items()}
            return char_to_idx, idx_to_char
        except Exception:
            pass
    # fallback: rebuild from data.txt
    text = Path("data.txt").read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def generate_from_checkpoint(checkpoint_path, start_seq, max_len=200, temperature=1.0, device=None):
    """Generate text using a checkpoint-bound context window.

    Maximum supported context length equals the checkpoint/model ``max_len``.
    """
    ckpt = _load_checkpoint(checkpoint_path)
    # allow caller to override device; otherwise prefer CUDA, else CPU
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = int(ckpt["vocab_size"])
    embed_size = ckpt.get("embed_size", 128)
    hidden_size = ckpt.get("hidden_size", 256)
    num_layers = ckpt.get("num_layers", 2)

    char_to_idx, idx_to_char = _coerce_mappings(ckpt)

    #model = CharRNN(vocab_size, embed_size, hidden_size, num_layers).to(device)
    num_heads = ckpt.get("num_heads", 4)
    # Determine the positional embedding length to use for model construction.
    # Prefer an explicit "max_len" saved in the checkpoint; otherwise
    # infer from the saved `pos_emb.weight` shape in the stored state_dict;
    # fall back to the saved sequence_length or 512. This is also the
    # maximum supported decoding context length.
    max_seq_len = ckpt.get("max_len")
    if max_seq_len is None:
        model_state = ckpt.get("model_state", {})
        pos_key = None
        for k in model_state.keys():
            if k.endswith("pos_emb.weight"):
                pos_key = k
                break
        if pos_key is not None:
            try:
                max_seq_len = int(model_state[pos_key].shape[0])
            except Exception:
                max_seq_len = None

    if max_seq_len is None:
        max_seq_len = ckpt.get("sequence_length", 512)

    model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_seq_len,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # normalize temperature safely
    temperature = max(1e-8, float(temperature))

    seq = [char_to_idx.get(ch, 0) for ch in start_seq]
    if len(seq) == 0:
        seq = [0]
    if len(seq) > max_seq_len:
        warnings.warn(
            "Start prompt exceeds checkpoint context window; truncating to the most recent "
            f"{max_seq_len} tokens (received {len(seq)}).",
            stacklevel=2,
        )
        seq = seq[-max_seq_len:]
        start_seq = start_seq[-max_seq_len:]

    input_tensor = torch.tensor([seq], dtype=torch.long, device=device)
    #hidden = model.init_hidden(1, device=device)

    out_chars = list(start_seq)
    #cur_input = input_tensor[:, -1:].clone()

    with torch.no_grad():
        for _ in range(max_len):
            context = input_tensor[:, -max_seq_len:]
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_idx]], device=device)], 
                dim=1
            )
            out_chars.append(idx_to_char.get(int(next_idx), "?"))
            cur_input = torch.tensor([[next_idx]], device=device)

    return "".join(out_chars)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate text from saved char-level RNN checkpoint")
    p.add_argument("--ckpt", default="checkpoint.pth")
    p.add_argument("--start", default="Sing, ")
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--temp", type=float, default=0.8)
    args = p.parse_args()

    print(generate_from_checkpoint(args.ckpt, args.start, max_len=args.length, temperature=args.temp))
