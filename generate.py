import torch
import torch.nn.functional as F
from model import CharRNN
from pathlib import Path


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
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # allow caller to override device; otherwise prefer CUDA, else CPU
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = ckpt.get("vocab_size")
    embed_size = ckpt.get("embed_size", 128)
    hidden_size = ckpt.get("hidden_size", 256)
    num_layers = ckpt.get("num_layers", 2)

    char_to_idx, idx_to_char = _coerce_mappings(ckpt)

    model = CharRNN(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # normalize temperature safely
    temperature = max(1e-8, float(temperature))

    seq = [char_to_idx.get(ch, 0) for ch in start_seq]
    if len(seq) == 0:
        seq = [0]
    input_tensor = torch.tensor([seq], dtype=torch.long, device=device)
    hidden = model.init_hidden(1, device=device)

    out_chars = list(start_seq)
    cur_input = input_tensor[:, -1:].clone()

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(cur_input, hidden)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            out_chars.append(idx_to_char.get(int(next_idx), "?"))
            cur_input = torch.tensor([[next_idx]], device=device)

    return "".join(out_chars)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate text from saved char-level RNN checkpoint")
    p.add_argument("--ckpt", default="checkpoint.pth")
    p.add_argument("--start", default="Sing, ")
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--temp", type=float, default=0.8)
    args = p.parse_args()

    print(generate_from_checkpoint(args.ckpt, args.start, max_len=args.length, temperature=args.temp))
