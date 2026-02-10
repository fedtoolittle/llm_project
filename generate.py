import torch
import torch.nn.functional as F
from model import CharRNN


def generate_from_checkpoint(checkpoint_path, start_seq, max_len=200, temperature=1.0, device=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = ckpt["vocab_size"]
    embed_size = ckpt.get("embed_size", 128)
    hidden_size = ckpt.get("hidden_size", 256)
    num_layers = ckpt.get("num_layers", 2)
    char_to_idx = ckpt["char_to_idx"]
    idx_to_char = ckpt["idx_to_char"]

    model = CharRNN(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq = [char_to_idx.get(ch, 0) for ch in start_seq]
    input_tensor = torch.tensor([seq], dtype=torch.long, device=device)
    # GRU returns a single hidden tensor (num_layers, batch, hidden_size)
    hidden = model.init_hidden(1, device=device)

    out_chars = list(start_seq)
    cur_input = input_tensor[:, -1:].clone()

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(cur_input, hidden)            # (1,1,vocab) and hidden: (num_layers, 1, hidden)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            out_chars.append(idx_to_char[next_idx])
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
