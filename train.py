import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model import CharRNN


def build_dataset(text: str, sequence_length: int):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    encoded_text = [char_to_idx[ch] for ch in text]

    inputs = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length):
        inputs.append(encoded_text[i : i + sequence_length])
        targets.append(encoded_text[i + 1 : i + sequence_length + 1])

    if len(inputs) == 0:
        raise ValueError(
            f"Not enough data to build sequences: text length={len(encoded_text)}, "
            f"sequence_length={sequence_length}. Provide more text or reduce sequence length."
        )

    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets, vocab_size, char_to_idx, idx_to_char


def pick_device():
    # Prefer DirectML on Windows with AMD GPUs (torch-directml), fall back to CUDA/CPU.
    try:
        import torch_directml as dml  # type: ignore
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
    else:
        device = dml.device()
        print("Using DirectML device:", device)
    return device


def main():
    parser = argparse.ArgumentParser(description="Train char-level RNN")
    parser.add_argument("--data", default="data.txt")
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--ckpt", default="checkpoint.pth")
    args = parser.parse_args()

    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("--val-split must be in [0.0, 1.0).")

    text = Path(args.data).read_text(encoding="utf-8")
    if not isinstance(text, str):
        raise TypeError("Loaded text is not a string.")

    inputs, targets, vocab_size, char_to_idx, idx_to_char = build_dataset(text, args.sequence_length)

    total_size = inputs.size(0)
    val_size = int(total_size * args.val_split)
    if total_size > 1 and val_size == 0 and args.val_split > 0:
        val_size = 1
    train_size = total_size - val_size

    if train_size <= 0:
        raise ValueError(
            f"Training split is empty (total={total_size}, val_size={val_size}). Lower --val-split."
        )

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    if val_size > 0:
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        if val_ds is not None
        else None
    )

    # sample preview from full tensors (safe now because build_dataset validates non-empty)
    print("Input:", "".join(idx_to_char[i.item()] for i in inputs[0]))
    print("Target:", "".join(idx_to_char[i.item()] for i in targets[0]))
    print(f"Dataset sizes -> train: {train_size}, val: {val_size}")

    device = pick_device()

    model = CharRNN(vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_items = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits.view(-1, vocab_size), yb.view(-1))
            loss.backward()
            opt.step()
            batch_items = xb.size(0)
            train_loss_sum += loss.item() * batch_items
            train_items += batch_items

        train_loss = train_loss_sum / max(train_items, 1)

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_items = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, _ = model(xb)
                    loss = crit(logits.view(-1, vocab_size), yb.view(-1))
                    batch_items = xb.size(0)
                    val_loss_sum += loss.item() * batch_items
                    val_items += batch_items
            val_loss = val_loss_sum / max(val_items, 1)
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            print(
                f"Epoch {epoch+1}/{args.epochs} train_loss: {train_loss:.4f} train_ppl: {train_ppl:.2f} "
                f"val_loss: {val_loss:.4f} val_ppl: {val_ppl:.2f}"
            )
        else:
            val_loss = float("inf")
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            print(f"Epoch {epoch+1}/{args.epochs} train_loss: {train_loss:.4f} train_ppl: {train_ppl:.2f}")

        is_best = val_loader is None or val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(
                {
                    "checkpoint_version": 2,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "epoch": epoch + 1,
                    "vocab_size": vocab_size,
                    "embed_size": args.embed_size,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "sequence_length": args.sequence_length,
                    "best_val_loss": None if val_loader is None else float(best_val),
                },
                args.ckpt,
            )
            print(f"Saved checkpoint to {args.ckpt}")


if __name__ == "__main__":
    main()
