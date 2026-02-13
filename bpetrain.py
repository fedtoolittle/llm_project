import argparse
from email import parser
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch import device, nn
from tokenizers import Tokenizer

#import model
from transformer import TransformerModel
from optim_lion import ManualLion



# -------------------------------------------------
# Utilities
# -------------------------------------------------

def get_batch(data, batch_size, seq_len, device):
    """
    Randomly sample batch of token chunks.
    """
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, eval_iters, batch_size, seq_len, device):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(data, batch_size, seq_len, device)
        logits = model(xb)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# -------------------------------------------------
# Lazy shard loader
# -------------------------------------------------
def load_shards(shard_dir, pattern):
    shard_dir = Path(shard_dir)  # ensure it's a Path object
    shard_paths = list(shard_dir.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found matching {pattern}")
    random.shuffle(shard_paths)
    for path in shard_paths:
        yield torch.load(path, weights_only=True)  # one shard at a time


# -------------------------------------------------
# Train/Val split generator per shard
# -------------------------------------------------
def shard_train_val_split(shard_gen, train_ratio=0.9):
    """
    Yields (train_data, val_data) for each shard.
    """
    for shard in shard_gen:
        n = len(shard)
        split_idx = int(train_ratio * n)
        yield shard[:split_idx], shard[split_idx:]

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="wikitext103_train.txt")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--embed_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--shard_dir", default=".\shards")
    parser.add_argument("--shard_pattern", type=str, default="wiki_shard_*.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    args = parser.parse_args()

# -------------------------------------------------
# Device selection
# -------------------------------------------------

    def pick_device(user_device=None):
        if user_device:
            return torch.device(user_device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        # try:
        #     import torch_directml
        #     dml = torch_directml.device()
        #     print("Using DirectML:", dml)
        #     return dml
        # except Exception:
        #     pass

        return torch.device("cpu")
    device = pick_device(args.device)
    print("Using device:", device)
    # print("Args OK", args.max_iters)
    

# -------------------------------------------------
# Load tokenizer
# -------------------------------------------------
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

# -------------------------------------------------
# Model
# -------------------------------------------------

    model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.sequence_length,
    ).to(device)


    #optimizer = ManualLion(model.parameters(), lr=args.lr, weight_decay=1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters:", count_parameters(model))

    best_val_loss = float("inf")
    patience = 10                 # number of evals with no improvement
    patience_counter = 0
    min_delta = 1e-4             # minimum improvement threshold

# -------------------------------------------------
# Training loop
# -------------------------------------------------

    step = 0

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"] + 1
        print(f"Resumed from checkpoint {args.resume_checkpoint} at step {step}")

    while step < args.max_iters:
        # Re-create shard generator each epoch
        shard_gen = load_shards(args.shard_dir, args.shard_pattern)

        for train_shard, val_shard in shard_train_val_split(shard_gen, args.train_ratio):
            # Convert shards to tensors on CPU for batch sampling
            train_shard = train_shard.clone().detach().long()
            val_shard = val_shard.clone().detach().long()
            
            # Loop over batches within this shard
            while step < args.max_iters:
                xb, yb = get_batch(train_shard, args.batch_size, args.sequence_length, device)
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1

                if step % args.eval_interval == 0:
                    train_loss = estimate_loss(model, train_shard, args.eval_iters, args.batch_size, args.sequence_length, device)
                    val_loss = estimate_loss(model, val_shard, args.eval_iters, args.batch_size, args.sequence_length, device)
                    print(f"Step {step} | train_loss {train_loss:.4f} | train_ppl {math.exp(train_loss):.2f} | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}")
                    torch.save({
                        "model_state_dict": model.state_dict(), 
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        }, f"checkpoint.pth")
                    
                # Break out of batch loop if step exceeds max_iters
                if step >= args.max_iters:
                    break
            # ---- Early stopping ----
            # if val_loss < best_val_loss - min_delta:
            #     best_val_loss = val_loss
            #     patience_counter = 0
            #     torch.save(model.state_dict(), "best_model.pth")
            #     print("New best model saved.")
            # else:
            #     patience_counter += 1
            #     print(f"No improvement. Patience: {patience_counter}/{patience}")

            #     if patience_counter >= patience:
            #         print("Early stopping triggered.")
            #         break
   

    # -------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "embed_size": args.embed_size,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "max_len": args.sequence_length,
        },
        "checkpoint_fin.pth",
    )

    if Path("best_model.pth").exists():
        model.load_state_dict(torch.load("best_model.pth"))
    print("Loaded best validation model.")

    print("Training complete. Checkpoint saved.")


if __name__ == "__main__":
    main()
