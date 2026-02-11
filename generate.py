import argparse
import pickle
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from transformer import TransformerModel


REQUIRED_CKPT_KEYS = {"model_state", "vocab_size"}


class CheckpointGenerator:
    """Load a checkpoint once and provide reusable text generation."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoint.pth",
        device: Optional[str] = None,
        allow_rebuild_vocab_from_data: bool = False,
        data_path: str = "data.txt",
    ):
        self.checkpoint_path = checkpoint_path
        self.ckpt = _load_checkpoint(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = int(self.ckpt["vocab_size"])
        self.embed_size = self.ckpt.get("embed_size", 128)
        self.hidden_size = self.ckpt.get("hidden_size", 256)
        self.num_layers = self.ckpt.get("num_layers", 2)
        self.num_heads = self.ckpt.get("num_heads", 4)

        self.char_to_idx, self.idx_to_char = _coerce_mappings(
            self.ckpt,
            allow_rebuild_from_data=allow_rebuild_vocab_from_data,
            data_path=data_path,
        )
        self.max_seq_len = _infer_max_seq_len(self.ckpt)

        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            embed_size=self.embed_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_len=self.max_seq_len,
        ).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state"])
        self.model.eval()

    def generate(self, start_seq: str, max_len: int = 300, temperature: float = 0.8) -> str:
        """Generate text using this checkpoint-bound context window."""
        temperature = max(1e-8, float(temperature))

        seq = [self.char_to_idx.get(ch, 0) for ch in start_seq]
        if len(seq) == 0:
            seq = [0]
        if len(seq) > self.max_seq_len:
            warnings.warn(
                "Start prompt exceeds checkpoint context window; truncating to the most recent "
                f"{self.max_seq_len} tokens (received {len(seq)}).",
                stacklevel=2,
            )
            seq = seq[-self.max_seq_len:]
            start_seq = start_seq[-self.max_seq_len:]

        input_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
        out_chars = list(start_seq)

        with torch.no_grad():
            for _ in range(max_len):
                context = input_tensor[:, -self.max_seq_len:]
                logits = self.model(context)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1).item()

                input_tensor = torch.cat(
                    [input_tensor, torch.tensor([[next_idx]], device=self.device)], dim=1
                )
                out_chars.append(self.idx_to_char.get(int(next_idx), "?"))

        return "".join(out_chars)


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


def _coerce_mappings(ckpt, allow_rebuild_from_data=False, data_path="data.txt"):
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
    if not allow_rebuild_from_data:
        raise ValueError(
            "Checkpoint is missing usable char/index mappings. "
            "Provide checkpoints with `char_to_idx` or `idx_to_char`, or rerun with "
            "--allow-rebuild-vocab-from-data and --data for legacy recovery."
        )

    warnings.warn(
        "Rebuilding vocabulary from data file for compatibility recovery. "
        "This may mismatch the original training vocabulary and produce degraded output.",
        stacklevel=2,
    )
    text = Path(data_path).read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def _infer_max_seq_len(ckpt):
    """Infer positional/context length supported by the checkpoint/model."""
    max_seq_len = ckpt.get("max_len")
    if max_seq_len is None:
        model_state = ckpt.get("model_state", {})
        pos_key = None
        for key in model_state.keys():
            if key.endswith("pos_emb.weight"):
                pos_key = key
                break
        if pos_key is not None:
            try:
                max_seq_len = int(model_state[pos_key].shape[0])
            except Exception:
                max_seq_len = None

    if max_seq_len is None:
        max_seq_len = ckpt.get("sequence_length", 512)

    return int(max_seq_len)


def generate_from_checkpoint(
    checkpoint_path,
    start_seq,
    max_len=300,
    temperature=0.8,
    device=None,
    allow_rebuild_vocab_from_data=False,
    data_path="data.txt",
):
    """Generate text from a checkpoint path in one call."""
    generator = CheckpointGenerator(
        checkpoint_path=checkpoint_path,
        device=device,
        allow_rebuild_vocab_from_data=allow_rebuild_vocab_from_data,
        data_path=data_path,
    )
    return generator.generate(start_seq=start_seq, max_len=max_len, temperature=temperature)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate text from saved Transformer checkpoint"
    )
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", default="checkpoint.pth")
    parser.add_argument("--start", default="Sing, ")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-rebuild-vocab-from-data",
        action="store_true",
        help=(
            "Allow rebuilding vocabulary from --data when checkpoint mappings are missing. "
            "Use only for legacy recovery; output quality may degrade due to vocab mismatch."
        ),
    )
    parser.add_argument(
        "--data",
        default="data.txt",
        help="Path used only with --allow-rebuild-vocab-from-data.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generator = CheckpointGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        allow_rebuild_vocab_from_data=args.allow_rebuild_vocab_from_data,
        data_path=args.data,
    )
    print(generator.generate(args.start, max_len=args.length, temperature=args.temp))


if __name__ == "__main__":
    main()
