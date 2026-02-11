import argparse
import unicodedata

from generate import CheckpointGenerator


QUIT_COMMANDS = {"q", "quit", "exit", ":q"}


def normalize_prompt(text: str) -> str:
    """Normalize and sanitize user input for generation."""
    normalized = unicodedata.normalize("NFC", text)
    # Encode/decode with replacement to keep prompt unicode-safe for model I/O.
    normalized = normalized.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return normalized.strip("\n\r")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Interactive text generation from a saved checkpoint"
    )
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", default="checkpoint.pth")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--default-prompt", default="Sing, ")
    return parser


def main():
    args = build_parser().parse_args()

    generator = CheckpointGenerator(checkpoint_path=args.checkpoint, device=args.device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print("Type a prompt and press Enter. Type 'quit'/'exit' to stop.")

    turn = 1
    while True:
        try:
            raw = input("\nYou> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive generation.")
            break

        normalized = normalize_prompt(raw)
        if normalized.lower() in QUIT_COMMANDS:
            print("Goodbye.")
            break

        if not normalized.strip():
            if args.default_prompt:
                print(f"(empty prompt; using default prompt: {args.default_prompt!r})")
                normalized = args.default_prompt
            else:
                print("Prompt is empty. Please enter non-empty text.")
                continue

        output = generator.generate(
            start_seq=normalized,
            max_len=args.length,
            temperature=args.temp,
        )

        print(f"\nModel [{turn}]>\n{output}")
        turn += 1


if __name__ == "__main__":
    main()
