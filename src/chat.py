"""Interactive host chat with a trained Indonesian model, no hardware needed.

Mirrors the firmware's prompt format (`user: <line>\\nJawaban:`) so you can
sanity-check a checkpoint before exporting/flashing it. Reuses the exact
generation path from `sample.py`.

Usage:
    python src/chat.py --run runs/ple-idfeas-s0.pt
    python src/chat.py --run runs/ple-idfeas-s0.pt --tok data/bpe_id32768.json
"""

import argparse
import os

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from train import get_device

HERE = os.path.dirname(os.path.abspath(__file__))


def reply(model, tok, line, max_tokens, temperature, top_k, device):
    prompt = f"user: {line}\nJawaban:"
    ids = torch.tensor([tok.encode(prompt).ids], device=device)
    ids = ids[:, -model.cfg.seq_len :]
    out = model.generate(ids, max_tokens, temperature=temperature, top_k=top_k)
    text = tok.decode(out[0].tolist())
    # Show only the continuation the model produced after the prompt.
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to a runs/*.pt checkpoint")
    ap.add_argument("--tok", default=os.path.join(HERE, "..", "data", "bpe_id32768.json"))
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.run, map_location=device, weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(args.tok)

    b = model.param_budget()
    print(f"{os.path.basename(args.run)}  core={b['core']:,} table={b['table']:,}")
    print("Ketik pesan (Ctrl+C atau 'keluar' untuk berhenti).\n")
    try:
        while True:
            try:
                line = input("user: ").strip()
            except EOFError:
                break
            if not line or line in ("keluar", "exit", "quit"):
                break
            print("Jawaban:", reply(model, tok, line, args.tokens,
                                    args.temperature, args.top_k, device), "\n")
    except KeyboardInterrupt:
        pass
    print("\nselesai.")


if __name__ == "__main__":
    main()
