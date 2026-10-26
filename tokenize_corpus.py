"""Encode data/pretrain.txt (+ data/train.txt) into data/pretrain.bin (uint16 token ids)."""
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

OUT = Path("data/pretrain.bin")
CHUNK = 1 << 20  # characters per encode call

def main():
    tok = Tokenizer.from_file("checkpoints/tokenizer.json")
    assert tok.get_vocab_size() < 2**16, "uint16 storage needs vocab < 65536"
    sources = [p for p in (Path("data/pretrain.txt"), Path("data/train.txt")) if p.exists()]
    if not sources:
        raise SystemExit("No pretraining text. Run prepare_hindi.py")
    total = 0
    with OUT.open("wb") as out:
        for src in sources:
            with src.open(encoding="utf-8") as f:
                carry = ""
                while True:
                    block = f.read(CHUNK)
                    if not block:
                        break
                    block = carry + block
                    cut = block.rfind("\n\n")  # split on document boundaries only
                    if cut == -1:
                        carry = block
                        continue
                    text, carry = block[:cut + 2], block[cut + 2:]
                    ids = np.asarray(tok.encode(text).ids, dtype=np.uint16)
                    out.write(ids.tobytes()); total += len(ids)
                if carry.strip():
                    ids = np.asarray(tok.encode(carry).ids, dtype=np.uint16)
                    out.write(ids.tobytes()); total += len(ids)
    print(f"Wrote {OUT}: {total:,} tokens")

if __name__ == "__main__":
    main()
