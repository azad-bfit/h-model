import json
from pathlib import Path

DATA = Path("data")
OUT = DATA / "all_train.txt"
SPECIAL = {"user": "<|user|>", "assistant": "<|assistant|>", "system": "<|system|>"}

def main():
    parts = []
    txt = DATA / "train.txt"
    if txt.exists():
        raw = txt.read_text(encoding="utf-8").strip()
        if raw:
            parts.append(raw)

    chats = DATA / "chat.jsonl"
    if chats.exists():
        for line_no, line in enumerate(chats.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                formatted = []
                for message in item["messages"]:
                    role = message["role"]
                    if role not in SPECIAL:
                        raise ValueError(f"Unsupported role: {role}")
                    formatted.append(f"{SPECIAL[role]}\n{message['content'].strip()}\n")
                parts.append("\n".join(formatted) + "<|end|>")
            except Exception as exc:
                raise ValueError(f"Bad JSONL at {chats}:{line_no}: {exc}") from exc

    if not parts:
        raise SystemExit("Add text to data/train.txt or conversations to data/chat.jsonl")
    OUT.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print("Tokenizer training text prepared. Keep chat.jsonl: train.py reads its message boundaries for loss masking.")

if __name__ == "__main__":
    main()
