import json
from pathlib import Path

from chat_format import ROLE_TOKENS, format_text

DATA = Path("data")
OUT = DATA / "all_train.txt"

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
                    if message["role"] not in ROLE_TOKENS:
                        raise ValueError(f"Unsupported role: {message['role']}")
                    formatted.append(format_text(message["role"], message["content"]))
                parts.append("\n".join(formatted))
            except Exception as exc:
                raise ValueError(f"Bad JSONL at {chats}:{line_no}: {exc}") from exc

    if not parts:
        raise SystemExit("Add text to data/train.txt or conversations to data/chat.jsonl")
    OUT.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print("Tokenizer training text prepared. Keep chat.jsonl: train.py reads its message boundaries for loss masking.")

if __name__ == "__main__":
    main()
