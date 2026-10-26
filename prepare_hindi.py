"""Download Hindi Wikipedia (wikimedia/wikipedia, CC BY-SA) and write cleaned NFC text to data/pretrain.txt."""
import os
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

MAX_MB = float(os.getenv("MAX_MB", "60"))
SHARD = "20231101.hi/train-00000-of-00002.parquet"
OUT = Path("data/pretrain.txt")
MIN_CHARS = 200

def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def main():
    path = hf_hub_download("wikimedia/wikipedia", SHARD, repo_type="dataset")
    OUT.parent.mkdir(exist_ok=True)
    limit = int(MAX_MB * 1024 * 1024)
    written, docs = 0, 0
    pf = pq.ParquetFile(path)
    with OUT.open("w", encoding="utf-8") as f:
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                text = clean(text)
                if len(text) < MIN_CHARS:
                    continue
                f.write(text + "\n\n")
                written += len(text.encode("utf-8")) + 2
                docs += 1
            if written >= limit:
                break
    print(f"Wrote {OUT}: {docs:,} articles, {written / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
