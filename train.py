import json, math, os, random, time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from model import GPT

BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "128"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
STEPS = int(os.getenv("STEPS", "3000"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "200"))
LR = float(os.getenv("LR", "3e-4"))
N_LAYER = int(os.getenv("N_LAYER", "4"))
N_HEAD = int(os.getenv("N_HEAD", "4"))
N_EMBD = int(os.getenv("N_EMBD", "128"))
SEED = 1337
IGNORE_INDEX = -100

ROLE_TOKENS = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
}

def load_conversations(tokenizer):
    path = Path("data/chat.jsonl")
    if not path.exists():
        raise SystemExit("Missing data/chat.jsonl. Add JSONL conversations first.")

    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            messages = item["messages"]
            ids, loss_mask = [], []
            for msg in messages:
                role = msg["role"]
                if role not in ROLE_TOKENS:
                    raise ValueError(f"Unsupported role {role!r}")
                role_id = tokenizer.token_to_id(ROLE_TOKENS[role])
                end_id = tokenizer.token_to_id("<|end|>")
                if role_id is None or end_id is None:
                    raise ValueError("Special tokens missing. Re-run train_tokenizer.py.")
                # Role marker and message text are encoded separately so the role marker is exact.
                ids.append(role_id)
                loss_mask.append(False)

                content_ids = tokenizer.encode(msg["content"].strip()).ids
                ids.extend(content_ids)
                # Only assistant response tokens contribute to loss.
                loss_mask.extend([role == "assistant"] * len(content_ids))

                # Newline is represented by the tokenizer and remains context-only.
                newline_ids = tokenizer.encode("\n").ids
                ids.extend(newline_ids)
                loss_mask.extend([False] * len(newline_ids))

                # End token is a target only after assistant messages.
                ids.append(end_id)
                loss_mask.append(role == "assistant")

            if len(ids) >= 2:
                records.append((ids, loss_mask))
        except Exception as exc:
            raise ValueError(f"Bad JSONL at {path}:{line_no}: {exc}") from exc

    if len(records) < 2:
        print("Warning: fewer than two conversations; validation will reuse a tail of training data.")
    return records

def pack_records(records):
    # Concatenate conversation streams; preserve each token's assistant-only loss mask.
    all_ids, all_mask = [], []
    for ids, mask in records:
        all_ids.extend(ids)
        all_mask.extend(mask)
    return torch.tensor(all_ids, dtype=torch.long), torch.tensor(all_mask, dtype=torch.bool)

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer_path = Path("checkpoints/tokenizer.json")
    if not tokenizer_path.exists():
        raise SystemExit("Missing tokenizer. Run prepare_data.py then train_tokenizer.py.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    records = load_conversations(tokenizer)
    random.shuffle(records)

    if len(records) >= 5:
        val_count = max(1, int(len(records) * 0.1))
        val_records, train_records = records[:val_count], records[val_count:]
    else:
        # Tiny toy datasets cannot support a meaningful held-out validation set.
        train_records = records
        val_records = records[-1:]

    train_ids, train_mask = pack_records(train_records)
    val_ids, val_mask = pack_records(val_records)

    # Tiny demo datasets often have a validation conversation shorter than the
    # context window. Fall back to a tail of the training stream for smoke tests.
    # This is NOT a valid generalization evaluation; use more data for real validation.
    if len(val_ids) < BLOCK_SIZE + 2:
        print("Warning: validation data is too short; using a training-data tail for smoke-test validation.")
        val_ids, val_mask = train_ids, train_mask

    def get_batch(source_ids, source_mask):
        # Each target is the next token; its loss is enabled only if that target
        # token belongs to an assistant message (or is its <|end|> marker).
        max_start = len(source_ids) - BLOCK_SIZE - 1
        if max_start < 0:
            raise SystemExit(
                f"Not enough tokens for BLOCK_SIZE={BLOCK_SIZE}. "
                f"Add more conversations or set BLOCK_SIZE=32."
            )
        # Retry if a sampled batch contains no assistant targets (cross-entropy
        # over an all-ignored batch would be undefined).
        for _ in range(50):
            starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))
            x = torch.stack([source_ids[i:i + BLOCK_SIZE] for i in starts.tolist()])
            y_ids = torch.stack([source_ids[i + 1:i + BLOCK_SIZE + 1] for i in starts.tolist()])
            y_mask = torch.stack([source_mask[i + 1:i + BLOCK_SIZE + 1] for i in starts.tolist()])
            if y_mask.any():
                y = y_ids.clone()
                y[~y_mask] = IGNORE_INDEX
                return x.to(device), y.to(device)
        raise RuntimeError("Could not sample assistant target tokens. Add more assistant messages to data/chat.jsonl.")

    model = GPT(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=0.1,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training objective: assistant-only next-token loss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(10):
            xb, yb = get_batch(val_ids, val_mask)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(xb, yb)
            if loss is not None and math.isfinite(loss.item()):
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses) if losses else float("inf")

    start = time.time()
    model.train()
    for step in range(1, STEPS + 1):
        xb, yb = get_batch(train_ids, train_mask)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, loss = model(xb, yb)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("No assistant target tokens in batch. Check chat.jsonl formatting.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % EVAL_EVERY == 0:
            val_loss = evaluate()
            print(
                f"step {step:5d} | train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f} | elapsed {time.time()-start:.1f}s"
            )
            if math.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": tokenizer.get_vocab_size(),
                        "block_size": BLOCK_SIZE,
                        "n_layer": N_LAYER,
                        "n_head": N_HEAD,
                        "n_embd": N_EMBD,
                    },
                }, "checkpoints/model.pt")
    print("Saved best checkpoint to checkpoints/model.pt")

if __name__ == "__main__":
    main()
