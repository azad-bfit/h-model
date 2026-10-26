import json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from model import GPT
from chat_format import encode_message

BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "128"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
PRETRAIN_STEPS = int(os.getenv("PRETRAIN_STEPS", "2000"))
STEPS = int(os.getenv("STEPS", "1000"))  # chat fine-tuning steps
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "200"))
LR = float(os.getenv("LR", "6e-4"))            # pre-training peak LR
FT_LR = float(os.getenv("FT_LR", "1e-4"))      # fine-tuning peak LR
WARMUP = int(os.getenv("WARMUP", "100"))
RESUME = os.getenv("RESUME", "0") == "1"
N_LAYER = int(os.getenv("N_LAYER", "4"))
N_HEAD = int(os.getenv("N_HEAD", "4"))
N_EMBD = int(os.getenv("N_EMBD", "128"))
SEED = 1337
IGNORE_INDEX = -100
CKPT = Path("checkpoints")

def load_conversations(tokenizer):
    path = Path("data/chat.jsonl")
    if not path.exists():
        raise SystemExit("Missing data/chat.jsonl. Add JSONL conversations first.")

    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            ids, loss_mask = [], []
            for msg in json.loads(line)["messages"]:
                m_ids, m_mask = encode_message(tokenizer, msg["role"], msg["content"])
                ids.extend(m_ids)
                loss_mask.extend(m_mask)
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

def lr_at(step, total, peak):
    """Linear warmup, then cosine decay to 10% of peak."""
    if step <= WARMUP:
        return peak * step / WARMUP
    progress = (step - WARMUP) / max(1, total - WARMUP)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer_path = CKPT / "tokenizer.json"
    if not tokenizer_path.exists():
        raise SystemExit("Missing tokenizer. Run prepare_data.py then train_tokenizer.py.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # ---- chat data (fine-tuning) ----
    records = load_conversations(tokenizer)
    random.shuffle(records)
    if len(records) >= 5:
        val_count = max(1, int(len(records) * 0.1))
        val_records, train_records = records[:val_count], records[val_count:]
    else:
        # Tiny toy datasets cannot support a meaningful held-out validation set.
        train_records = records
        val_records = records[-1:]
        print("Warning: <5 conversations, so validation overlaps training data and best-checkpoint selection is not a generalization signal.")
    train_ids, train_mask = pack_records(train_records)
    val_ids, val_mask = pack_records(val_records)
    if len(val_ids) < BLOCK_SIZE + 2:
        print("Warning: validation data is too short; using a training-data tail for smoke-test validation.")
        val_ids, val_mask = train_ids, train_mask

    def chat_batch(source_ids, source_mask):
        # Each target is the next token; loss is enabled only on assistant tokens.
        max_start = len(source_ids) - BLOCK_SIZE - 1
        if max_start < 0:
            raise SystemExit(f"Not enough chat tokens for BLOCK_SIZE={BLOCK_SIZE}. Add conversations or lower BLOCK_SIZE.")
        for _ in range(50):
            starts = torch.randint(0, max_start + 1, (BATCH_SIZE,)).tolist()
            x = torch.stack([source_ids[i:i + BLOCK_SIZE] for i in starts])
            y = torch.stack([source_ids[i + 1:i + BLOCK_SIZE + 1] for i in starts])
            y_mask = torch.stack([source_mask[i + 1:i + BLOCK_SIZE + 1] for i in starts])
            if y_mask.any():
                y = y.clone()
                y[~y_mask] = IGNORE_INDEX
                return x.to(device), y.to(device)
        raise RuntimeError("Could not sample assistant target tokens. Add more assistant messages to data/chat.jsonl.")

    # ---- raw text data (pre-training) ----
    corpus_path = Path("data/pretrain.bin")
    corpus = None
    if PRETRAIN_STEPS > 0:
        if not corpus_path.exists():
            raise SystemExit("Missing data/pretrain.bin. Run prepare_hindi.py, train_tokenizer.py, tokenize_corpus.py (or set PRETRAIN_STEPS=0).")
        corpus = np.memmap(corpus_path, dtype=np.uint16, mode="r")
        split = int(len(corpus) * 0.98)
        pre_train, pre_val = corpus[:split], corpus[split:]
        if len(pre_val) < BLOCK_SIZE + 2:
            raise SystemExit("Pre-training corpus too small for BLOCK_SIZE.")
        print(f"Pre-training corpus: {len(corpus):,} tokens ({len(pre_val):,} held out)")

    def text_batch(data):
        starts = np.random.randint(0, len(data) - BLOCK_SIZE - 1, BATCH_SIZE)
        x = torch.from_numpy(np.stack([data[i:i + BLOCK_SIZE] for i in starts]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in starts]).astype(np.int64))
        return x.to(device), y.to(device)

    config = {
        "vocab_size": tokenizer.get_vocab_size(), "block_size": BLOCK_SIZE,
        "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
    }
    model = GPT(**config, dropout=0.1).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    CKPT.mkdir(exist_ok=True)

    def make_optimizer():
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
            lr=LR, betas=(0.9, 0.95))

    # ---- resume ----
    start_phase, start_step, opt_state = 0, 1, None
    if RESUME and (CKPT / "last.pt").exists():
        last = torch.load(CKPT / "last.pt", map_location=device, weights_only=True)
        if last["config"] != config:
            raise SystemExit("last.pt was trained with a different config/tokenizer; delete it or unset RESUME.")
        model.load_state_dict(last["model"])
        start_phase, start_step, opt_state = last["phase"], last["step"] + 1, last["optimizer"]
        print(f"Resumed phase {start_phase} at step {start_step}")

    @torch.no_grad()
    def evaluate(get_val):
        model.eval()
        losses = []
        for _ in range(10):
            xb, yb = get_val()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(xb, yb)
            if math.isfinite(loss.item()):
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses) if losses else float("inf")

    def run_phase(phase, name, steps, peak_lr, get_train, get_val, best_path):
        if steps <= 0 or phase < start_phase:
            return
        optimizer = make_optimizer()
        first = 1
        if phase == start_phase:
            first = start_step
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
        if first > steps:
            return
        best_val = float("inf")
        t0 = time.time()
        model.train()
        print(f"== {name}: {steps} steps, peak lr {peak_lr:g} ==")
        for step in range(first, steps + 1):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, peak_lr)
            xb, yb = get_train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(xb, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss.item()}) at {name} step {step}; try a lower LR.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step == 1 or step % EVAL_EVERY == 0 or step == steps:
                val_loss = evaluate(get_val)
                print(f"{name} step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} | elapsed {time.time()-t0:.1f}s")
                if math.isfinite(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.state_dict(), "config": config}, best_path)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "phase": phase, "step": step, "config": config}, CKPT / "last.pt")
        # continue the next phase from the best weights of this one
        best = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best["model"])

    print("Phase 0: next-token pre-training on raw text (loss on every token)")
    if corpus is not None:
        run_phase(0, "pretrain", PRETRAIN_STEPS, LR,
                  lambda: text_batch(pre_train), lambda: text_batch(pre_val), CKPT / "pretrain.pt")
    print("Phase 1: chat fine-tuning (assistant-only loss)")
    run_phase(1, "finetune", STEPS, FT_LR,
              lambda: chat_batch(train_ids, train_mask), lambda: chat_batch(val_ids, val_mask), CKPT / "model.pt")
    if STEPS <= 0:
        torch.save({"model": model.state_dict(), "config": config}, CKPT / "model.pt")
    print("Saved best checkpoint to checkpoints/model.pt")

if __name__ == "__main__":
    main()
