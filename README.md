# Tiny T2T chatbot trained from scratch

This project trains a small decoder-only Transformer with **randomly initialized model weights** and a **BPE tokenizer trained on your corpus**. No pretrained model is downloaded.

## 0. Reality check

This is a learning-scale GPT-style language model, not a ChatGPT-equivalent model. The included dataset is only a format example and will not produce useful conversational intelligence. Add a large, clean, legally usable corpus before training.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- 8 GB system RAM minimum for tiny experiments; 16–32 GB is more comfortable
- NVIDIA GPU with 8–16 GB VRAM recommended for this small configuration
- Disk: reserve 5–20 GB for environment, data, checkpoints, and logs
- CPU-only works for debugging but can be very slow

## 2. Install

Create a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# Linux:
source .venv/bin/activate
```

Install PyTorch using the command recommended for your OS/CUDA at https://pytorch.org/get-started/locally/ . Then install the rest:

```bash
pip install -r requirements.txt
```

Check GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Prepare your data

- Add general text to `data/train.txt`.
- Add one conversation per line to `data/chat.jsonl` using this format:

```json
{"messages":[{"role":"user","content":"What is photosynthesis?"},{"role":"assistant","content":"Photosynthesis is the process plants use to turn light, water, and carbon dioxide into sugars and oxygen."}]}
```

Use thousands or millions of high-quality examples, not just the samples included. Remove duplicates, private information, spam, and low-quality text. Split out a held-out validation set in a production-quality pipeline. Use only data you have permission to train on.

## 4. Build data and tokenizer

```bash
python prepare_data.py
python train_tokenizer.py
```

Tokenizer vocabulary is trained from your own `data/all_train.txt`.

## 5. Train from random weights

```bash
python train.py
```

Defaults: 4 layers, 4 attention heads, 128 embedding dimensions, 128-token context, batch size 16, 3,000 steps. This is deliberately small. Adjust with environment variables, for example:

Linux/macOS:
```bash
STEPS=10000 BATCH_SIZE=16 BLOCK_SIZE=256 python train.py
```

Windows PowerShell:
```powershell
$env:STEPS="10000"; $env:BATCH_SIZE="16"; $env:BLOCK_SIZE="256"; python train.py
```

If CUDA runs out of memory, reduce `BATCH_SIZE`, `BLOCK_SIZE`, `N_LAYER`, or `N_EMBD`. Keep `N_EMBD` divisible by `N_HEAD`.

## 6. Chat locally

```bash
python chat.py
```

Type `/exit` to stop.

## 7. Start API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs`, or test:

```bash
curl -X POST http://127.0.0.1:8000/generate ^
  -H "Content-Type: application/json" ^
  -d "{\\"prompt\\":\\"What is Python?\\",\\"max_new_tokens\\":80}"
```

Use `curl` with backslash line continuation on Linux/macOS instead of `^`.

## 8. Next steps toward a chatbot

1. Pretrain on a broad corpus so the model learns language patterns.
2. Continue training on curated prompt/response conversation examples.
3. Use instruction-format examples consistently; add system messages if needed.
4. Evaluate on held-out questions and safety tests.
5. For stronger instruction following, preference optimization and human evaluation are typically needed.
6. For public deployment, add authentication, rate limits, logging controls, input validation, and monitoring.

## Important implementation note

This starter uses one concatenated text stream and standard next-token loss. It does **not** mask loss to assistant-only tokens, and it does not implement a sophisticated instruction-tuning pipeline. For serious chat training, build an assistant-only loss mask, proper validation splits, checkpoint-resume support, learning-rate scheduling, and data deduplication.
