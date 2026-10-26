# T2T from scratch — assistant-only loss version

This update still initializes the Transformer randomly and trains a custom BPE tokenizer from your data. It changes training so the loss is calculated only on assistant response tokens and the assistant `<|end|>` token.

## Replace files
Replace `prepare_data.py` and `train.py` with the versions in this package. Keep `model.py`, `train_tokenizer.py`, `chat.py`, and `api.py` from your original project.

## Commands
```bash
python prepare_data.py
python train_tokenizer.py
python train.py
python chat.py
```

Because the tokenizer is retrained, the model must also be trained from scratch after it. Do not use the old checkpoint with the new tokenizer.

## Small Colab dataset tip
If you have only a few examples, use a shorter block size:
```bash
BLOCK_SIZE=32 BATCH_SIZE=4 STEPS=2000 python train.py
```
On Colab/PowerShell, environment variable syntax differs. In PowerShell:
```powershell
$env:BLOCK_SIZE="32"; $env:BATCH_SIZE="4"; $env:STEPS="2000"; python train.py
```

`data/chat.jsonl` must contain one JSON object per line, e.g.:
```json
{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"Hello! How can I help you?"}]}
```

Important limitations:
- This is an educational implementation, not a ChatGPT-scale training system.
- Use many diverse, high-quality, permission-cleared conversations.
- With fewer than five conversations, validation reuses data and is not a meaningful generalization metric.
- The current `chat.py` expects the `<|assistant|>` and `<|end|>` special tokens, which are included by `train_tokenizer.py`.
- The loss is masked for non-assistant tokens, but examples are concatenated and batches are sampled from the resulting stream. For larger training runs, upgrade to conversation-aware batching, assistant-only target masking, a proper validation split, and checkpoint resume.
