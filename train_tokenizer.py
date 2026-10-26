from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA = Path("data/all_train.txt")
OUT = Path("checkpoints/tokenizer.json")
VOCAB_SIZE = 2048

def main():
    if not DATA.exists():
        raise SystemExit("Run: python prepare_data.py")
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = ["<|unk|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=1, special_tokens=special)
    tok.train([str(DATA)], trainer)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT))
    print(f"Saved tokenizer to {OUT}; vocab size={tok.get_vocab_size()}")

if __name__ == "__main__":
    main()
