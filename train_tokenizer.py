import os
from pathlib import Path

from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders, normalizers

DATA = [Path("data/all_train.txt"), Path("data/pretrain.txt")]
OUT = Path("checkpoints/tokenizer.json")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "16000"))

# GPT-2's default split regex uses \p{L}, which excludes Devanagari matras (\p{M}) and
# would cut words apart. Keep letters and combining marks together.
WORD = r" ?[\p{L}\p{M}]+|[\p{N}]+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+"

def main():
    files = [str(p) for p in DATA if p.exists()]
    if not Path("data/all_train.txt").exists():
        raise SystemExit("Run: python prepare_data.py")
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(WORD), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    special = ["<|unk|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=special,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    print("Training tokenizer on:", files)
    tok.train(files, trainer)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT))
    print(f"Saved tokenizer to {OUT}; vocab size={tok.get_vocab_size()}")

if __name__ == "__main__":
    main()
