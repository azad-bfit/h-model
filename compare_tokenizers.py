"""Compare token counts: our Hindi tokenizer vs GPT-2's, and check round-trip fidelity."""
import unicodedata
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SAMPLES = [
    "How are you?",
    "आप कैसे हैं?",
    "भारत एक विशाल देश है जिसकी राजधानी नई दिल्ली है।",
    "Python is a programming language used for machine learning.",
    "पायथन एक प्रोग्रामिंग भाषा है जिसका उपयोग मशीन लर्निंग के लिए किया जाता है।",
]

def main():
    ours = Tokenizer.from_file("checkpoints/tokenizer.json")
    gpt2 = Tokenizer.from_file(hf_hub_download("openai-community/gpt2", "tokenizer.json"))
    print(f"{'ours':>5} {'gpt2':>5}  text")
    for s in SAMPLES:
        enc = ours.encode(s)
        assert ours.decode(enc.ids) == unicodedata.normalize("NFC", s), f"round-trip failed: {s}"
        print(f"{len(enc.ids):5d} {len(gpt2.encode(s).ids):5d}  {s}")
    print("tokens for one sentence:", ours.encode(SAMPLES[2]).tokens)

if __name__ == "__main__":
    main()
