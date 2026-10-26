"""Print tensor shapes through the model and the attention weights for a short sentence."""
import sys
import torch
from tokenizers import Tokenizer
from model import GPT

def main():
    text = " ".join(sys.argv[1:]) or "भारत एक विशाल देश है"
    tok = Tokenizer.from_file("checkpoints/tokenizer.json")
    ckpt = torch.load("checkpoints/model.pt", map_location="cpu", weights_only=True)
    model = GPT(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    enc = tok.encode(text)
    idx = torch.tensor([enc.ids])
    print("tokens :", enc.tokens)
    print("ids    :", tuple(idx.shape))
    for blk in model.blocks:
        blk.attn.keep_attn = True
    with torch.no_grad():
        x = model.token_emb(idx) + model.pos_emb(torch.arange(idx.size(1)))
        print("embeddings:", tuple(x.shape))
        for i, blk in enumerate(model.blocks):
            x = blk(x)
            print(f"after block {i}:", tuple(x.shape))
        logits = model.head(model.ln_f(x))
    print("logits :", tuple(logits.shape))
    attn = model.blocks[-1].attn.last_attn[0]  # (heads, T, T)
    print("\nlast-layer attention, head 0 (rows = query token, cols = key token):")
    print("          " + " ".join(f"{i:>5d}" for i in range(attn.size(-1))))
    for t in range(attn.size(1)):
        print(f"{t:>2d} {enc.tokens[t][:6]:<7}" + " ".join(f"{v:5.2f}" for v in attn[0, t].tolist()))

if __name__ == "__main__":
    main()
