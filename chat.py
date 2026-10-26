import torch
from tokenizers import Tokenizer
from model import GPT

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file("checkpoints/tokenizer.json")
    ckpt = torch.load("checkpoints/model.pt", map_location=device, weights_only=True)
    model = GPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    user_id = tok.token_to_id("<|user|>")
    assistant_id = tok.token_to_id("<|assistant|>")
    end_id = tok.token_to_id("<|end|>")
    print("Tiny from-scratch model. Type /exit to quit.")
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in {"/exit", "/quit"}:
            break
        prefix = f"<|user|>\n{prompt}\n<|assistant|>\n"
        ids = tok.encode(prefix).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=120, temperature=0.8, top_k=40, stop_id=end_id)
        new_ids = out[0, len(ids):].tolist()
        answer = tok.decode(new_ids, skip_special_tokens=True).strip()
        print("Bot:", answer or "(empty response; train longer and add more data)")

if __name__ == "__main__":
    main()
