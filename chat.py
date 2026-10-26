import torch
from tokenizers import Tokenizer
from model import GPT
from chat_format import answer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file("checkpoints/tokenizer.json")
    ckpt = torch.load("checkpoints/model.pt", map_location=device, weights_only=True)
    model = GPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Tiny from-scratch model. Type /exit to quit.")
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if not prompt:
            continue
        text, conf = answer(model, tok, prompt, device)
        print(f"Bot: {text}  [confidence {conf:.2f}]")

if __name__ == "__main__":
    main()
