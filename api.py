from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from tokenizers import Tokenizer
from model import GPT

app = FastAPI(title="From-Scratch T2T Model API")
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_file("checkpoints/tokenizer.json")
ckpt = torch.load("checkpoints/model.pt", map_location=device, weights_only=True)
model = GPT(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
end_id = tok.token_to_id("<|end|>")

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=100, ge=1, le=300)
    temperature: float = Field(default=0.8, gt=0, le=2.0)

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/generate")
def generate(req: GenerateRequest):
    text = f"<|user|>\n{req.prompt}\n<|assistant|>\n"
    ids = tok.encode(text).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=req.max_new_tokens,
                             temperature=req.temperature, top_k=40, stop_id=end_id)
    answer = tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True).strip()
    return {"response": answer}
