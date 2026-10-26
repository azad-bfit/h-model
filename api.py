from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from tokenizers import Tokenizer
from model import GPT
from chat_format import answer

app = FastAPI(title="From-Scratch T2T Model API")
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_file("checkpoints/tokenizer.json")
ckpt = torch.load("checkpoints/model.pt", map_location=device, weights_only=True)
model = GPT(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=100, ge=1, le=300)
    temperature: float = Field(default=0.0, ge=0, le=2.0)

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/generate")
def generate(req: GenerateRequest):
    answer_text, conf = answer(model, tok, req.prompt, device, req.max_new_tokens, req.temperature)
    return {"response": answer_text, "mean_logprob": conf}
