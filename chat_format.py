"""Single source of truth for the chat token layout used by training and inference.

Every message is:  <|role|> "\n" content <|end|>
A generation prompt is the user turn followed by:  <|assistant|> "\n"
"""

ROLE_TOKENS = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
}
FALLBACK = "I don't know."


def special_id(tok, token):
    token_id = tok.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Special token {token} missing. Re-run train_tokenizer.py.")
    return token_id


def encode_message(tok, role, content):
    """Return (ids, loss_mask). Only assistant content and its <|end|> are targets."""
    if role not in ROLE_TOKENS:
        raise ValueError(f"Unsupported role {role!r}")
    is_assistant = role == "assistant"
    header = [special_id(tok, ROLE_TOKENS[role])] + tok.encode("\n").ids
    body = tok.encode(content.strip()).ids
    end_id = special_id(tok, "<|end|>")
    ids = header + body + [end_id]
    mask = [False] * len(header) + [is_assistant] * (len(body) + 1)
    return ids, mask


def build_prompt_ids(tok, prompt, system=None):
    ids = []
    if system:
        ids += encode_message(tok, "system", system)[0]
    ids += encode_message(tok, "user", prompt)[0]
    ids += [special_id(tok, ROLE_TOKENS["assistant"])] + tok.encode("\n").ids
    return ids


def format_text(role, content):
    """Plain-text rendering of one message, for tokenizer training."""
    return f"{ROLE_TOKENS[role]}\n{content.strip()}<|end|>"


def answer(model, tok, prompt, device, max_new_tokens=120, temperature=0.0, top_k=40, min_logprob=None):
    """Generate a reply; return FALLBACK when the model's own confidence is too low.

    A tiny model cannot know facts outside its training data, so rather than emit a
    fluent guess we refuse when the mean token log-prob is below min_logprob.
    """
    import os
    import torch
    if min_logprob is None:
        min_logprob = float(os.getenv("MIN_LOGPROB", "-1.5"))
    ids = build_prompt_ids(tok, prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out, mean_logprob = model.generate(
        x, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, stop_id=special_id(tok, "<|end|>"))
    text = tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True).strip()
    if not text or mean_logprob < min_logprob:
        return FALLBACK, mean_logprob
    return text, mean_logprob
