"""Greedy QA generation (EM / F1) and perplexity."""
import torch
import torch.nn.functional as F

from .qa import build_prompt, exact_match, token_f1


def clean_generation(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line.strip()
    return ""


@torch.no_grad()
def generate_answers(model, tok, examples, batch_size: int = 32, max_new_tokens: int = 12, prefix: str = ""):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        enc = tok([build_prompt(e, prefix) for e in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        out += [clean_generation(t) for t in texts]
    return out


def score(preds, examples) -> dict:
    em = [exact_match(p, e.answers) for p, e in zip(preds, examples)]
    f1 = [token_f1(p, e.answers) for p, e in zip(preds, examples)]
    n = max(1, len(examples))
    return {"em": sum(em) / n, "f1": sum(f1) / n, "n": len(examples), "per_item_em": em}


def evaluate_qa(model, tok, examples, **kw) -> dict:
    return score(generate_answers(model, tok, examples, **kw), examples)


@torch.no_grad()
def perplexity(model, batches) -> float:
    total, count = 0.0, 0
    for x in batches:
        x = x.to(model.device)
        logits = model(x).logits[:, :-1]
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum"))
        count += x[:, 1:].numel()
    return float(torch.exp(torch.tensor(total / count)))
