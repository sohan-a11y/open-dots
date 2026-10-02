"""Rebuild WikiText-2 token chunks (background 400x128, held-out 100x128) exactly as prepare_data.py did."""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dots.common import DATA, MODEL


def chunks(tok, texts, size, count):
    ids = tok("\n".join(t for t in texts if t.strip()), add_special_tokens=False).input_ids
    n = min(count, len(ids) // size)
    return torch.tensor(ids[: n * size]).view(n, size)


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    DATA.mkdir(exist_ok=True)
    out = {"background": chunks(tok, wt["train"]["text"], 128, 400), "heldout": chunks(tok, wt["test"]["text"], 128, 100)}
    torch.save(out, DATA / "wikitext_chunks.pt")
    print({k: tuple(v.shape) for k, v in out.items()})
