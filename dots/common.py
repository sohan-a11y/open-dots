"""Shared loading helpers for Phase 1 scripts."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .memory import HDCMemory
from .qa import FEWSHOT, QAExample
from .retrofit import attach_memory, calibrate

ROOT = Path(__file__).resolve().parent.parent
DATA, RUNS = ROOT / "data", ROOT / "runs"
MODEL = "Qwen/Qwen2.5-0.5B"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_base():
    """fp32; on GPU use eager attention so training is bit-reproducible (exact removal)."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"attn_implementation": "eager"} if DEVICE == "cuda" else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, **kw)
    return model.to(DEVICE).eval(), tok


def load_data():
    d = json.loads((DATA / "phase1_data.json").read_text(encoding="utf-8"))
    ex = lambda xs: [QAExample(r["question"], r["answers"]) for r in xs]
    chunks = torch.load(DATA / "wikitext_chunks.pt")
    return {"teach": ex(d["teach"]), "known": ex(d["known"]), "nq": ex(d["nq"]), "base": d["base"],
            "background": chunks["background"], "heldout": chunks["heldout"]}


def qa_background(tok, n_zero=1000, n_few=500, seed=7, batch=32):
    """Chat-format background: NQ-open TRAIN Q&A (disjoint from every evaluation set), with and
    without the few-shot prefix, so slots used by the Q&A format itself count as common."""
    import random
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/nq_open", split="train")
    idx = random.Random(seed).sample(range(len(ds)), n_zero + n_few)
    texts = []
    for j, i in enumerate(idx):
        ex = QAExample(ds[i]["question"].strip() + "?", list(ds[i]["answer"]))
        texts.append((FEWSHOT if j >= n_zero else "") + ex.prompt() + ex.target())
    tok.padding_side = "right"
    out = []
    for k in range(0, len(texts), batch):
        enc = tok(texts[k:k + batch], return_tensors="pt", padding=True, add_special_tokens=False)
        out.append((enc.input_ids, enc.attention_mask.bool()))
    return out


def attach_calibrated(model, data, layer=15, n_sub=512, key_dim=128, topk=32, temp=1.0, tok=None):
    """tok given -> background = WikiText + chat-format Q&A; otherwise WikiText only."""
    mem = attach_memory(model, layer, HDCMemory(model.config.hidden_size, n_sub, key_dim, topk, temp))
    RUNS.mkdir(exist_ok=True)
    tag = "_qa2" if tok is not None else "_v2"
    path = RUNS / f"calib_L{layer}_s{n_sub}_k{key_dim}_t{topk}{tag}.pt"
    if path.exists():
        st = torch.load(path)
        for k in ("mu", "sigma", "bg_df", "bg_mass", "bg_docs_t"):
            getattr(mem, k).copy_(st[k])
    else:
        batches = list(data["background"].split(16))
        if tok is not None:
            qa_b = qa_background(tok)
            mixed = [b for pair in zip(batches, qa_b) for b in pair]
            batches = mixed + batches[len(qa_b):] + qa_b[len(batches):]
        calibrate(model, batches, stats_batches=12)
        torch.save({k: getattr(mem, k).clone() for k in ("mu", "sigma", "bg_df", "bg_mass", "bg_docs_t")}, path)
    return mem


def dots_of(teach, n_dots):
    size = len(teach) // n_dots
    return [teach[i * size:(i + 1) * size] for i in range(n_dots)]
