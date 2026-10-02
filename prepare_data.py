"""Build Phase 1 datasets with the base model (no memory attached).

teach : 1,000 TriviaQA train facts the base model gets WRONG (to be learned as 10 Dots)
known : 300 TriviaQA validation facts the base model gets RIGHT (retention probe)
nq    : 300 NQ-open validation questions (forgetting metric used by Lin et al.)
background / heldout : WikiText-2 token chunks (calibration / perplexity)
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from dots.evaluate import generate_answers, score  # noqa: E402
from dots.qa import FEWSHOT, QAExample  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = Path(__file__).parent / "data"
torch.set_num_threads(6)


def short(a) -> bool:
    return 1 <= len(a.split()) <= 3 and len(a) <= 30


def trivia(split, n, seed):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = ds[i]
        v = r["answer"]["value"]
        if short(v) and len(r["question"]) <= 160:
            aliases = list(dict.fromkeys([v] + r["answer"]["aliases"]))
            out.append(QAExample(r["question"].strip(), aliases))
        if len(out) >= n:
            break
    return out


def filtered(model, tok, cands, want_correct, n, label):
    t = time.time()
    preds = generate_answers(model, tok, cands, prefix=FEWSHOT)
    s = score(preds, cands)
    keep = [e for e, ok in zip(cands, s["per_item_em"]) if bool(ok) == want_correct][:n]
    print(f"{label}: base EM {s['em']:.3f} on {len(cands)} candidates; kept {len(keep)} ({time.time() - t:.0f}s)", flush=True)
    return keep, s["em"]


def chunks(tok, texts, size, count):
    ids = tok("\n".join(t for t in texts if t.strip()), add_special_tokens=False).input_ids
    n = min(count, len(ids) // size)
    return torch.tensor(ids[: n * size]).view(n, size)


def main():
    OUT.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    teach, base_teach = filtered(model, tok, trivia("train", 1500, 0), False, 1000, "teach")
    known, base_known = filtered(model, tok, trivia("validation", 1400, 1), True, 300, "known")
    nq_ds = load_dataset("google-research-datasets/nq_open", split="validation")
    rng = random.Random(2)
    nq = [QAExample(nq_ds[i]["question"].strip() + "?", list(nq_ds[i]["answer"])) for i in rng.sample(range(len(nq_ds)), 300)]
    nq_base = score(generate_answers(model, tok, nq, prefix=FEWSHOT), nq)
    print(f"nq: base EM {nq_base['em']:.3f} F1 {nq_base['f1']:.3f}", flush=True)
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    bg = chunks(tok, wt["train"]["text"], 128, 400)
    ho = chunks(tok, wt["test"]["text"], 128, 100)
    torch.save({"background": bg, "heldout": ho}, OUT / "wikitext_chunks.pt")
    dump = lambda xs: [{"question": e.question, "answers": e.answers} for e in xs]
    json.dump({"model": MODEL, "eval_prefix": FEWSHOT, "teach": dump(teach), "known": dump(known), "nq": dump(nq),
               "base": {"teach_em_candidates": base_teach, "known_em_candidates": base_known,
                        "nq_em": nq_base["em"], "nq_f1": nq_base["f1"]}},
              open(OUT / "phase1_data.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", len(teach), "teach,", len(known), "known,", len(nq), "nq;", tuple(bg.shape), tuple(ho.shape), flush=True)


if __name__ == "__main__":
    main()
