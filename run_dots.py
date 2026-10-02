"""Phase 1 main run: Qwen2.5-0.5B + HDC sparse memory + Dots ledger.

Learns 1,000 TriviaQA facts as 10 Dots, then measures:
  acquisition (EM on taught facts), retention (known facts, NQ-open F1, WikiText-2 perplexity),
  taint fan-out, fast removal, exact removal (bitwise vs a never-learned counterfactual), reset.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dots.common import RUNS, attach_calibrated, dots_of, load_base, load_data  # noqa: E402
from dots.evaluate import evaluate_qa, perplexity  # noqa: E402
from dots.ledger import Ledger  # noqa: E402
from dots.qa import FEWSHOT, augment  # noqa: E402
from dots.writer import DotWriter, WriterConfig  # noqa: E402

torch.set_num_threads(6)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def qa(model, tok, xs, prefix=""):
    r = evaluate_qa(model, tok, xs, prefix=prefix)
    return {"em": round(r["em"], 4), "f1": round(r["f1"], 4), "n": r["n"]}


def full_eval(model, tok, data, groups):
    out = {"dots": [qa(model, tok, g)["em"] for g in groups]}
    out["teach_em"] = round(sum(out["dots"]) / len(out["dots"]), 4)
    out["teach_fewshot_dots1_2"] = qa(model, tok, groups[0] + groups[1], prefix=FEWSHOT)
    out["known"] = qa(model, tok, data["known"], prefix=FEWSHOT)
    out["nq"] = qa(model, tok, data["nq"], prefix=FEWSHOT)
    out["ppl"] = round(perplexity(model, list(data["heldout"].split(10))), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--slots", type=int, default=3000)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--dots", type=int, default=10)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--opt", default="adam")
    ap.add_argument("--protect", type=float, default=None)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--augment", action="store_true", help="train each fact in extra demonstration contexts")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
    a = ap.parse_args()
    res = {"args": vars(a)}
    model, tok = load_base()
    data = load_data()
    if a.smoke:
        data = {**data, "teach": data["teach"][:12], "known": data["known"][:6], "nq": data["nq"][:6],
                "heldout": data["heldout"][:2]}
        a.dots, a.steps = 3, 2
    probe = tok(["Question: Who wrote Hamlet?\nAnswer:", "The capital of France is"], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        ref = model(**probe).logits
    attach_calibrated(model, data, layer=a.layer, tok=tok)
    with torch.no_grad():
        res["function_preserving"] = bool(torch.equal(ref, model(**probe).logits))
    res["base"] = {**data["base"], "ppl": round(perplexity(model, list(data["heldout"].split(10))), 4)}
    log("attached; function preserving:", res["function_preserving"], "| base ppl", res["base"]["ppl"])

    groups = dots_of(data["teach"], a.dots)
    train_groups = [augment(g) for g in groups] if a.augment else groups
    ledger = Ledger(model.dots_memory)
    writer = DotWriter(model, tok, a.layer, WriterConfig(write_slots=a.slots, steps=a.steps, lr=a.lr, optimizer=a.opt, protect_quantile=a.protect, anchor_weight=a.anchor))
    ckpt = RUNS / f"ckpt_{a.tag}.pt"
    if ckpt.exists() and not a.smoke:      # resume after an interruption: skip learning + first eval
        st = torch.load(ckpt, weights_only=False)
        ledger.dots, ledger._next = st["dots"], st["next"]
        ledger.materialize()
        res.update(st["res"])
        log("resumed from checkpoint:", len(ledger.dots), "dots")
    else:
        times = []
        for i, g in enumerate(groups):
            t = time.time()
            d = writer.learn(train_groups[i], f"triviaqa-dot-{i + 1}", ledger)
            times.append(round(time.time() - t, 1))
            log(f"dot {i + 1}: {len(d.write_slots)} slots, {d.nbytes() / 2**20:.1f} MiB, {times[-1]}s")
            if i == 0:
                res["dot1_right_after"] = qa(model, tok, g)
                log("dot1 recall right after learning:", res["dot1_right_after"])
        res["dot_seconds"] = times
        res["after_all"] = full_eval(model, tok, data, groups)
        log("after all dots:", {k: v for k, v in res["after_all"].items() if k != "dots"})
        fan = ledger.fanout()
        res["fanout"] = {"per_dot": fan, "mean_fraction": round(sum(v / max(1, a.dots - i - 1) for i, v in enumerate(fan.values()) if i < a.dots - 1) / max(1, a.dots - 1), 3)}
        res["ledger_mib"] = round(sum(d.nbytes() for d in ledger.dots) / 2**20, 1)
        log("fan-out:", res["fanout"])
        if not a.smoke:
            torch.save({"dots": ledger.dots, "next": ledger._next, "res": res}, ckpt)

    saved = copy.deepcopy(ledger.dots)
    d1 = ledger.dots[0].dot_id
    probe_sets = {"dot1": (groups[0], ""), "dot2": (groups[1], ""), "known": (data["known"], FEWSHOT)}
    rep = ledger.remove(d1, exact=False)
    res["fast_remove_dot1"] = {**rep, **{k: qa(model, tok, v, p)["em"] for k, (v, p) in probe_sets.items()}}
    log("fast removal:", res["fast_remove_dot1"])
    ledger.dots = copy.deepcopy(saved)
    ledger.materialize()
    rep = ledger.remove(d1, writer=writer, exact=True)
    exact_state = model.dots_memory.values.clone()
    res["exact_remove_dot1"] = {**rep, **{k: qa(model, tok, v, p)["em"] for k, (v, p) in probe_sets.items()}}
    log("exact removal:", {k: v for k, v in res["exact_remove_dot1"].items() if k != "replayed"})
    ledger.reset()
    t = time.time()
    for i, g in enumerate(train_groups[1:], start=2):
        writer.learn(g, f"triviaqa-dot-{i}", ledger)
    res["counterfactual_seconds"] = round(time.time() - t, 1)
    res["exact_equals_never_learned"] = bool(torch.equal(exact_state, model.dots_memory.values))
    del exact_state
    ledger.reset()
    with torch.no_grad():
        res["reset_hash_equal"] = ledger.state_hash() == ledger.base_hash
        res["reset_logits_equal_original"] = bool(torch.equal(ref, model(**probe).logits))
    log("exact == never learned:", res["exact_equals_never_learned"], "| reset:", res["reset_hash_equal"], res["reset_logits_equal_original"])
    (RUNS / f"dots_{a.tag}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1))
    log("saved")


if __name__ == "__main__":
    main()
