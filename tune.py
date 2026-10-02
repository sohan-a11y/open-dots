"""Quick write-rate sweep: learn ONE Dot (100 facts), measure its recall and known-fact retention."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dots.common import RUNS, attach_calibrated, dots_of, load_base, load_data  # noqa: E402
from dots.evaluate import evaluate_qa  # noqa: E402
from dots.ledger import Ledger  # noqa: E402
from dots.writer import DotWriter, WriterConfig  # noqa: E402
from dots.qa import FEWSHOT  # noqa: E402

torch.set_num_threads(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", default="2,8,32")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--slots", type=int, default=3000)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--opt", default="sgd")
    ap.add_argument("--protect", type=float, default=None)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, tok = load_base()
    data = load_data()
    t = time.time()
    attach_calibrated(model, data, layer=a.layer, tok=tok)
    print(f"calibrated in {time.time() - t:.0f}s", flush=True)
    dot1 = dots_of(data["teach"], 10)[0]
    known = data["known"][:100]
    ledger = Ledger(model.dots_memory)
    res = []
    for lr in [float(x) for x in a.lrs.split(",")]:
        ledger.reset()
        w = DotWriter(model, tok, a.layer, WriterConfig(write_slots=a.slots, steps=a.steps, lr=lr, optimizer=a.opt, protect_quantile=a.protect, anchor_weight=a.anchor))
        t = time.time()
        d = w.learn(dot1, "tune", ledger)
        secs = time.time() - t
        r1, rk = evaluate_qa(model, tok, dot1), evaluate_qa(model, tok, known, prefix=FEWSHOT)
        row = {"opt": a.opt, "layer": a.layer, "protect": a.protect, "anchor": a.anchor, "lr": lr, "steps": a.steps, "slots": len(d.write_slots), "train_s": round(secs, 1),
               "dot_em": r1["em"], "dot_f1": round(r1["f1"], 3), "known_em": rk["em"],
               "delta_norm": round(float(d.delta.norm(dim=1).mean()), 3)}
        res.append(row)
        print(row, flush=True)
    (RUNS / f"tune_L{a.layer}_{a.opt}{a.out}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
