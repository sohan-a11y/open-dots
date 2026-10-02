"""DOTS-core v0 experiment: continual learning, forgetting, reset, fast/exact removal, curiosity.

Usage:  python experiment.py            (pretrains once, caches to runs/pretrained.pt)
Outputs runs/results.json and a printed table. CPU-only, ~minutes on a laptop.
"""
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dots_core.curiosity import doc_surprise, goldilocks_select  # noqa: E402
from dots_core.data import encode_with_mask, make_world  # noqa: E402
from dots_core.learner import PlasticWriter, WriterConfig, token_nll  # noqa: E402
from dots_core.ledger import Ledger  # noqa: E402
from dots_core.model import DotsLM, ModelConfig  # noqa: E402

RUNS = Path(__file__).parent / "runs"
torch.set_num_threads(4)


@torch.no_grad()
def recall(model, probes) -> float:
    """Exact-match greedy recall of the answer (teacher-forced argmax == greedy decode)."""
    hits = 0
    for i in range(0, len(probes), 128):
        chunk = probes[i:i + 128]
        x, mask = encode_with_mask([p + a for p, a in chunk], model.cfg.max_len)
        pred = model(x).argmax(-1)
        for j, (p, a) in enumerate(chunk):
            s, e = len(p.encode()), len((p + a).encode())
            hits += bool(torch.equal(pred[j, s - 1:e - 1], x[j, s:e]))
    return hits / len(probes)


@torch.no_grad()
def mean_nll(model, lines) -> float:
    x, mask = encode_with_mask(lines, model.cfg.max_len)
    return float(token_nll(model(x), x, mask).sum() / mask[:, 1:].sum())


def pretrain(model, lines, steps=3000, bs=64, lr=3e-3, log_every=500):
    x, mask = encode_with_mask(lines, model.cfg.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(steps):
        bi = torch.randint(0, x.shape[0], (bs,), generator=g)
        nll = token_nll(model(x[bi]), x[bi], mask[bi])
        loss = nll.sum() / mask[bi, 1:].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"  pretrain step {step:5d} loss {loss.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def background_counts(model, lines):
    x, mask = encode_with_mask(lines, model.cfg.max_len)
    _, slots, _ = model(x, return_addr=True)
    valid = mask.bool().unsqueeze(-1).expand_as(slots)
    return torch.zeros(model.memory.n_slots).index_add_(0, slots[valid], torch.ones(int(valid.sum())))


def full_finetune(model, lines, steps, bs, lr):
    x, mask = encode_with_mask(lines, model.cfg.max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        bi = torch.randint(0, x.shape[0], (min(bs, x.shape[0]),), generator=g)
        nll = token_nll(model(x[bi]), x[bi], mask[bi])
        loss = nll.sum() / mask[bi, 1:].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def snapshot(model, w):
    return {k: round(recall(model, w[k]["probes"]), 3) for k in ("C", "A", "B")} | \
        {"C_nll": round(mean_nll(model, w["C"]["facts"]), 4)}


def run_continual(base, w, bg, wcfg, label):
    model = copy.deepcopy(base)
    ledger = Ledger(model)
    writer = PlasticWriter(model, wcfg, bg, replay_lines=w["C"]["facts"][::4])
    t0 = time.time()
    ca = writer.learn(w["A"]["facts"], "set-A", ledger)
    ta = time.time() - t0
    after_a = snapshot(model, w)
    cb = writer.learn(w["B"]["facts"], "set-B", ledger)
    after_ab = snapshot(model, w)
    print(f"  [{label}] after A {after_a} | after A,B {after_ab} | A-commit {ta:.1f}s, "
          f"{len(ca.write_slots)} slots, {ca.nbytes() / 1024:.0f} KiB")
    return model, ledger, writer, {"after_A": after_a, "after_AB": after_ab, "commit_A_seconds": round(ta, 2),
                                   "write_slots_A": len(ca.write_slots), "write_slots_B": len(cb.write_slots),
                                   "commit_A_kib": round(ca.nbytes() / 1024, 1)}


def main():
    RUNS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    w = make_world(seed=0)
    subkeys = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    cfg = ModelConfig(mem_subkeys=subkeys)
    ckpt = RUNS / f"pretrained_{subkeys}.pt"
    base = DotsLM(cfg)
    n_params = sum(p.numel() for p in base.parameters())
    n_mem = base.memory.values.numel()
    print(f"model params {n_params:,} (memory values {n_mem:,}; slots {base.memory.n_slots})")
    if ckpt.exists():
        base.load_state_dict(torch.load(ckpt))
    else:
        t0 = time.time()
        pretrain(base, w["C"]["facts"])
        torch.save(base.state_dict(), ckpt)
        print(f"  pretrain done in {time.time() - t0:.0f}s")
    base.eval()
    results = {"config": asdict(cfg), "params": n_params, "memory_params": n_mem,
               "pretrained": snapshot(base, w)}
    print("pretrained:", results["pretrained"])
    bg = background_counts(base, w["C"]["facts"])

    naive = WriterConfig(max_write_slots=1024, steps=300, lr=0.05, batch_size=32)
    wcfg = WriterConfig(max_write_slots=4096, steps=600, lr=0.5, batch_size=32, protect_quantile=0.9,
                        replay_weight=1.0, own_slots=True)
    variants = {
        "ours: DOTS write policy (gate + TF-IDF + protect + replay + ownership)": wcfg,
        "ours: zero-forgetting policy (protect unused-only + ownership)":
            WriterConfig(**{**asdict(wcfg), "protect_quantile": 0.5, "replay_weight": 0.0}),
        "ablation: no surprise gate": WriterConfig(**{**asdict(wcfg), "use_gate": False}),
        "ablation: no ownership": WriterConfig(**{**asdict(wcfg), "own_slots": False}),
        "v0 naive sparse write (TF-IDF only)": naive,
    }
    if subkeys > 128:  # lite run for the larger memory: key variants only
        variants = {k: v for k, v in variants.items() if k.startswith("ours")}
    results["continual"] = {}
    for label, vc in variants.items():
        m, ledger, writer, r = run_continual(base, w, bg, vc, label)
        results["continual"][label] = r
        if label.startswith("ours: DOTS"):
            kept = (m, ledger, writer)

    results["full_finetune"] = {}
    for lr in (1e-3,):
        m = copy.deepcopy(base)
        full_finetune(m, w["A"]["facts"], wcfg.steps, wcfg.batch_size, lr)
        a = snapshot(m, w)
        full_finetune(m, w["B"]["facts"], wcfg.steps, wcfg.batch_size, lr)
        ab = snapshot(m, w)
        results["full_finetune"][f"lr={lr}"] = {"after_A": a, "after_AB": ab}
        print(f"  [full FT lr={lr}] after A {a} | after A,B {ab}")

    # ---- reset & removal on the 'ours' model -------------------------------------
    m, ledger, writer = kept
    rem = {}
    commit_a = ledger.find("set-A")[0]
    saved = copy.deepcopy(ledger.commits)
    rep = ledger.remove(commit_a.commit_id, exact=False)
    rem["fast_remove_A"] = {**rep, **snapshot(m, w)}
    ledger.commits = copy.deepcopy(saved)
    ledger.materialize()
    rep = ledger.remove(commit_a.commit_id, writer=writer, exact=True)
    exact_state = m.memory.values.detach().clone()
    rem["exact_remove_A"] = {**rep, **snapshot(m, w)}
    ledger.reset()
    writer.learn(w["B"]["facts"], "set-B", ledger)
    rem["exact_equals_never_learned_A"] = bool(torch.equal(exact_state, m.memory.values.detach()))
    base_hash = Ledger(copy.deepcopy(base)).state_hash()
    ledger.reset()
    rem["reset_bitwise_equals_pretrained"] = ledger.state_hash() == base_hash
    rem["after_reset"] = snapshot(m, w)
    results["removal"] = rem
    print("removal:", json.dumps(rem, indent=1))

    # ---- curiosity --------------------------------------------------------------------
    docs = {f"new{i}": w["D"]["facts"][2 * i:2 * i + 2] for i in range(50)}
    docs |= {f"known{i}": w["C"]["facts"][2 * i:2 * i + 2] for i in range(50)}
    docs |= {f"noise{i}": w["noise"][2 * i:2 * i + 2] for i in range(50)}
    s = doc_surprise(base, docs)
    known_hi = max(v for k, v in s.items() if k.startswith("known"))
    picked = goldilocks_select(s, low=0.5, high=0.75 * math.log(256), budget=50)
    results["curiosity"] = {
        "mean_surprise": {g: round(sum(v for k, v in s.items() if k.startswith(g)) / 50, 3)
                          for g in ("new", "known", "noise")},
        "max_known": round(known_hi, 3),
        "picked_new_fraction": round(sum(p.startswith("new") for p in picked) / max(1, len(picked)), 3),
        "n_picked": len(picked)}
    print("curiosity:", results["curiosity"])
    (RUNS / f"results_{subkeys}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
