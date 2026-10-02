"""Yardstick baselines on the same 10 fact chunks: full fine-tuning (Adafactor) and LoRA.

Same answer-only loss, batch size and steps per chunk as the Dots run. Embeddings (tied with
the LM head) stay frozen in full FT to fit 8 GB of RAM; every transformer block is trained.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dots.common import RUNS, dots_of, load_base, load_data  # noqa: E402
from dots.qa import augment, encode_qa  # noqa: E402
from run_dots import full_eval, log, qa  # noqa: E402

torch.set_num_threads(6)


def train_chunk(model, tok, examples, opt, steps, bs, seed):
    dev = next(model.parameters()).device
    ids, mask = encode_qa(tok, examples)
    ids, mask = ids.to(dev), mask.to(dev)
    targets = ids.roll(-1, dims=1)
    gen = torch.Generator().manual_seed(seed)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for _ in range(steps):
        bi = torch.randint(0, ids.shape[0], (min(bs, ids.shape[0]),), generator=gen).to(dev)
        pos = mask[bi].bool()
        h = base.model(input_ids=ids[bi]).last_hidden_state
        loss = F.cross_entropy(base.lm_head(h[pos]), targets[bi][pos])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fullft", "lora"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dots", type=int, default=10)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
    a = ap.parse_args()
    model, tok = load_base()
    data = load_data()
    if a.smoke:
        data = {**data, "teach": data["teach"][:12], "known": data["known"][:6], "nq": data["nq"][:6],
                "heldout": data["heldout"][:2]}
        a.dots, a.steps = 3, 2
    if a.method == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    else:
        from transformers.optimization import Adafactor
        model.model.embed_tokens.weight.requires_grad_(False)
        opt = Adafactor([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                        scale_parameter=False, relative_step=False, warmup_init=False)
    model.train()
    groups = dots_of(data["teach"], a.dots)
    res = {"args": vars(a), "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    times = []
    for i, g in enumerate(groups):
        g_train = augment(g) if a.augment else g
        t = time.time()
        train_chunk(model, tok, g_train, opt, a.steps, 16, seed=1000 + i)
        times.append(round(time.time() - t, 1))
        log(f"{a.method} chunk {i + 1}: {times[-1]}s")
        if i == 0:
            model.eval()
            res["dot1_right_after"] = qa(model, tok, g)
            log("dot1 recall right after:", res["dot1_right_after"])
            model.train()
    res["chunk_seconds"] = times
    model.eval()
    res["after_all"] = full_eval(model, tok, data, groups)
    log("after all:", {k: v for k, v in res["after_all"].items() if k != "dots"})
    (RUNS / f"baseline_{a.method}_lr{a.lr:g}{'_aug' if a.augment else ''}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
