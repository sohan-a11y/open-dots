# DOTS-0 — mechanism prototype

A 2.6M-parameter byte-level LM that tests the OPEN DOTS learning loop on a CPU:

| Module | File | Concept |
|---|---|---|
| Gated delta-rule fast weights | `dots_core/model.py` (`GatedDeltaRule`) | test-time training / fast-weight programmers; error-corrective Hebbian write with decay gate |
| HDC-addressed sparse memory | `dots_core/memory.py` | product-key slots with **fixed** random bipolar sub-keys; MAP-bound queries; only `values` learn |
| Plastic writer | `dots_core/learner.py` | TF-IDF slot selection, surprise gate, slot protection, slot ownership, replay anchor; backward depth = memory → head |
| Ledger | `dots_core/ledger.py` | every write is a commit (read set, write set, delta, seed, data); reset; fast removal; exact removal by taint-closure replay |
| Curiosity | `dots_core/curiosity.py` | Goldilocks surprise band (skip known + noise) |

## Run

```bash
python -m pytest -q tests
```

```bash
python experiment.py 128
```

`128` = sub-keys per product half (128² = 16,384 slots); `256` gives 65,536 slots. The first run pretrains (~17 min on a 6-thread CPU) and caches `runs/pretrained_<n>.pt`. Results land in `runs/results_<n>.json`.

## What it proves / doesn't

Proves: bit-exact reset; bit-exact targeted removal (equals never having learned the removed set); sparse slot writes forget far less than full fine-tuning; slot ownership stops new-vs-new overwriting. Doesn't prove: language-model quality (synthetic facts, tiny model).
