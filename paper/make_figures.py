"""Figures for the Dots paper, built only from the measured result files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
J = lambda p: json.loads((ROOT / p).read_text())

RUNS = {  # label: (file, color, marker, linestyle)
    "DOTS v2 (context-augmented)": ("colab_t4_v2/dots_aug.json", "#2a78d6", "o", "-"),
    "DOTS v1": ("colab_t4_run2/dots_main.json", "#2a78d6", "s", "--"),
    "Full FT (lr 5e-6)": ("colab_t4_v2/baseline_fullft_lr5e-06.json", "#eb6834", "^", "-"),
    "LoRA (lr 1e-4)": ("colab_t4_v2/baseline_lora_lr0.0001.json", "#1baf7a", "D", "-"),
}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "grid.color": "#e1e0d9", "grid.linewidth": 0.6})

# Figure 2: recall of each Dot after all ten were learned (Dot 1 = oldest)
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for label, (f, c, m, ls) in RUNS.items():
    ys = [100 * v for v in J(f)["after_all"]["dots"]]
    ax.plot(range(1, 11), ys, color=c, marker=m, ls=ls, lw=1.6, ms=4, label=label)
ax.set_xlabel("Dot (1 = learned first)")
ax.set_ylabel("Recall of the Dot's facts (%)")
ax.set_xticks(range(1, 11))
ax.set_ylim(0, 102)
ax.grid(axis="y")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
fig.tight_layout()
fig.savefig(OUT / "per_dot_recall.pdf")
fig.savefig(OUT / "per_dot_recall.png", dpi=200)

# Figure 3: acquisition vs retention for every run
pts = [("DOTS v2", "colab_t4_v2/dots_aug.json", "#2a78d6", "o"), ("DOTS v1", "colab_t4_run2/dots_main.json", "#2a78d6", "s")]
for lr in ("5e-06", "1e-05"):
    pts.append((f"Full FT {lr.replace('e-0', 'e-')}", f"colab_t4_v2/baseline_fullft_lr{lr}.json", "#eb6834", "^"))
pts += [("Full FT 2e-5", "colab_t4_run2/baseline_fullft.json", "#eb6834", "^"),
        ("Full FT 1e-5 +aug", "colab_t4_v2/baseline_fullft_lr1e-05_aug.json", "#eb6834", "v")]
for lr, name in (("0.0001", "1e-4"), ("3e-05", "3e-5")):
    pts.append((f"LoRA {name}", f"colab_t4_v2/baseline_lora_lr{lr}.json", "#1baf7a", "D"))
pts += [("LoRA 3e-4", "colab_t4_run2/baseline_lora.json", "#1baf7a", "D"),
        ("LoRA 1e-4 +aug", "colab_t4_v2/baseline_lora_lr0.0001_aug.json", "#1baf7a", "d")]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for label, f, c, m in pts:
    a = J(f)["after_all"]
    x, y = 100 * a["known"]["em"], 100 * a["teach_em"]
    ax.scatter(x, y, color=c, marker=m, s=28, edgecolor="white", linewidth=0.6, zorder=3)
    off = {"Full FT 2e-5": (-4, -9, "right"), "Full FT 1e-5": (0, -10, "center"), "Full FT 1e-5 +aug": (0, 5, "center"),
           "Full FT 5e-6": (4, -9, "left"), "DOTS v2": (-5, 2, "right"), "DOTS v1": (-5, -8, "right"),
           "LoRA 1e-4": (5, -2, "left"), "LoRA 3e-5": (5, -6, "left")}.get(label, (4, 2, "left"))
    ax.annotate(label, (x, y), xytext=off[:2], textcoords="offset points", fontsize=6, color="#52514e", ha=off[2])
ax.set_xlabel("Known facts kept (%)")
ax.set_ylabel("New facts learned (%)")
ax.set_xlim(0, 100)
ax.set_ylim(30, 100)
ax.grid()
fig.tight_layout()
fig.savefig(OUT / "tradeoff.pdf")
fig.savefig(OUT / "tradeoff.png", dpi=200)
print("figures written to", OUT)
