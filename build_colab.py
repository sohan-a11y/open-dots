"""Build a self-contained Colab notebook (code + data embedded) for the Phase 1 run."""
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).parent
FILES = ["dots/__init__.py", "dots/memory.py", "dots/retrofit.py", "dots/qa.py", "dots/ledger.py", "dots/writer.py",
         "dots/evaluate.py", "dots/common.py", "run_dots.py", "run_baselines.py", "make_chunks.py",
         "conftest.py", "tests/test_retrofit.py"]
QUIET = '2>&1 | grep --line-buffered -v -i -E "warn|loading|it/s|^\\s*$"'


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


data_b64 = base64.b64encode(gzip.compress((ROOT / "data/phase1_data.json").read_bytes(), 9)).decode()

cells = [md(
    "# OPEN DOTS · Phase 1 on a GPU\n"
    "Qwen2.5-0.5B + 262k-slot HDC sparse memory + Dots ledger: learn 1,000 facts as 10 Dots, measure forgetting, "
    "exact removal (bit-for-bit) and reset, then LoRA and full fine-tuning baselines.\n\n"
    "**v2:** context-robust DOTS + fairly tuned LoRA / full fine-tuning baselines. **Before Run all:** Runtime → Change runtime type → **T4 GPU**. Keep this tab open. About 1.5 h in total. "
    "The last cell downloads `phase1_v2_results.zip`."),
    code("import os, subprocess\n"
         "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # deterministic cuBLAS for exact removal\n"
         "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
         "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)\n"
         "!pip -q install \"transformers==5.16.1\" \"peft==0.21.0\" \"datasets>=3.0\" pytest\n"
         "!pip -q uninstall -y torchao   # Colab's old torchao breaks peft LoRA\n"
         "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())\n"
         "assert torch.cuda.is_available(), 'Switch the runtime to a GPU (Runtime > Change runtime type > T4 GPU)'\n"
         "os.makedirs('/content/phase1/dots', exist_ok=True); os.makedirs('/content/phase1/tests', exist_ok=True); os.makedirs('/content/phase1/data', exist_ok=True)\n"
         "%cd /content/phase1")]
for f in FILES:
    cells.append(code(f"%%writefile {f}\n" + (ROOT / f).read_text(encoding="utf-8")))
cells += [
    code("# Embedded data: 1,000 facts to learn, 300 known facts, 300 NQ-open questions (selected with the base model)\n"
         "import base64, gzip\n"
         f"DATA_B64 = '{data_b64}'\n"
         "open('data/phase1_data.json', 'wb').write(gzip.decompress(base64.b64decode(DATA_B64)))\n"
         "!python make_chunks.py " + QUIET + "\n"
         "import torch; c = torch.load('data/wikitext_chunks.pt')\n"
         "print('chunk checksums match local run:', int(c['background'].sum()) == 438013084 and int(c['heldout'].sum()) == 112696201)"),
    code("# 1) Unit tests (CPU, ~1 min)\n!python -m pytest -q tests 2>&1 | tail -3"),
    code("# 2) Determinism check on this GPU: learn the same Dot twice, deltas must be bit-identical\n"
         "import sys; sys.path.insert(0, '/content/phase1')\n"
         "from dots.common import attach_calibrated, load_base, load_data\n"
         "from dots.ledger import Ledger\n"
         "from dots.writer import DotWriter, WriterConfig\n"
         "m, tok = load_base(); data = load_data()\n"
         "attach_calibrated(m, data, layer=15, tok=tok)\n"
         "led = Ledger(m.dots_memory)\n"
         "w = DotWriter(m, tok, 15, WriterConfig(write_slots=3000, steps=5, lr=0.03, optimizer='adam', protect_quantile=0.9))\n"
         "d1 = w.learn(data['teach'][:16], 'x', led); led.reset(); d2 = w.learn(data['teach'][:16], 'x', led); led.reset()\n"
         "print('GPU write is bit-reproducible:', torch.equal(d1.write_slots, d2.write_slots) and torch.equal(d1.delta, d2.delta))\n"
         "del m, w, led, d1, d2; import gc; gc.collect(); torch.cuda.empty_cache()"),
    code("# 3) Smoke test (tiny sizes, ~2 min)\n"
         "!python run_dots.py --lr 0.03 --protect 0.9 --slots 3000 --augment --smoke " + QUIET + " | tail -3"),
    code("# 4) DOTS v2: each fact also trained under 2 demonstration contexts; tested under an UNSEEN 3rd (~20 min)\n"
         "!python run_dots.py --lr 0.03 --steps 240 --slots 6000 --opt adam --protect 0.9 --augment --tag aug " + QUIET),
    code("# 5) Fair baselines: gentler learning rates, plain data (~25 min)\n"
         "for m, lr in (('lora', '1e-4'), ('lora', '3e-5'), ('fullft', '1e-5'), ('fullft', '5e-6')):\n"
         "    print('====', m, lr)\n"
         "    !python run_baselines.py --method {m} --lr {lr} " + QUIET + " | grep -E 'after all|right after'"),
    code("# 6) Baselines with the same context augmentation as DOTS v2 (~20 min)\n"
         "for m, lr in (('lora', '1e-4'), ('fullft', '1e-5')):\n"
         "    print('====', m, lr, 'aug')\n"
         "    !python run_baselines.py --method {m} --lr {lr} --steps 120 --augment " + QUIET + " | grep -E 'after all|right after'"),
    code("# 7) Summary + download\n"
         "import json, os, glob, shutil, pandas as pd\n"
         "rows = []\n"
         "for f in sorted(glob.glob('runs/*.json')):\n"
         "    if 'smoke' in f: continue\n"
         "    r = json.load(open(f)); a = r.get('after_all')\n"
         "    if not a: continue\n"
         "    rows.append({'run': os.path.basename(f)[:-5], 'taught EM': a['teach_em'], 'taught EM unseen context': a['teach_fewshot_dots1_2']['em'],\n"
         "                 'known kept': a['known']['em'], 'NQ F1': round(a['nq']['f1'], 4), 'ppl': a['ppl'],\n"
         "                 'exact removal': r.get('exact_equals_never_learned', '')})\n"
         "display(pd.DataFrame(rows))\n"
         "os.makedirs('results', exist_ok=True)\n"
         "for f in glob.glob('runs/*.json'): shutil.copy(f, 'results/')\n"
         "shutil.make_archive('phase1_v2_results', 'zip', 'results')\n"
         "from google.colab import files; files.download('phase1_v2_results.zip')"),
]
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = ROOT / "notebooks" / "OPEN_DOTS_Phase1_Colab_v2.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, round(out.stat().st_size / 1024), "KiB")
