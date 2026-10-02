# OPEN DOTS

**A frozen language model that keeps learning new facts in its own weights, and can forget any of them exactly.**

DOTS attaches a 262,144-slot sparse memory to a frozen pretrained transformer. Every batch of knowledge it learns is a **Dot**: a logged write to a few memory rows. Because each Dot records which rows it read and wrote, any Dot can be removed so that the weights are **bit-for-bit identical** to a model that never learned it.

Paper: [`paper/DOTS_paper.pdf`](paper/DOTS_paper.pdf), *DOTS: Exactly Removable Continual Learning for Frozen Language Models via Hyperdimensional Sparse Memory* (Sohan Merugu, 2026).

## Results (Qwen2.5-0.5B, 1,000 new TriviaQA facts learned as 10 Dots, Colab T4)

| Method | New facts learned | Under an unseen prompt | Known facts kept | NQ F1 | Perplexity |
|---|---|---|---|---|---|
| Base model | 0% | 0% | 100% | 9.9 | 27.43 |
| **DOTS v2** | **96.1%** | **73.5%** | **86.3%** | 9.2 | 27.88 |
| DOTS v1 | 91.5% | 10.5% | 91.3% | 10.3 | 27.75 |
| Full fine-tuning (best) | 82.3% | 51.0% | 51.3% | 6.9 | 31.66 |
| LoRA (best) | 57.9% | 22.0% | 33.0% | 4.8 | 47.72 |

- **Exact removal:** after removing a Dot, the memory is bitwise identical to a model that never learned it. Verified on CPU and on a T4 GPU.
- **Reset:** restores the original model exactly; the weights hash and output logits match.
- **Main open problem:** removing one Dot currently replays every Dot learned after it, about 3 to 12 minutes on a T4 for 9 Dots.

All numbers come from the JSON files in [`results/`](results/).

## How it works

1. **Retrofit, no training:** a memory branch is added beside the MLP of layer 15. Its addresses are fixed random hypervectors (product keys) applied to whitened hidden states, and its values start at zero, so the model's outputs are unchanged.
2. **Write a Dot:** pick rows the new text reads that ordinary text rarely uses (TF-IDF), skip the 10% busiest rows and rows owned by earlier Dots, then train only those rows with Adam under deterministic kernels.
3. **Ledger:** each Dot stores its source, seed, read set, write set and delta. Reset drops all Dots. Exact removal drops one Dot and replays only the Dots whose reads it could have influenced.

## Run it

**Easiest:** open [`notebooks/OPEN_DOTS_Phase1_Colab_v2.ipynb`](notebooks/OPEN_DOTS_Phase1_Colab_v2.ipynb) in Google Colab, select a T4 GPU, and click Run all. It takes about 1.5 h and downloads the results.

**Locally (Python 3.11+, PyTorch, transformers 5.x, peft, datasets):**

```bash
pip install torch "transformers==5.16.1" "peft==0.21.0" datasets pytest
python make_chunks.py
python -m pytest -q tests
python run_dots.py --lr 0.03 --steps 240 --slots 6000 --opt adam --protect 0.9 --augment --tag aug
python run_baselines.py --method fullft --lr 5e-6
```

A CPU run works but is slow: about 5 hours for the v1 main run on a 6-core laptop.

## Repository layout

| Path | Contents |
|---|---|
| `dots/` | memory, retrofit, writer, ledger, Q&A utilities, evaluation |
| `tests/` | unit tests on a tiny Qwen2 model (exactness, reset, ownership, determinism) |
| `run_dots.py`, `run_baselines.py`, `tune.py` | main experiment, LoRA and full fine-tuning baselines, write-policy sweeps |
| `data/phase1_data.json` | the 1,000 facts to learn, 300 known facts and 300 NQ-open questions |
| `results/` | measured result files from two Colab runs |
| `paper/` | LaTeX source, figures and PDF |
| `prototype/` | DOTS-0, the earlier 2.6M-parameter mechanism prototype on synthetic facts |

## Roadmap

- Reduce removal cost, for example with source-sharded memory regions or snapshots.
- Larger models and learning from documents rather than single facts.
- `pip install dots`: a toolkit to retrofit any Hugging Face model.
- OPEN DOTS app: a local assistant with a timeline of everything it learned, where any item can be removed.

## Data and licenses

Code is Apache-2.0. The base model is Qwen2.5-0.5B (Apache-2.0). The data subsets come from TriviaQA, NaturalQuestions-open and WikiText-2 and keep their original licenses.

## Citation

```bibtex
@misc{merugu2026dots,
  title  = {DOTS: Exactly Removable Continual Learning for Frozen Language Models via Hyperdimensional Sparse Memory},
  author = {Merugu, Sohan},
  year   = {2026},
  url    = {https://github.com/sohan-a11y/open-dots}
}
```

AI assistance (Claude, Anthropic) was used for code, experiments and writing. The author reviewed everything and is responsible for it.
