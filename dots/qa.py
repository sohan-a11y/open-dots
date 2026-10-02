"""QA formatting, answer-only loss masks, and SQuAD-style scoring."""
import re
import string
from collections import Counter
from dataclasses import dataclass

import torch
from dataclasses import replace

PROMPT = "Question: {q}\nAnswer:"
TARGET = " {a}\n"
FEWSHOT = ("Question: What is the capital of Japan?\nAnswer: Tokyo\n"
           "Question: Which planet is known as the Red Planet?\nAnswer: Mars\n"
           "Question: Who painted the Mona Lisa?\nAnswer: Leonardo da Vinci\n")


# Training-only contexts (different demonstrations from the evaluation FEWSHOT prefix)
TRAIN_PREFIXES = (
    "Question: What is the chemical symbol for gold?\nAnswer: Au\n"
    "Question: Which ocean is the largest?\nAnswer: Pacific Ocean\n",
    "Question: Who invented the telephone?\nAnswer: Alexander Graham Bell\n"
    "Question: What is the tallest mountain on Earth?\nAnswer: Mount Everest\n"
    "Question: How many legs does a spider have?\nAnswer: 8\n",
)


@dataclass(frozen=True)
class QAExample:
    question: str
    answers: list
    prefix: str = ""

    def prompt(self) -> str:
        return self.prefix + PROMPT.format(q=self.question)

    def target(self) -> str:
        return TARGET.format(a=self.answers[0])


def augment(examples, prefixes=TRAIN_PREFIXES):
    """Each fact in its plain form plus one copy per demonstration context."""
    out = []
    for ex in examples:
        out += [ex] + [replace(ex, prefix=p) for p in prefixes]
    return out


def build_prompt(ex, prefix: str = "") -> str:
    """Evaluation prompt; training never uses a prefix."""
    return prefix + ex.prompt()


def encode_qa(tok, examples, max_len: int = 96):
    """Right-padded ids and a mask marking positions whose NEXT token is an answer token."""
    rows, spans = [], []
    for ex in examples:
        p = tok.encode(ex.prompt(), add_special_tokens=False)
        t = tok.encode(ex.target(), add_special_tokens=False)
        ids = (p + t)[:max_len]
        rows.append(ids)
        spans.append((len(p) - 1, min(len(ids), len(p) + len(t)) - 1))
    T = max(len(r) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    x = torch.full((len(rows), T), pad, dtype=torch.long)
    mask = torch.zeros(len(rows), T)
    for i, (r, (a, b)) in enumerate(zip(rows, spans)):
        x[i, :len(r)] = torch.tensor(r)
        mask[i, a:b] = 1.0
    return x, mask


def valid_positions(mask: torch.Tensor) -> torch.Tensor:
    """All positions up to and including the last answer token (excludes right padding)."""
    ar = torch.arange(mask.shape[1], device=mask.device)
    last = (mask * ar).max(1).values + 1
    return ar.unsqueeze(0) <= last.unsqueeze(1)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(" " if ch in string.punctuation else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred: str, golds) -> float:
    def f1(a, b):
        pa, pb = normalize_answer(a).split(), normalize_answer(b).split()
        common = sum((Counter(pa) & Counter(pb)).values())
        if not pa or not pb or common == 0:
            return 0.0
        prec, rec = common / len(pa), common / len(pb)
        return 2 * prec * rec / (prec + rec)
    return max(f1(pred, g) for g in golds)
