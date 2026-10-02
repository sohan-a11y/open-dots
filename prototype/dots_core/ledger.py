"""Provenance ledger: every memory write is a commit (source, time, read set, write set, delta).

Canonical state = base + sum of commit deltas in commit order. That makes:
  reset          -> drop all commits (bitwise base)
  fast removal   -> drop one commit's delta (O(write slots)), no retraining
  exact removal  -> drop it AND replay only the commits that read slots it (transitively)
                    influenced; result is bitwise identical to never having learned it.
"""
import hashlib
import time
from dataclasses import dataclass, field

import torch


@dataclass
class Commit:
    commit_id: int
    source: str
    data: list
    seed: int
    write_slots: torch.Tensor           # LongTensor [w]
    read_mask: torch.Tensor             # BoolTensor [n_slots]
    delta: torch.Tensor                 # [w, d]
    created: float = field(default_factory=time.time)

    def nbytes(self) -> int:
        return self.delta.numel() * self.delta.element_size() + self.write_slots.numel() * 8


class Ledger:
    def __init__(self, model):
        self.model = model
        self.base = model.memory.values.detach().clone()
        self.commits: list[Commit] = []
        self._next_id = 0

    # ---- state -------------------------------------------------------------
    def _compose(self, commits) -> torch.Tensor:
        v = self.base.clone()
        for c in commits:
            v[c.write_slots] += c.delta
        return v

    def materialize(self, commits=None) -> None:
        v = self._compose(self.commits if commits is None else commits)
        with torch.no_grad():
            self.model.memory.values.copy_(v)

    def state_hash(self) -> str:
        return hashlib.sha256(self.model.memory.values.detach().cpu().numpy().tobytes()).hexdigest()

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def record(self, commit: Commit) -> None:
        self.commits.append(commit)
        self.materialize()

    def reset(self) -> None:
        self.commits = []
        self.materialize()

    def find(self, source: str):
        return [c for c in self.commits if c.source == source]

    # ---- removal -----------------------------------------------------------
    def affected_by(self, commit_id: int) -> list[int]:
        """Commits after `commit_id` whose reads touch slots written by it or by another affected commit."""
        tainted = None
        affected = []
        for c in self.commits:
            if c.commit_id == commit_id:
                tainted = torch.zeros_like(c.read_mask)
                tainted[c.write_slots] = True
                continue
            if tainted is not None and bool((c.read_mask & tainted).any()):
                affected.append(c.commit_id)
                tainted[c.write_slots] = True
        return affected

    def remove(self, commit_id: int, writer=None, exact: bool = True) -> dict:
        t0 = time.perf_counter()
        affected = set(self.affected_by(commit_id)) if exact else set()
        if affected and writer is None:
            raise ValueError("exact removal with dependent commits needs a writer to replay them")
        kept: list[Commit] = []
        for c in self.commits:
            if c.commit_id == commit_id:
                continue
            if c.commit_id in affected:
                self.materialize(kept)
                c = writer.retrain(c, kept)
            kept.append(c)
        self.commits = kept
        self.materialize()
        return {"removed": commit_id, "mode": "exact" if exact else "fast",
                "replayed": sorted(affected), "seconds": round(time.perf_counter() - t0, 3)}
