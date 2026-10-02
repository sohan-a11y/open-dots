"""Dots ledger: every memory write is a Dot (source, time, seed, read set, write set, delta).

Canonical memory = base + sum of Dot deltas in Dot order, recomputed (never subtracted).
Only rows ever written are recomputed, so it stays cheap for a 262k-slot table.
  reset         -> no Dots (bitwise base)
  fast removal  -> drop one Dot's delta, no retraining
  exact removal -> drop it and replay only Dots whose reads touched rows it (transitively)
                   influenced; result is bitwise identical to never having learned it
"""
import hashlib
import time
from dataclasses import dataclass, field

import torch


@dataclass
class Dot:
    dot_id: int
    source: str
    data: list
    seed: int
    write_slots: torch.Tensor
    read_mask: torch.Tensor
    delta: torch.Tensor
    created: float = field(default_factory=time.time)

    def nbytes(self) -> int:
        return self.delta.numel() * self.delta.element_size() + self.write_slots.numel() * 8


class Ledger:
    def __init__(self, memory):
        self.memory = memory
        v = memory.values
        self.base = None if int(torch.count_nonzero(v)) == 0 else v.clone()
        self.base_hash = self.state_hash()
        self.dots: list[Dot] = []
        self._dirty = torch.zeros(memory.n_slots, dtype=torch.bool, device=v.device)
        self._next = 0

    def _base_rows(self, rows):
        if self.base is None:
            return torch.zeros(len(rows), self.memory.d_model, device=self.memory.values.device)
        return self.base[rows].clone()

    def materialize(self, dots=None) -> None:
        dots = self.dots if dots is None else dots
        touched = self._dirty.clone()
        for d in dots:
            touched[d.write_slots] = True
        rows = touched.nonzero().flatten()
        v = self.memory.values
        with torch.no_grad():
            v[rows] = self._base_rows(rows)
            for d in dots:
                v[d.write_slots] += d.delta
        self._dirty = touched

    def state_hash(self) -> str:
        return hashlib.sha256(self.memory.values.detach().cpu().numpy().tobytes()).hexdigest()

    def next_id(self) -> int:
        self._next += 1
        return self._next

    def record(self, dot: Dot) -> None:
        self.dots.append(dot)
        self.materialize()

    def reset(self) -> None:
        self.dots = []
        self.materialize()

    def owned(self, dots=None) -> torch.Tensor:
        m = torch.zeros(self.memory.n_slots, dtype=torch.bool, device=self.memory.values.device)
        for d in (self.dots if dots is None else dots):
            m[d.write_slots] = True
        return m

    def find(self, source: str):
        return [d for d in self.dots if d.source == source]

    def affected_by(self, dot_id: int) -> list[int]:
        tainted, out = None, []
        for d in self.dots:
            if d.dot_id == dot_id:
                tainted = torch.zeros(self.memory.n_slots, dtype=torch.bool, device=d.read_mask.device)
                tainted[d.write_slots] = True
            elif tainted is not None and bool((d.read_mask & tainted).any()):
                out.append(d.dot_id)
                tainted[d.write_slots] = True
        return out

    def fanout(self) -> dict:
        return {d.dot_id: len(self.affected_by(d.dot_id)) for d in self.dots}

    def remove(self, dot_id: int, writer=None, exact: bool = True) -> dict:
        t0 = time.perf_counter()
        affected = set(self.affected_by(dot_id)) if exact else set()
        if affected and writer is None:
            raise ValueError("exact removal with dependent Dots needs a writer to replay them")
        kept: list[Dot] = []
        for d in self.dots:
            if d.dot_id == dot_id:
                continue
            if d.dot_id in affected:
                self.materialize(kept)
                d = writer.retrain(d, kept)
            kept.append(d)
        self.dots = kept
        self.materialize()
        return {"removed": dot_id, "mode": "exact" if exact else "fast", "replayed": sorted(affected),
                "seconds": round(time.perf_counter() - t0, 2)}
