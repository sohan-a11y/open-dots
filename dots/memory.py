"""HDC-addressed sparse memory for retrofitting a pretrained transformer.

Training-free addressing: the input is whitened with background statistics, projected by a
FIXED Gaussian matrix (SimHash-style, angle preserving), split in two halves and matched
against FIXED random bipolar sub-keys (product keys). Nothing in the address path is ever
trained, so slot addresses never drift and every write stays findable (and removable).

Values start at zero, so attaching the memory leaves the model's function unchanged.
Values are buffers: the only trainable tensor is a per-write-session copy of the rows
being written (see begin_write / end_write).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HDCMemory(nn.Module):
    def __init__(self, d_model: int, n_sub: int = 512, key_dim: int = 128, topk: int = 32,
                 temp: float = 1.0, seed: int = 1234):
        super().__init__()
        self.d_model, self.n_sub, self.key_dim, self.topk, self.temp = d_model, n_sub, key_dim, topk, temp
        self.n_slots = n_sub * n_sub
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(d_model, 2 * key_dim, generator=g) / math.sqrt(d_model))
        self.register_buffer("sub1", torch.randint(0, 2, (n_sub, key_dim), generator=g).float() * 2 - 1)
        self.register_buffer("sub2", torch.randint(0, 2, (n_sub, key_dim), generator=g).float() * 2 - 1)
        self.register_buffer("mu", torch.zeros(d_model))
        self.register_buffer("sigma", torch.ones(d_model))
        self.register_buffer("values", torch.zeros(self.n_slots, d_model))
        self.register_buffer("bg_df", torch.zeros(self.n_slots))
        self.register_buffer("bg_mass", torch.zeros(self.n_slots))
        self.register_buffer("bg_docs_t", torch.zeros((), dtype=torch.long))
        self._write = None          # (slot -> row position map, trainable rows)
        self._record = None         # list collecting slot indices when recording reads

    @property
    def bg_docs(self) -> int:
        return int(self.bg_docs_t)

    def address(self, x: torch.Tensor):
        z = (x.float() - self.mu) / self.sigma
        q1, q2 = (z @ self.proj).split(self.key_dim, dim=-1)
        scale = 1.0 / math.sqrt(self.key_dim)
        s1, i1 = (q1 @ self.sub1.T * scale).topk(self.topk, dim=-1)
        s2, i2 = (q2 @ self.sub2.T * scale).topk(self.topk, dim=-1)
        cand = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)
        cand_idx = (i1.unsqueeze(-1) * self.n_sub + i2.unsqueeze(-2)).flatten(-2)
        best, pos = cand.topk(self.topk, dim=-1)
        return cand_idx.gather(-1, pos), F.softmax(best / self.temp, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx, w = self.address(x)
        if self._record is not None:
            self._record.append((idx.detach(), w.detach()))
        v = F.embedding(idx, self.values)
        if self._write is not None:
            row, rows = self._write
            r = row[idx]
            hit = r >= 0
            if bool(hit.any()):
                v = torch.where(hit.unsqueeze(-1), rows[r.clamp(min=0)], v)
        return (w.unsqueeze(-1) * v).sum(-2).to(x.dtype)

    # ---- write session ------------------------------------------------------------------
    def begin_write(self, slots: torch.Tensor) -> nn.Parameter:
        dev = self.values.device
        row = torch.full((self.n_slots,), -1, dtype=torch.long, device=dev)
        row[slots] = torch.arange(len(slots), device=dev)
        rows = nn.Parameter(self.values[slots].clone())
        self._write = (row, rows)
        return rows

    def end_write(self) -> torch.Tensor:
        _, rows = self._write
        self._write = None
        return rows.detach().clone()
