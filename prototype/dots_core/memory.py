"""HDC-addressed sparse memory (the plastic knowledge store).

Product-key memory whose sub-keys are FIXED random bipolar hypervectors (never trained),
so slot addresses never drift while the model keeps learning. The query is formed with a
multiplicative (MAP-style) binding of two projections, so "entity (x) relation" addresses
a different region than either factor alone. Only `values` is plastic.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HDCSparseMemory(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_subkeys: int = 128, key_dim: int = 128,
                 topk: int = 8, seed: int = 1234):
        super().__init__()
        self.n_subkeys, self.key_dim, self.topk = n_subkeys, key_dim, topk
        self.n_slots = n_subkeys * n_subkeys
        g = torch.Generator().manual_seed(seed)
        bipolar = lambda: torch.randint(0, 2, (n_subkeys, key_dim), generator=g).float() * 2 - 1
        self.register_buffer("subkeys1", bipolar())
        self.register_buffer("subkeys2", bipolar())
        # binding: q = tanh(A x) * tanh(B x)  (elementwise product = MAP binding)
        self.proj_a = nn.Linear(d_in, 2 * key_dim)
        self.proj_b = nn.Linear(d_in, 2 * key_dim)
        self.values = nn.Parameter(torch.randn(self.n_slots, d_out) * d_out ** -0.5)

    def address(self, x: torch.Tensor):
        """Return (slot_idx [...,k], weights [...,k]) for inputs x [..., d_in]."""
        q = torch.tanh(self.proj_a(x)) * torch.tanh(self.proj_b(x))
        q1, q2 = q.split(self.key_dim, dim=-1)
        scale = 1.0 / math.sqrt(self.key_dim)
        s1, i1 = (q1 @ self.subkeys1.T * scale).topk(self.topk, dim=-1)
        s2, i2 = (q2 @ self.subkeys2.T * scale).topk(self.topk, dim=-1)
        cand = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)          # [..., k*k]
        cand_idx = (i1.unsqueeze(-1) * self.n_subkeys + i2.unsqueeze(-2)).flatten(-2)
        best, pos = cand.topk(self.topk, dim=-1)
        idx = cand_idx.gather(-1, pos)
        return idx, F.softmax(best * 4.0, dim=-1)

    def forward(self, x: torch.Tensor, return_addr: bool = False):
        idx, w = self.address(x)
        out = (F.embedding(idx, self.values) * w.unsqueeze(-1)).sum(-2)
        return (out, idx, w) if return_addr else out
