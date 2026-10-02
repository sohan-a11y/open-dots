"""DotsLM v0: byte-level LM = attention block -> gated delta-rule fast-weight block -> HDC sparse memory.

Three timescales of plasticity:
  fast   : delta-rule fast weights S_t, rewritten every token, reset per sequence (in-context)
  medium : sparse memory `values`, written by surprise-gated local updates (persistent, logged)
  slow   : everything else, trained once in pretraining (frozen in v0; "sleep" consolidation later)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import HDCSparseMemory


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 256
    d_model: int = 128
    n_heads: int = 4
    max_len: int = 64
    mlp_mult: int = 4
    mem_subkeys: int = 128      # slots = mem_subkeys ** 2
    mem_dim: int = 128          # hypervector half-dimension per product key
    mem_topk: int = 8


class MLP(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.up, self.down = nn.Linear(d, d * mult), nn.Linear(d * mult, d)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class CausalAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.qkv, self.out = h, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class GatedDeltaRule(nn.Module):
    """Fast-weight programmer: S_t = a_t S_{t-1}(I - b_t k k^T) + b_t v k^T ;  o_t = S_t q_t.
    Error-corrective Hebbian write (delta rule) + decay gate = no runaway saturation."""

    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.beta = nn.Linear(d, h)
        self.alpha = nn.Linear(d, h)
        nn.init.constant_(self.alpha.bias, 3.0)       # start with slow decay
        self.out = nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dh).unbind(2)
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)
        beta, alpha = torch.sigmoid(self.beta(x)), torch.sigmoid(self.alpha(x))   # [b,t,h]
        S = x.new_zeros(b, self.h, self.dh, self.dh)
        outs = []
        for i in range(t):
            ki, vi, qi = k[:, i], v[:, i], q[:, i]                 # [b,h,dh]
            bi, ai = beta[:, i, :, None, None], alpha[:, i, :, None, None]
            pred = torch.einsum("bhvk,bhk->bhv", S, ki)
            S = ai * (S + bi * torch.einsum("bhv,bhk->bhvk", vi - pred, ki))
            outs.append(torch.einsum("bhvk,bhk->bhv", S, qi))
        return self.out(torch.stack(outs, 1).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, mixer, d, mult):
        super().__init__()
        self.ln1, self.ln2, self.mix, self.mlp = nn.LayerNorm(d), nn.LayerNorm(d), mixer, MLP(d, mult)

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class DotsLM(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab, d)
        self.pos = nn.Embedding(cfg.max_len, d)
        self.block_attn = Block(CausalAttention(d, cfg.n_heads), d, cfg.mlp_mult)
        self.block_fast = Block(GatedDeltaRule(d, cfg.n_heads), d, cfg.mlp_mult)
        self.ln_mem = nn.LayerNorm(d)
        self.memory = HDCSparseMemory(d, d, cfg.mem_subkeys, cfg.mem_dim, cfg.mem_topk)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab)

    def trunk(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        return self.block_fast(self.block_attn(x))

    def forward(self, idx, return_addr: bool = False):
        h = self.trunk(idx)
        m, slots, w = self.memory(self.ln_mem(h), return_addr=True)
        logits = self.head(self.ln_f(h + m))
        return (logits, slots, w) if return_addr else logits
