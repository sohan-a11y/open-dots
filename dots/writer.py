"""DotWriter: learn a batch of QA facts as one Dot by writing only selected memory rows.

1. Encode facts; cache the hidden state entering the memory layer (lower layers never change).
2. Record which slots the facts read (read set) and how often (TF).
3. Choose write slots by TF-IDF against background usage, excluding slots owned by earlier Dots.
4. Plain SGD on those rows only, answer-token loss, deterministic seed from the data.
Backward runs only through the layers above the memory; nothing else in the model changes.
"""
import hashlib
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .ledger import Dot, Ledger
from .qa import encode_qa, valid_positions
from .retrofit import forward_from_layer, layer_input, run_layers


@dataclass(frozen=True)
class WriterConfig:
    write_slots: int = 2000
    steps: int = 60
    lr: float = 1.0
    batch_size: int = 16
    use_ownership: bool = True
    max_len: int = 160
    optimizer: str = "sgd"          # "sgd" or "adam"; a fresh optimizer per Dot keeps removal exact
    protect_quantile: float | None = None   # never write slots whose background read mass is above this quantile
    anchor_weight: float = 0.0      # penalty on row changes, weighted by how much background reads each row


def deterministic_ce(logits, targets):
    """Cross-entropy via log_softmax * one-hot: no atomics, bit-reproducible on GPU too."""
    logp = F.log_softmax(logits, dim=-1)
    return -(logp * F.one_hot(targets, logits.shape[-1]).to(logp.dtype)).sum(-1).mean()


class DotWriter:
    def __init__(self, model, tok, layer: int, cfg: WriterConfig = WriterConfig()):
        self.model, self.tok, self.layer, self.cfg = model, tok, layer, cfg
        self.mem = model.dots_memory

    @staticmethod
    def seed_for(examples) -> int:
        blob = "\x1e".join(e.prefix + e.question + "\x1f" + "\x1f".join(e.answers) for e in examples)
        return int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)

    @torch.no_grad()
    def _reads(self, h, valid):
        self.mem._record = []
        try:
            run_layers(self.model, h, self.layer, self.layer + 1)
            idx = self.mem._record[0][0]
        finally:
            self.mem._record = None
        sel = idx[valid]
        counts = torch.bincount(sel.flatten(), minlength=self.mem.n_slots).float()
        return counts > 0, counts

    def _select(self, counts, owned):
        eligible = counts > 0
        if self.cfg.use_ownership:
            eligible &= ~owned
        if self.cfg.protect_quantile is not None:
            eligible &= self.mem.bg_mass <= torch.quantile(self.mem.bg_mass, self.cfg.protect_quantile)
        idf = torch.log((self.mem.bg_docs + 1.0) / (self.mem.bg_df + 1.0))
        score = torch.where(eligible, counts * idf, torch.full_like(counts, -1.0))
        k = min(self.cfg.write_slots, int(eligible.sum()))
        return score.topk(k).indices.sort().values

    def _prepare(self, examples):
        ids, mask = encode_qa(self.tok, examples, self.cfg.max_len)
        dev = self.mem.values.device
        ids, mask = ids.to(dev), mask.to(dev)
        h = layer_input(self.model, ids, self.layer)
        return ids, mask, h

    def _train(self, ids, mask, h, slots, seed):
        was_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)   # exact removal needs bit-reproducible replay
        rows = self.mem.begin_write(slots)
        flags = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        opt = (torch.optim.Adam([rows], lr=self.cfg.lr) if self.cfg.optimizer == "adam"
               else torch.optim.SGD([rows], lr=self.cfg.lr))
        gen = torch.Generator().manual_seed(seed)
        targets_all = ids.roll(-1, dims=1)
        start = rows.detach().clone()
        used = self.mem.bg_mass[self.mem.bg_mass > 0]
        importance = (self.mem.bg_mass[slots] / (used.mean() if len(used) else 1.0)).unsqueeze(1)
        try:
            for _ in range(self.cfg.steps):
                bi = torch.randint(0, ids.shape[0], (min(self.cfg.batch_size, ids.shape[0]),), generator=gen).to(ids.device)
                pos = mask[bi].bool()
                logits = forward_from_layer(self.model, h[bi], self.layer, positions=pos)
                loss = deterministic_ce(logits, targets_all[bi][pos])
                if self.cfg.anchor_weight > 0:
                    loss = loss + self.cfg.anchor_weight * (importance * (rows - start) ** 2).sum(1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        finally:
            after = self.mem.end_write()
            for n, p in self.model.named_parameters():
                p.requires_grad_(flags[n])
            torch.use_deterministic_algorithms(was_det)
        return after

    def learn(self, examples, source: str, ledger: Ledger) -> Dot:
        ledger.materialize()
        ids, mask, h = self._prepare(examples)
        read_mask, counts = self._reads(h, valid_positions(mask))
        slots = self._select(counts, ledger.owned())
        seed = self.seed_for(examples)
        before = self.mem.values[slots].clone()
        after = self._train(ids, mask, h, slots, seed)
        dot = Dot(ledger.next_id(), source, list(examples), seed, slots, read_mask, after - before)
        ledger.record(dot)
        return dot

    def retrain(self, dot: Dot, prior) -> Dot:
        """Replay `dot` as if `prior` were the whole history (state already materialized)."""
        ids, mask, h = self._prepare(dot.data)
        _, counts = self._reads(h, valid_positions(mask))
        owned = torch.zeros(self.mem.n_slots, dtype=torch.bool, device=self.mem.values.device)
        for d in prior:
            owned[d.write_slots] = True
        slots = self._select(counts, owned)
        before = self.mem.values[slots].clone()
        after = self._train(ids, mask, h, slots, dot.seed)
        return replace(dot, write_slots=slots, delta=after - before)
