"""Surprise-gated sparse plastic writer (the "neuromodulated delta rule" for persistent memory).

Per commit:
  1. address the new text through the frozen trunk -> read set + per-slot access counts
  2. pick write slots by TF-IDF (accessed a lot here, rarely in background) - Lin et al. 2025 style
  3. update ONLY those slot values with a loss where each byte is weighted by a surprise gate
     g_t = sigmoid((NLL_t - tau)/temp): predictable bytes write nothing (neuromodulation)
Backward depth is one layer (memory -> head), so the update is local and cheap.
"""
import hashlib
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .data import encode_with_mask
from .ledger import Commit, Ledger


def surprise_gate(nll: torch.Tensor, tau: float = 1.0, temp: float = 0.25) -> torch.Tensor:
    return torch.sigmoid((nll - tau) / temp)


@dataclass(frozen=True)
class WriterConfig:
    max_write_slots: int = 512
    steps: int = 60
    lr: float = 0.05
    batch_size: int = 32
    tau: float = 1.0
    temp: float = 0.25
    use_gate: bool = True
    use_selection: bool = True
    protect_quantile: float | None = None   # only write slots whose background use is <= this quantile
    replay_weight: float = 0.0              # anchor: also fit a small background replay sample
    own_slots: bool = False                 # a slot written by an earlier dot is never rewritten by a later one


def token_nll(logits, x, mask):
    nll = F.cross_entropy(logits[:, :-1].transpose(1, 2), x[:, 1:], reduction="none")
    return nll * mask[:, 1:]


class PlasticWriter:
    def __init__(self, model, cfg: WriterConfig = WriterConfig(), background_counts=None,
                 replay_lines=None):
        self.model, self.cfg = model, cfg
        self.replay = encode_with_mask(replay_lines, model.cfg.max_len) if replay_lines else None
        n = model.memory.n_slots
        self.bg = background_counts if background_counts is not None else torch.zeros(n)

    @staticmethod
    def seed_for(lines) -> int:
        return int(hashlib.sha256("".join(lines).encode()).hexdigest()[:8], 16)

    @torch.no_grad()
    def access_stats(self, x, mask):
        _, slots, w = self.model(x, return_addr=True)
        valid = mask.bool().unsqueeze(-1).expand_as(slots)
        n = self.model.memory.n_slots
        counts = torch.zeros(n).index_add_(0, slots[valid], torch.ones(int(valid.sum())))
        return counts > 0, counts

    def owned_mask(self, prior_commits) -> torch.Tensor:
        owned = torch.zeros(self.model.memory.n_slots, dtype=torch.bool)
        if self.cfg.own_slots:
            for c in prior_commits:
                owned[c.write_slots] = True
        return owned

    def select_slots(self, counts, owned=None) -> torch.Tensor:
        accessed = counts > 0
        if owned is not None:
            accessed = accessed & ~owned
        if not self.cfg.use_selection:
            return accessed.nonzero().flatten()
        total = float(self.bg.sum()) + 1.0
        idf = torch.log(total / (1.0 + self.bg))
        eligible = accessed
        if self.cfg.protect_quantile is not None:
            eligible = accessed & (self.bg <= torch.quantile(self.bg, self.cfg.protect_quantile))
        score = torch.where(eligible, counts * idf, torch.full_like(counts, -1.0))
        k = min(self.cfg.max_write_slots, int(eligible.sum()))
        return score.topk(k).indices.sort().values

    def _train(self, lines, seed, write_slots):
        m, cfg = self.model, self.cfg
        x, mask = encode_with_mask(lines, m.cfg.max_len)
        values = m.memory.values
        flags = {n: p.requires_grad for n, p in m.named_parameters()}
        for p in m.parameters():
            p.requires_grad_(False)
        values.requires_grad_(True)
        writable = torch.zeros(values.shape[0], 1, dtype=torch.bool)
        writable[write_slots] = True
        opt = torch.optim.Adam([values], lr=cfg.lr)
        gen = torch.Generator().manual_seed(seed)
        try:
            for _ in range(cfg.steps):
                bi = torch.randint(0, x.shape[0], (min(cfg.batch_size, x.shape[0]),), generator=gen)
                nll = token_nll(m(x[bi]), x[bi], mask[bi])
                g = surprise_gate(nll.detach(), cfg.tau, cfg.temp) if cfg.use_gate else torch.ones_like(nll)
                g = g * mask[bi, 1:]
                # normalise by token count, not gate mass: the gate only ever REDUCES plasticity
                loss = (g * nll).sum() / mask[bi, 1:].sum().clamp_min(1.0)
                if self.replay is not None and cfg.replay_weight > 0:
                    rx, rm = self.replay
                    ri = torch.randint(0, rx.shape[0], (min(cfg.batch_size, rx.shape[0]),), generator=gen)
                    loss = loss + cfg.replay_weight * token_nll(m(rx[ri]), rx[ri], rm[ri]).sum() / rm[ri, 1:].sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                values.grad.mul_(writable)
                opt.step()
        finally:
            values.grad = None
            for n, p in m.named_parameters():
                p.requires_grad_(flags[n])

    def learn(self, lines, source: str, ledger: Ledger | None = None) -> Commit:
        if ledger is not None:
            ledger.materialize()
        x, mask = encode_with_mask(lines, self.model.cfg.max_len)
        read_mask, counts = self.access_stats(x, mask)
        write_slots = self.select_slots(counts, self.owned_mask(ledger.commits if ledger else []))
        seed = self.seed_for(lines)
        before = self.model.memory.values.detach()[write_slots].clone()
        self._train(lines, seed, write_slots)
        delta = self.model.memory.values.detach()[write_slots] - before
        commit = Commit(ledger.next_id() if ledger else 0, source, list(lines), seed,
                        write_slots, read_mask, delta)
        if ledger is not None:
            ledger.record(commit)
        return commit

    def retrain(self, commit: Commit, prior_commits) -> Commit:
        """Replay a commit from the current (already materialized) state as if `prior_commits`
        were the only history: same data and seed, slot selection recomputed."""
        x, mask = encode_with_mask(commit.data, self.model.cfg.max_len)
        _, counts = self.access_stats(x, mask)
        write_slots = self.select_slots(counts, self.owned_mask(prior_commits))
        before = self.model.memory.values.detach()[write_slots].clone()
        self._train(commit.data, commit.seed, write_slots)
        delta = self.model.memory.values.detach()[write_slots] - before
        return replace(commit, write_slots=write_slots, delta=delta)
