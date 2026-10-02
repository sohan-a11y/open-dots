"""Attach an HDCMemory to one decoder layer of a HF Qwen2-style model, calibrate it, and run
partial forwards (layers below the memory never change, so their output can be cached)."""
import torch
import torch.nn as nn

from .memory import HDCMemory


class MLPWithMemory(nn.Module):
    """Parallel branch: mlp(x) + memory(x). Memory values start at zero -> identical function."""

    def __init__(self, mlp: nn.Module, memory: HDCMemory):
        super().__init__()
        self.mlp, self.memory = mlp, memory

    def forward(self, x):
        return self.mlp(x) + self.memory(x)


def attach_memory(model, layer: int, memory: HDCMemory) -> HDCMemory:
    block = model.model.layers[layer]
    memory = memory.to(next(block.parameters()).device)     # follow the model onto GPU
    block.mlp = MLPWithMemory(block.mlp, memory)
    model.__dict__["dots_memory"] = memory        # plain attribute: not registered twice
    model.__dict__["dots_layer"] = layer
    return memory


def _layer_out(out):
    return out[0] if isinstance(out, tuple) else out


def _rope(model, h):
    pid = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    return model.model.rotary_emb(h, pid), pid


def causal_mask(h):
    """Explicit additive causal mask: correct for eager and SDPA attention alike."""
    T = h.shape[1]
    return torch.full((T, T), torch.finfo(h.dtype).min, device=h.device).triu(1)[None, None]


def run_layers(model, h, start: int, stop: int):
    pe, pid = _rope(model, h)
    mask = causal_mask(h)
    for blk in model.model.layers[start:stop]:
        h = _layer_out(blk(h, attention_mask=mask, position_embeddings=pe, position_ids=pid))
    return h


@torch.no_grad()
def layer_input(model, ids: torch.Tensor, layer: int) -> torch.Tensor:
    """Hidden state entering `layer` (causal attention, right padding assumed)."""
    ids = ids.to(model.model.embed_tokens.weight.device)
    return run_layers(model, model.model.embed_tokens(ids), 0, layer)


def forward_from_layer(model, h: torch.Tensor, layer: int, positions: torch.Tensor | None = None):
    """Logits from `layer` upward. `positions` (bool [B,T]) restricts the LM head to those tokens."""
    h = model.model.norm(run_layers(model, h, layer, len(model.model.layers)))
    return model.lm_head(h[positions] if positions is not None else h)


def _split(batch):
    """A batch is ids [B,T] or (ids, valid [B,T] bool) for right-padded data."""
    ids, valid = batch if isinstance(batch, tuple) else (batch, torch.ones_like(batch, dtype=torch.bool))
    return ids, valid.to(ids.device)


@torch.no_grad()
def calibrate(model, batches, chunk_docs: bool = True, stats_batches: int | None = None):
    """Set whitening stats (pass 1) and background slot document-frequency (pass 2).
    Padding positions (valid == False) contribute to neither."""
    mem, layer = model.dots_memory, model.dots_layer
    block = model.model.layers[layer]
    xs, cur = [], {}
    hook = block.mlp.register_forward_pre_hook(lambda m, a: xs.append(a[0][cur["valid"]].float()))
    try:
        for batch in batches[:stats_batches]:
            ids, cur["valid"] = _split(batch)
            cur["valid"] = cur["valid"].to(mem.values.device)
            run_layers(model, layer_input(model, ids, layer), layer, layer + 1)
    finally:
        hook.remove()
    x = torch.cat(xs)
    mem.mu.copy_(x.mean(0))
    mem.sigma.copy_(x.std(0).clamp_min(1e-5))
    mem.bg_df.zero_()
    mem.bg_mass.zero_()
    mem.bg_docs_t.zero_()
    for batch in batches:
        ids, valid = _split(batch)
        valid = valid.to(mem.values.device)
        mem._record = []
        try:
            run_layers(model, layer_input(model, ids, layer), layer, layer + 1)
            idx, w = mem._record[0]
        finally:
            mem._record = None
        for seq, ok in zip(idx, valid):
            mem.bg_df[seq[ok].unique()] += 1
        mem.bg_mass.index_add_(0, idx[valid].flatten(), w[valid].flatten())
        mem.bg_docs_t += idx.shape[0]
