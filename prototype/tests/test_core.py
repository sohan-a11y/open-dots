"""Tests for the DOTS-core v0 prototype: addressing, gating, ledger exactness, reset."""
import torch
import pytest

from dots_core.memory import HDCSparseMemory
from dots_core.model import DotsLM, ModelConfig
from dots_core.ledger import Ledger
from dots_core.learner import surprise_gate, PlasticWriter, WriterConfig
from dots_core.curiosity import goldilocks_select
from dots_core.data import encode


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, mem_subkeys=16, mem_dim=64, mem_topk=4)
    return DotsLM(cfg).eval()


def test_memory_keys_are_fixed_bipolar_hypervectors():
    mem = HDCSparseMemory(d_in=16, d_out=8, n_subkeys=8, key_dim=32, topk=2)
    for k in (mem.subkeys1, mem.subkeys2):
        assert not k.requires_grad
        assert set(k.unique().tolist()) <= {-1.0, 1.0}


def test_memory_addressing_is_deterministic_and_sparse():
    torch.manual_seed(1)
    mem = HDCSparseMemory(d_in=16, d_out=8, n_subkeys=8, key_dim=32, topk=3)
    x = torch.randn(2, 5, 16)
    _, idx1, w1 = mem(x, return_addr=True)
    _, idx2, w2 = mem(x, return_addr=True)
    assert torch.equal(idx1, idx2) and torch.equal(w1, w2)
    assert idx1.shape == (2, 5, 3)
    assert torch.allclose(w1.sum(-1), torch.ones(2, 5))
    assert int(idx1.max()) < mem.n_slots


def test_surprise_gate_suppresses_predictable_tokens():
    nll = torch.tensor([0.01, 0.1, 3.0, 6.0])
    g = surprise_gate(nll, tau=1.0, temp=0.25)
    assert g[0] < 0.05 and g[1] < 0.05 and g[2] > 0.95
    assert torch.all((g >= 0) & (g <= 1))


def test_writer_only_touches_selected_slots(tiny_model):
    model = tiny_model
    ledger = Ledger(model)
    writer = PlasticWriter(model, WriterConfig(max_write_slots=20, steps=3, lr=0.05, batch_size=4))
    before = model.memory.values.detach().clone()
    commit = writer.learn(["zorvak lives in plimtor.\n", "quell likes the lamp.\n"], source="unit", ledger=ledger)
    changed = (model.memory.values.detach() != before).any(-1).nonzero().flatten()
    assert set(changed.tolist()) <= set(commit.write_slots.tolist())
    assert len(commit.write_slots) <= 20
    # everything outside memory values is untouched
    for name, p in model.named_parameters():
        if name != "memory.values":
            assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_reset_restores_base_bitwise(tiny_model):
    model = tiny_model
    ledger = Ledger(model)
    base_hash = ledger.state_hash()
    writer = PlasticWriter(model, WriterConfig(max_write_slots=10, steps=2, lr=0.05, batch_size=2))
    writer.learn(["abc def ghi.\n"], source="s1", ledger=ledger)
    assert ledger.state_hash() != base_hash
    ledger.reset()
    assert ledger.state_hash() == base_hash
    assert ledger.commits == []


def test_exact_removal_equals_never_learned(tiny_model):
    """Learn A then B, remove A exactly == model that only ever learned B (bitwise)."""
    model = tiny_model
    ledger = Ledger(model)
    wc = WriterConfig(max_write_slots=12, steps=3, lr=0.05, batch_size=2)
    data_a = ["alpha lives in north.\n", "beta likes the cup.\n"]
    data_b = ["gamma lives in south.\n", "delta likes the pen.\n"]

    writer = PlasticWriter(model, wc)
    ca = writer.learn(data_a, source="A", ledger=ledger)
    writer.learn(data_b, source="B", ledger=ledger)
    report = ledger.remove(ca.commit_id, writer=writer, exact=True)
    after_removal = model.memory.values.detach().clone()

    ledger.reset()
    writer.learn(data_b, source="B", ledger=ledger)
    counterfactual = model.memory.values.detach().clone()
    assert torch.equal(after_removal, counterfactual), report


def test_fast_removal_restores_untouched_slots_exactly(tiny_model):
    model = tiny_model
    ledger = Ledger(model)
    base = model.memory.values.detach().clone()
    writer = PlasticWriter(model, WriterConfig(max_write_slots=8, steps=2, lr=0.05, batch_size=2))
    c = writer.learn(["omega lives in east.\n"], source="X", ledger=ledger)
    ledger.remove(c.commit_id, writer=writer, exact=False)
    assert torch.equal(model.memory.values.detach(), base)


def test_goldilocks_prefers_learnable_novelty():
    scores = {"known": 0.2, "novel1": 2.0, "novel2": 1.8, "noise": 5.4}
    picked = goldilocks_select(scores, low=0.6, high=4.5, budget=2)
    assert picked == ["novel1", "novel2"]


def test_encode_is_bytes():
    t = encode(["hi\n"])
    assert t.dtype == torch.long and t.tolist()[0][:3] == [104, 105, 10]


def test_exact_removal_with_slot_ownership(tiny_model):
    model = tiny_model
    ledger = Ledger(model)
    wc = WriterConfig(max_write_slots=12, steps=3, lr=0.05, batch_size=2, own_slots=True)
    writer = PlasticWriter(model, wc)
    line_a, line_b = "kilo lives in west." + chr(10), "lima likes the cup." + chr(10)
    ca = writer.learn([line_a], source="A", ledger=ledger)
    cb = writer.learn([line_b], source="B", ledger=ledger)
    assert not set(ca.write_slots.tolist()) & set(cb.write_slots.tolist())
    ledger.remove(ca.commit_id, writer=writer, exact=True)
    after = model.memory.values.detach().clone()
    ledger.reset()
    writer.learn([line_b], source="B", ledger=ledger)
    assert torch.equal(after, model.memory.values.detach())
