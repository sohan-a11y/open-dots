"""Phase 1 retrofit tests on a tiny random Qwen2 (fast, CPU)."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dots.memory import HDCMemory
from dots.retrofit import attach_memory, calibrate, layer_input, forward_from_layer
from dots.qa import QAExample, encode_qa, normalize_answer, exact_match, token_f1
from dots.writer import DotWriter, WriterConfig
from dots.ledger import Ledger

LAYER = 2


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).eval()


def toks(n, length, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, 300, (n, length), generator=g)


@pytest.fixture()
def model():
    m = tiny_model()
    attach_memory(m, LAYER, HDCMemory(64, n_sub=16, key_dim=16, topk=4))
    calibrate(m, [toks(8, 12, seed=1)], chunk_docs=True)
    return m


def facts(offset=0, n=4):
    return [QAExample(question=f"q{offset + i} what is {offset + i}?", answers=[f"a{offset + i}"]) for i in range(n)]


class FakeTok:
    """Deterministic char-level tokenizer stand-in (ids 5..254)."""
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, s, add_special_tokens=False):
        return [5 + (ord(c) % 250) for c in s]


# ---- memory --------------------------------------------------------------------------
def test_memory_keys_fixed_bipolar_and_values_zero():
    mem = HDCMemory(32, n_sub=8, key_dim=8, topk=2)
    for name in ("sub1", "sub2"):
        k = getattr(mem, name)
        assert not k.requires_grad and set(k.unique().tolist()) <= {-1.0, 1.0}
    assert mem.n_slots == 64 and torch.count_nonzero(mem.values) == 0
    assert list(mem.parameters()) == []          # nothing trainable outside a write session


def test_memory_addressing_deterministic_topk():
    mem = HDCMemory(32, n_sub=8, key_dim=8, topk=3)
    x = torch.randn(2, 5, 32)
    i1, w1 = mem.address(x)
    i2, w2 = mem.address(x)
    assert torch.equal(i1, i2) and torch.equal(w1, w2) and i1.shape == (2, 5, 3)
    assert torch.allclose(w1.sum(-1), torch.ones(2, 5))


# ---- retrofit ------------------------------------------------------------------------
def test_attach_is_function_preserving():
    base = tiny_model()
    x = toks(3, 10, seed=2)
    with torch.no_grad():
        ref = base(x).logits
    attach_memory(base, LAYER, HDCMemory(64, n_sub=16, key_dim=16, topk=4))
    with torch.no_grad():
        out = base(x).logits
    assert torch.equal(ref, out)


def test_calibration_sets_whitening_and_background_counts(model):
    mem = model.dots_memory
    assert mem.mu.abs().sum() > 0 and torch.all(mem.sigma > 0)
    assert mem.bg_df.sum() > 0 and mem.bg_docs == 8


def test_forward_from_layer_matches_full_forward(model):
    x = toks(3, 10, seed=3)
    with torch.no_grad():
        full = model(x).logits
        h = layer_input(model, x, LAYER)
        part = forward_from_layer(model, h, LAYER)
    assert torch.allclose(full, part, atol=1e-5)


# ---- QA utilities ----------------------------------------------------------------------
def test_encode_qa_masks_only_answer_tokens():
    ids, mask = encode_qa(FakeTok(), [QAExample("q1?", ["ab"])])
    prompt_len = len(FakeTok().encode("Question: q1?\nAnswer:"))
    target = mask[0].nonzero().flatten().tolist()
    # loss positions predict the answer tokens " ab" and the newline
    assert target[0] == prompt_len - 1 and len(target) == len(FakeTok().encode(" ab\n"))


def test_answer_normalization_and_scores():
    assert normalize_answer("The  Eiffel-Tower.") == "eiffel tower"
    assert exact_match("the eiffel tower", ["Eiffel Tower"]) == 1.0
    assert token_f1("eiffel", ["Eiffel Tower"]) == pytest.approx(2 / 3)


# ---- writer + ledger -------------------------------------------------------------------
def test_dot_writes_only_selected_slots_and_nothing_else(model):
    ledger = Ledger(model.dots_memory)
    writer = DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=20, steps=3, lr=0.5, batch_size=2))
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    before = model.dots_memory.values.clone()
    dot = writer.learn(facts(0), source="s0", ledger=ledger)
    changed = (model.dots_memory.values != before).any(-1).nonzero().flatten().tolist()
    assert changed and set(changed) <= set(dot.write_slots.tolist()) and len(dot.write_slots) <= 20
    for n, p in model.named_parameters():
        assert torch.equal(p, params_before[n]), n


def test_reset_is_bitwise(model):
    mem = model.dots_memory
    ledger = Ledger(mem)
    h0 = ledger.state_hash()
    DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=10, steps=2, lr=0.5, batch_size=2)).learn(facts(0), "s", ledger)
    assert ledger.state_hash() != h0
    ledger.reset()
    assert ledger.state_hash() == h0 and ledger.dots == []


def test_ownership_keeps_dots_disjoint(model):
    ledger = Ledger(model.dots_memory)
    w = DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=12, steps=2, lr=0.5, batch_size=2))
    a = w.learn(facts(0), "A", ledger)
    b = w.learn(facts(10), "B", ledger)
    assert not set(a.write_slots.tolist()) & set(b.write_slots.tolist())


def test_exact_removal_equals_never_learned(model):
    ledger = Ledger(model.dots_memory)
    w = DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=12, steps=3, lr=0.5, batch_size=2))
    a = w.learn(facts(0), "A", ledger)
    w.learn(facts(10), "B", ledger)
    w.learn(facts(20), "C", ledger)
    ledger.remove(a.dot_id, writer=w, exact=True)
    after = model.dots_memory.values.clone()
    ledger.reset()
    w.learn(facts(10), "B", ledger)
    w.learn(facts(20), "C", ledger)
    assert torch.equal(after, model.dots_memory.values)


def test_fanout_counts_transitive_dependents(model):
    ledger = Ledger(model.dots_memory)
    w = DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=12, steps=1, lr=0.5, batch_size=2))
    dots = [w.learn(facts(10 * i), f"D{i}", ledger) for i in range(3)]
    fan = ledger.fanout()
    assert set(fan) == {d.dot_id for d in dots} and fan[dots[-1].dot_id] == 0
    assert all(0 <= v <= 2 for v in fan.values())


# ---- evaluation helpers ------------------------------------------------------------------
def test_clean_generation_takes_first_line():
    from dots.evaluate import clean_generation
    assert clean_generation(" Paris.\nQuestion: next") == "Paris."
    assert clean_generation("\n\n") == ""


def test_perplexity_matches_manual_cross_entropy(model):
    from dots.evaluate import perplexity
    x = toks(2, 9, seed=5)
    with torch.no_grad():
        logits = model(x).logits
    ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 300), x[:, 1:].reshape(-1))
    assert perplexity(model, [x]) == pytest.approx(float(torch.exp(ce)), rel=1e-4)


def test_prompt_prefix_is_prepended_for_eval_only():
    from dots.qa import FEWSHOT, build_prompt
    ex = QAExample("q?", ["a"])
    assert build_prompt(ex, FEWSHOT).startswith(FEWSHOT) and build_prompt(ex, FEWSHOT).endswith(ex.prompt())
    assert build_prompt(ex) == ex.prompt()


def test_exact_removal_holds_with_per_dot_adam(model):
    ledger = Ledger(model.dots_memory)
    w = DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=12, steps=3, lr=0.05, batch_size=2, optimizer="adam"))
    a = w.learn(facts(0), "A", ledger)
    w.learn(facts(10), "B", ledger)
    ledger.remove(a.dot_id, writer=w, exact=True)
    after = model.dots_memory.values.clone()
    ledger.reset()
    w.learn(facts(10), "B", ledger)
    assert torch.equal(after, model.dots_memory.values)


def test_calibrate_ignores_padding_positions():
    m1, m2 = tiny_model(), tiny_model()
    for m in (m1, m2):
        attach_memory(m, LAYER, HDCMemory(64, n_sub=16, key_dim=16, topk=4))
    a, b = toks(1, 10, seed=7), toks(1, 6, seed=8)
    calibrate(m1, [a, b])
    padded = torch.cat([b, torch.zeros(1, 4, dtype=torch.long)], 1)
    valid = torch.cat([torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool)], 1)
    calibrate(m2, [a, (padded, valid)])
    assert torch.equal(m1.dots_memory.bg_df, m2.dots_memory.bg_df)
    assert torch.allclose(m1.dots_memory.mu, m2.dots_memory.mu, atol=1e-6)


def test_calibration_records_background_read_mass(model):
    mem = model.dots_memory
    assert mem.bg_mass.sum() > 0
    assert torch.all((mem.bg_mass > 0) == (mem.bg_df > 0))


def test_protect_quantile_never_writes_heavily_used_slots(model):
    mem = model.dots_memory
    ledger = Ledger(mem)
    w = DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=50, steps=1, lr=0.1, batch_size=2,
                                                       optimizer="adam", protect_quantile=0.5))
    d = w.learn(facts(0), "A", ledger)
    cut = torch.quantile(mem.bg_mass, 0.5)
    assert len(d.write_slots) > 0 and torch.all(mem.bg_mass[d.write_slots] <= cut)


def test_anchor_penalty_shrinks_updates_and_keeps_exactness(model):
    mem = model.dots_memory
    ledger = Ledger(mem)
    free = DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=12, steps=3, lr=0.05, batch_size=2, optimizer="adam"))
    tied = DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=12, steps=3, lr=0.05, batch_size=2, optimizer="adam",
                                                          anchor_weight=1000.0))
    n_free = free.learn(facts(0), "A", ledger).delta.norm()
    ledger.reset()
    a = tied.learn(facts(0), "A", ledger)
    assert a.delta.norm() < n_free
    tied.learn(facts(10), "B", ledger)
    ledger.remove(a.dot_id, writer=tied, exact=True)
    after = mem.values.clone()
    ledger.reset()
    tied.learn(facts(10), "B", ledger)
    assert torch.equal(after, mem.values)


def test_writes_run_with_deterministic_algorithms(model):
    seen = []
    h = model.dots_memory.register_forward_pre_hook(lambda m, a: seen.append(torch.are_deterministic_algorithms_enabled()))
    try:
        DotWriter(model, FakeTok(), LAYER, WriterConfig(write_slots=8, steps=2, lr=0.1, batch_size=2)).learn(
            facts(0), "A", Ledger(model.dots_memory))
    finally:
        h.remove()
    assert any(seen) and not torch.are_deterministic_algorithms_enabled()


def test_prefix_augmentation_builds_context_variants():
    from dots.qa import augment
    ex = QAExample("q?", ["a"])
    out = augment([ex], ["P1\n", "P2\n"])
    assert [e.prompt() for e in out] == [ex.prompt(), "P1\n" + ex.prompt(), "P2\n" + ex.prompt()]
    assert all(e.target() == ex.target() for e in out)
