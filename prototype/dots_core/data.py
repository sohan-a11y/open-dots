"""Byte encoding + a synthetic 'world' of facts with fictional entities (no leakage from pretraining)."""
import random

import torch

CITIES = ["plimtor", "vask", "orlune", "kethra", "dunmere", "solvig", "brannoc", "tirel",
          "quorra", "anvel", "hesk", "mirow", "zentar", "callis", "uderon", "fenwick",
          "galt", "irsa", "joven", "lukar", "nembra", "ostrin", "pelloc", "rhune",
          "sarn", "tovel", "umbra", "varn", "wexley", "yorin", "zell", "ardent"]
OBJECTS = ["lamp", "cup", "pen", "drum", "kite", "rope", "bell", "coin", "map", "mask",
           "harp", "flute", "shell", "stone", "ring", "boat", "clock", "glove", "key", "sled",
           "vase", "wheel", "fork", "brush", "comb", "net", "jar", "fan", "hook", "plank",
           "torch", "quill"]
SYL = ["ka", "zo", "ri", "vel", "mo", "tan", "qu", "el", "dra", "ny", "sor", "pi", "lu", "gex",
       "ba", "fen", "oth", "wy", "ix", "har", "ce", "dum", "ro", "ty"]


def encode_with_mask(lines, max_len: int = 64):
    rows = [list(s.encode("utf-8"))[:max_len] for s in lines]
    t = max(len(r) for r in rows)
    x = torch.zeros(len(rows), t, dtype=torch.long)
    mask = torch.zeros(len(rows), t)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r)
        mask[i, :len(r)] = 1.0
    return x, mask


def encode(lines, max_len: int = 64):
    return encode_with_mask(lines, max_len)[0]


def _names(rng, n, taken):
    out = []
    while len(out) < n:
        name = "".join(rng.choice(SYL) for _ in range(rng.choice([2, 3])))
        if name not in taken and len(name) >= 4:
            taken.add(name)
            out.append(name)
    return out


def make_facts(entities, rng):
    facts, probes = [], []
    for e in entities:
        c, o = rng.choice(CITIES), rng.choice(OBJECTS)
        facts += [f"{e} lives in {c}.\n", f"{e} likes the {o}.\n"]
        probes += [(f"{e} lives in ", f"{c}."), (f"{e} likes the ", f"{o}.")]
    return facts, probes


def make_world(seed=0, n_base=400, n_a=100, n_b=100, n_d=50):
    rng = random.Random(seed)
    taken: set = set()
    world = {}
    for key, n in (("C", n_base), ("A", n_a), ("B", n_b), ("D", n_d)):
        facts, probes = make_facts(_names(rng, n, taken), rng)
        world[key] = {"facts": facts, "probes": probes}
    noise_rng = random.Random(seed + 99)
    world["noise"] = ["".join(chr(noise_rng.randint(33, 126)) for _ in range(24)) + "\n" for _ in range(100)]
    return world
