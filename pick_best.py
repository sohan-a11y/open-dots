"""Pick the write config with the highest single-Dot recall among those keeping >= 97% of known facts."""
import json
from pathlib import Path

rows = []
for f in Path("runs").glob("tune_L15_adam*.json"):
    rows += json.loads(f.read_text())
ok = [r for r in rows if r["known_em"] >= 0.97]
best = max(ok, key=lambda r: (r["dot_em"], r["known_em"]))
args = f"--lr {best['lr']} --steps {best['steps']} --slots {best['slots']} --opt adam"
if best.get("protect") is not None:
    args += f" --protect {best['protect']}"
if best.get("anchor"):
    args += f" --anchor {best['anchor']}"
Path("runs/best_config.json").write_text(json.dumps(best, indent=1))
print(args)
