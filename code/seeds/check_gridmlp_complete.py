"""Completeness check on the grid's outputs: every (arm, N) cell must have all 5 seeds."""
import collections
import json
import os
import sys

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V11 = os.path.join(EXP, "results_seeds_gridmlp.jsonl")
ES = os.path.join(EXP, "results_seeds_gridmlp_es.jsonl")

EXPECT_FIXED = ["IQP-n12-affine-M4-MMDflat-resample",
                "IQP-n12-affine-M4-MMDflat-resample-zinit05",
                "IQP-mlpH32-M4-MMDflat-resample",
                "IQP-mlpH32zs-M4-MMDflat-resample",
                "IQP-mlpH32-M2-MMDflat-resample"]
EXPECT_ES = ["IQP-n12-affine-M4-MMDflat-resample", "IQP-mlpH32-M4-MMDflat-resample"]

bad = []
rows = [json.loads(l) for l in open(V11) if l.strip()]
fixed = collections.defaultdict(set)
snaps = collections.defaultdict(set)
for r in rows:
    arm = str(r.get("arm", ""))
    n = r.get("n_train")
    seed = r.get("seed")
    if arm.rstrip().endswith("k") and " s" in arm:
        snaps[(arm.rsplit(" s", 1)[0], n)].add(seed)
    else:
        fixed[(arm, n)].add(seed)

print("[complete] %d v11 rows in %s" % (len(rows), os.path.basename(V11)))
for arm in EXPECT_FIXED:
    for n in (11, 100):
        got = sorted(x for x in fixed.get((arm, n), set()) if x is not None)
        ok = got == [0, 1, 2, 3, 4]
        print("  fixed  %-46s N=%-4d seeds=%s %s" % (arm, n, got, "OK" if ok else "MISSING"))
        if not ok:
            bad.append((arm, n, "fixed"))

sel = collections.defaultdict(set)
steps = collections.defaultdict(list)
for l in open(ES):
    if not l.strip():
        continue
    r = json.loads(l)
    if r.get("kind") != "gridmlp_early_stopped":
        continue
    sel[(str(r.get("arm", "")).rsplit(" s", 1)[0], r.get("n_train"))].add(r.get("seed"))
    steps[(str(r.get("arm", "")).rsplit(" s", 1)[0], r.get("n_train"))].append(r.get("selected_steps"))
for arm in EXPECT_ES:
    for n in (11, 100):
        got = sorted(x for x in sel.get((arm, n), set()) if x is not None)
        ok = got == [0, 1, 2, 3, 4]
        print("  select %-46s N=%-4d seeds=%s steps=%s %s"
              % (arm, n, got, sorted(steps.get((arm, n), [])), "OK" if ok else "MISSING"))
        if not ok:
            bad.append((arm, n, "selected"))

print("[complete] %s" % ("ALL CELLS FULL" if not bad else "INCOMPLETE: %s" % bad))
sys.exit(1 if bad else 0)
