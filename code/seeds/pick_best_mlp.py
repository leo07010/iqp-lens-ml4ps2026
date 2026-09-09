"""Pick the (M, H) combination the rest of the MLP suite is built on.

Rule (fixed before the numbers were seen): the winner is the combination with the lowest
mean over seeds 0-4 at N = 11, judged on the flux-aligned Conv-MMD2 and on PDF-JS.  If the
two metrics disagree, BOTH combinations go forward -- that disagreement is information, not
something to break with a tie-break rule invented after the fact.

Prints one "M H" pair per line so a shell can read it, and a short table to stderr.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(EXP, "results_seeds_mlp_suite.jsonl")
ARM_RE = re.compile(r"^IQP-mlpH(\d+)-M(\d+)-MMDflat-resample$")


def main():
    rows = []
    for line in open(RES):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    cells = OrderedDict()
    for r in rows:
        if int(r.get("n_train", 0)) != 11:
            continue
        m = ARM_RE.match(str(r.get("arm") or ""))
        if not m:
            continue
        key = (int(m.group(2)), int(m.group(1)))          # (M, H)
        cells.setdefault(key, {"pdf_js": [], "conv_mmd2_aligned": []})
        cells[key]["pdf_js"].append(r["pdf_js"])
        cells[key]["conv_mmd2_aligned"].append(r["conv_mmd2_aligned"])
    if not cells:
        raise SystemExit("pick_best_mlp.py: no N=11 MMDflat rows in %s" % RES)

    stat = {k: {kk: float(np.mean(v)) for kk, v in d.items()} for k, d in cells.items()}
    best_js = min(stat, key=lambda k: stat[k]["pdf_js"])
    best_mm = min(stat, key=lambda k: stat[k]["conv_mmd2_aligned"])
    sys.stderr.write("%-10s %10s %10s %6s\n" % ("(M, H)", "PDF-JS", "MMD2_al", "seeds"))
    for k in sorted(stat, key=lambda k: stat[k]["conv_mmd2_aligned"]):
        sys.stderr.write("%-10s %10.5f %10.5f %6d%s\n"
                         % ("M=%d H=%d" % k, stat[k]["pdf_js"], stat[k]["conv_mmd2_aligned"],
                            len(cells[k]["pdf_js"]),
                            "   <- best MMD2_al" if k == best_mm else
                            ("   <- best PDF-JS" if k == best_js else "")))
    sys.stderr.write("best by PDF-JS : M=%d H=%d\nbest by MMD2_al: M=%d H=%d\n%s\n"
                     % (best_js[0], best_js[1], best_mm[0], best_mm[1],
                        "the two metrics AGREE" if best_js == best_mm
                        else "the two metrics DISAGREE -> both go forward"))
    winners = [best_mm] if best_js == best_mm else [best_mm, best_js]
    for M, H in winners:
        print("%d %d" % (M, H))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
