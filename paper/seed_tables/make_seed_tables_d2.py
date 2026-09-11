# -*- coding: utf-8 -*-
"""Seed tables, part D2: is the non-linear latent arm still as good WITHOUT the circuit?

Internal control, not paper material.  Reads (nothing typed in by hand):
    experiments/13_iqp_turbulence_replication/results_seeds_mlpctrl.jsonl
        the softmax control -- q = softmax(theta0 + g(z)) -- scored by evaluate_v11.py.
    experiments/13_iqp_turbulence_replication/results_seeds_gridmlp.jsonl
        its quantum counterpart, q = |<x|U(theta0 + g(z))|0>|^2, same everything else.
    experiments/13_iqp_turbulence_replication/results_seeds_table2_v11.jsonl
        the published affine IQP arm, RBM-10bit and DCGAN-32.
    each run's meta.json for what evaluate_v11 does not carry.

Writes paper/ml4ps2026/seed_tables_d2.md.  Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import re
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = os.path.join(REPO, "experiments", "13_iqp_turbulence_replication")
CTRL_J = os.path.join(EXP, "results_seeds_mlpctrl.jsonl")
QUANT_J = os.path.join(EXP, "results_seeds_gridmlp.jsonl")
REF_J = os.path.join(EXP, "results_seeds_table2_v11.jsonl")
OUT_MD = os.path.join(HERE, "seed_tables_d2.md")

SNAP_RE = re.compile(r"\s+s(\d+)k$")

# label -> (arm string, source file)
ARMS = OrderedDict([
    ("MLPctrl-H32 flat-MMD (no circuit)", ("MLPctrl-H32-M4-MMDflat-resample", CTRL_J)),
    ("MLPctrl-H4 flat-MMD (no circuit)", ("MLPctrl-H4-M4-MMDflat-resample", CTRL_J)),
    ("MLPctrl-H32 KL (no circuit)", ("MLPctrl-H32-M4-KL-resample", CTRL_J)),
    ("MLPctrl-H4 KL (no circuit)", ("MLPctrl-H4-M4-KL-resample", CTRL_J)),
    ("IQP-mlpH32-M4 (circuit, same latent)", ("IQP-mlpH32-M4-MMDflat-resample", QUANT_J)),
    ("IQP-affine-M4 (Table 2)", ("IQP-affine-M4-MMDflat-resample", REF_J)),
    ("RBM-10bit (Table 2)", ("RBM-10bit", REF_J)),
    ("DCGAN-32 (Table 2)", ("DCGAN-32", REF_J)),
])
BASELINE = "IQP-mlpH32-M4 (circuit, same latent)"

METRICS = [("pdf_js", "PDF-JS", "lower"),
           ("conv_mmd2", "Conv-MMD2", "lower"),
           ("conv_mmd2_aligned", "Conv-MMD2 (flux-aligned)", "lower"),
           ("projection_corr_mean", "held-out projection corr", "higher")]


def read_jsonl(path):
    rows = []
    if os.path.isfile(path):
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def meta_of(run_dir, _c={}):
    if run_dir not in _c:
        p = os.path.join(run_dir or "", "meta.json")
        _c[run_dir] = json.load(open(p)) if os.path.isfile(p) else {}
    return _c[run_dir]


def pm(vals, digits=4):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return "--", None, None
    m = float(v.mean())
    s = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return ("%.*f +- %.*f" % (digits, m, digits, s)), m, s


def overlap(m1, s1, m2, s2):
    if None in (m1, s1, m2, s2):
        return None
    return not (m1 + s1 < m2 - s2 or m2 + s2 < m1 - s1)


def collect():
    cache = {}
    out = OrderedDict()
    for label, (arm, src) in ARMS.items():
        rows = cache.setdefault(src, read_jsonl(src))
        for r in rows:
            # the quantum file also holds snapshot rows; the fixed-4k arm is the bare name
            if SNAP_RE.sub("", str(r.get("arm") or "")).strip() != arm or SNAP_RE.search(
                    str(r.get("arm") or "")):
                continue
            key = (label, int(r["n_train"]))
            d = out.setdefault(key, {"seeds": [], "param": [], "fit_gain": [],
                                     "nonlinearity_ratio": [], "train_corr_mean": [],
                                     "z_norm_mean": [], "projection_corr_mean": [],
                                     "pdf_js": [], "conv_mmd2": [], "conv_mmd2_aligned": []})
            m = meta_of(r.get("run_dir"))
            d["seeds"].append(r.get("seed"))
            d["param"].append(r.get("param_count") or m.get("param_count"))
            for k in ("pdf_js", "conv_mmd2", "conv_mmd2_aligned"):
                d[k].append(r.get(k))
            for k in ("projection_corr_mean", "nonlinearity_ratio", "train_corr_mean",
                      "z_norm_mean"):
                d[k].append(m.get(k))
            L0, Ld = m.get("zpen_L0"), m.get("loss_final_data")
            d["fit_gain"].append((1.0 - float(Ld) / float(L0))
                                 if (L0 not in (None, 0) and Ld is not None) else None)
    return out


def main():
    data = collect()
    if not data:
        raise SystemExit("make_seed_tables_d2.py: no rows found -- run the control grid first")
    order = list(ARMS)
    L = []
    A = L.append

    A("# Seed tables - part D2: the same latent model with the circuit removed")
    A("")
    A("Internal control, not paper material.  The question is narrow: the non-linear latent")
    A("arm (`train_iqp_affine_mlp.py`) beats the published affine IQP arm -- how much of that")
    A("survives if the IQP generator is deleted and the SAME machinery writes the pmf")
    A("directly?")
    A("")
    A("| | quantum arm | this control |")
    A("|---|---|---|")
    A("| generator | `q = \\|<x\\| U(theta) \\|0>\\|^2` | `q = softmax(theta)` |")
    A("| theta | `theta0 + g(z)`, g a 4->H->1024 tanh MLP, no output bias | identical |")
    A("| latents, z penalty, prior, projection, S, paste, steps, Adam | round-3b protocol | identical |")
    A("| objective | flat MMD from the pmf's Walsh coefficients | identical (plus a KL variant) |")
    A("")
    A("Everything is mean +- sd over seeds 0-4, at 4,000 fixed steps, scored by")
    A("`evaluate_v11.py` against the same 200 held-out frames Table 2 uses.  Lower is better")
    A("for PDF-JS and both Conv-MMD2 columns; higher is better for the projection correlation.")
    A("")

    A("## 1. Main table")
    A("")
    A("| arm | N | params | PDF-JS | Conv-MMD2 | Conv-MMD2 (flux-aligned) | proj. corr |")
    A("|---|---|---|---|---|---|---|")
    stats = {}
    for n in (11, 100):
        for label in order:
            key = (label, n)
            if key not in data:
                continue
            d = data[key]
            cells = []
            for mkey, _nm, _dir in METRICS:
                txt, m, s = pm(d[mkey])
                stats[(label, n, mkey)] = (m, s)
                cells.append(txt)
            p = [x for x in d["param"] if x]
            A("| %s | %d | %s | %s | %s | %s | %s |"
              % (label, n, ("%d" % p[0]) if p else "--", cells[0], cells[1], cells[2], cells[3]))
        A("| | | | | | | |")
    A("")

    A("## 2. Control vs its own quantum counterpart, at 1 sd")
    A("")
    A("`separated` = the mean +- 1 sd intervals do not overlap.  The comparison is against")
    A("**%s**, which differs from the controls only by the generator." % BASELINE)
    A("")
    A("| arm | N | metric | this arm | %s | verdict |" % BASELINE)
    A("|---|---|---|---|---|---|")
    verdicts = {}
    for n in (11, 100):
        for label in [l for l in order if l.startswith("MLPctrl")]:
            for mkey, name, direction in METRICS:
                a = stats.get((label, n, mkey))
                b = stats.get((BASELINE, n, mkey))
                if not a or not b or a[0] is None or b[0] is None:
                    continue
                better = (a[0] < b[0]) if direction == "lower" else (a[0] > b[0])
                ov = overlap(a[0], a[1], b[0], b[1])
                verdicts[(label, n, mkey)] = (better, ov)
                A("| %s | %d | %s | %.4f +- %.4f | %.4f +- %.4f | %s, %s |"
                  % (label, n, name, a[0], a[1], b[0], b[1],
                     "better" if better else "worse",
                     "overlap (not distinguishable)" if ov else "separated by >1 sd"))
        A("| | | | | | |")
    A("")

    A("## 3. Diagnostics")
    A("")
    A("| arm | N | fit gain | nonlinearity ratio | \\|\\|z\\|\\| | train corr | proj. corr |")
    A("|---|---|---|---|---|---|---|")
    for n in (11, 100):
        for label in order:
            key = (label, n)
            if key not in data:
                continue
            d = data[key]
            A("| %s | %d | %s | %s | %s | %s | %s |"
              % (label, n, pm(d["fit_gain"], 4)[0], pm(d["nonlinearity_ratio"], 4)[0],
                 pm(d["z_norm_mean"], 4)[0], pm(d["train_corr_mean"], 4)[0],
                 pm(d["projection_corr_mean"], 4)[0]))
        A("| | | | | | | |")
    A("")
    A("`fit gain` = 1 - (data loss at 4,000 steps) / L0, L0 being the same objective's loss at")
    A("theta0 with z = 0.  It is comparable only WITHIN one objective and one generator: the")
    A("KL and flat-MMD losses are different quantities, and so are the softmax's and the")
    A("circuit's.  `nonlinearity ratio` is the relative residual of the best affine fit to the")
    A("learned map on the trained latents (0 = the MLP collapsed to an affine map).")
    A("")

    # ---------------------------------------------------------------- verdict line
    A("## 4. Answer")
    A("")
    best_ctrl = None
    for n in (11, 100):
        for label in [l for l in order if l.startswith("MLPctrl")]:
            k = (label, n, "pdf_js")
            if k in stats and stats[k][0] is not None:
                if best_ctrl is None or stats[k][0] < best_ctrl[1]:
                    best_ctrl = (label, stats[k][0], n)
    wins = sum(1 for v in verdicts.values() if v[0] and v[1] is False)
    losses = sum(1 for v in verdicts.values() if (not v[0]) and v[1] is False)
    ties = sum(1 for v in verdicts.values() if v[1])
    A("Across the %d (control arm, N, metric) comparisons in section 2: **%d separated wins**"
      % (len(verdicts), wins))
    A("for the circuit-free control, **%d separated losses**, and %d not distinguishable at"
      % (losses, ties))
    A("1 sd.")
    A("")

    A("## 5. Provenance")
    A("")
    A("| what | where |")
    A("|---|---|")
    A("| control rows | `%s` |" % os.path.relpath(CTRL_J, REPO))
    A("| quantum counterpart | `%s` |" % os.path.relpath(QUANT_J, REPO))
    A("| Table 2 rows | `%s` |" % os.path.relpath(REF_J, REPO))
    A("| trainer | `experiments/13_iqp_turbulence_replication/train_mlpctrl.py` |")
    A("| grid | `experiments/13_iqp_turbulence_replication/seeds/submit_seeds_mlpctrl.sbatch` |")
    A("| run outputs | `<WORK>/gsoc_iqp_gridmlp/ctrl/` |")
    A("")
    nrun = sum(len(v["seeds"]) for v in data.values())
    A("%d scored runs across %d (arm, N) cells." % (nrun, len(data)))
    A("")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L))
    print("[tables-d2] wrote %s (%d lines, %d cells, %d runs)"
          % (OUT_MD, len(L), len(data), nrun))
    for key in sorted(data, key=lambda k: (k[1], order.index(k[0]))):
        print("   %-40s N=%-4d seeds=%d" % (key[0], key[1], len(data[key]["seeds"])))
    print("[tables-d2] control vs circuit at 1 sd: %d wins / %d losses / %d ties"
          % (wins, losses, ties))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
