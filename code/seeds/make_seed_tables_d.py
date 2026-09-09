# -*- coding: utf-8 -*-
"""Seed tables, part D: the two candidate fixes for the v11 paper's stated limitations.

Reads (nothing is typed in by hand):
    experiments/13_iqp_turbulence_replication/results_seeds_gridmlp.jsonl
        v11 rows for the new arms -- (A) the 12-qubit whole-frame arm and (B) the non-linear
        latent map -- produced by evaluate_v11.py, the same scorer Table 2 uses.
    experiments/13_iqp_turbulence_replication/results_seeds_gridmlp_es.jsonl
        seeds/select_checkpoint.py's output: which step count the validation block picked.
    experiments/13_iqp_turbulence_replication/results_seeds_table2_v11.jsonl
        the published Table 2 rows this has to be compared against (the current best IQP arm,
        RBM-10bit, DCGAN-32) -- read, never recomputed.
    each run's meta.json for the quantities evaluate_v11 does not carry (projection
    correlation, z norms, the MLP's non-linearity diagnostics).

Writes paper/ml4ps2026/seed_tables_d.md.  Python 3.9 compatible.
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
NEW_J = os.path.join(EXP, "results_seeds_gridmlp.jsonl")
ES_J = os.path.join(EXP, "results_seeds_gridmlp_es.jsonl")
REF_J = os.path.join(EXP, "results_seeds_table2_v11.jsonl")
OUT_MD = os.path.join(HERE, "seed_tables_d.md")

SNAP_RE = re.compile(r"\s+s(\d+)k$")
REF_ARMS = ("IQP-affine-M4-MMDflat-resample", "RBM-10bit", "DCGAN-32")

# label -> (arm string as it appears in meta.json, protocol tag)
NEW_ARMS = OrderedDict([
    ("IQP-n12-affine-M4 (fixed 4k)", ("IQP-n12-affine-M4-MMDflat-resample", "fixed")),
    ("IQP-n12-affine-M4 (val-selected)", ("IQP-n12-affine-M4-MMDflat-resample", "selected")),
    ("IQP-mlpH32-M4 (fixed 4k)", ("IQP-mlpH32-M4-MMDflat-resample", "fixed")),
    ("IQP-mlpH32-M4 (val-selected)", ("IQP-mlpH32-M4-MMDflat-resample", "selected")),
    ("IQP-mlpH32-M4 zscale (fixed 4k)", ("IQP-mlpH32zs-M4-MMDflat-resample", "fixed")),
    ("IQP-mlpH32-M2 (fixed 4k)", ("IQP-mlpH32-M2-MMDflat-resample", "fixed")),
    ("IQP-n12-affine-M4 z-init 0.05 (fixed 4k)",
     ("IQP-n12-affine-M4-MMDflat-resample-zinit05", "fixed")),
])
REF_LABEL = OrderedDict([
    ("IQP-affine-M4 (Table 2, fixed 4k)", "IQP-affine-M4-MMDflat-resample"),
    ("RBM-10bit (Table 2)", "RBM-10bit"),
    ("DCGAN-32 (Table 2)", "DCGAN-32"),
])

METRICS = [("pdf_js", "PDF-JS", "lower"),
           ("conv_mmd2", "Conv-MMD2", "lower"),
           ("conv_mmd2_aligned", "Conv-MMD2 (flux-aligned)", "lower"),
           ("projection_corr_mean", "held-out projection corr", "higher")]


# --------------------------------------------------------------------------------------
def read_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def meta_of(run_dir, _cache={}):
    if run_dir in _cache:
        return _cache[run_dir]
    p = os.path.join(run_dir or "", "meta.json")
    m = json.load(open(p)) if os.path.isfile(p) else {}
    _cache[run_dir] = m
    return m


def base_arm(arm):
    return SNAP_RE.sub("", str(arm or "")).strip()


def is_snapshot(arm):
    return bool(SNAP_RE.search(str(arm or "")))


def pm(vals, digits=4):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return "--", None, None
    m, s = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
    return ("%.*f +- %.*f" % (digits, m, digits, s)), m, s


def overlap(m1, s1, m2, s2):
    """Do the two mean +- 1 sd intervals overlap?"""
    if None in (m1, s1, m2, s2):
        return None
    return not (m1 + s1 < m2 - s2 or m2 + s2 < m1 - s1)


# --------------------------------------------------------------------------------------
def collect():
    """{(label, n_train): {metric: [per-seed values], ...}} for every arm in the tables."""
    new_rows = read_jsonl(NEW_J)
    ref_rows = read_jsonl(REF_J)
    es_rows = read_jsonl(ES_J)
    selected = {}                      # run_dir -> selected step count
    for r in es_rows:
        if r.get("kind") == "gridmlp_early_stopped" and r.get("run_dir"):
            selected[r["run_dir"]] = r.get("selected_steps")

    out = OrderedDict()

    def put(label, n, row, steps):
        key = (label, int(n))
        d = out.setdefault(key, {"seeds": [], "steps": [], "param": [],
                                 "pdf_js": [], "conv_mmd2": [], "conv_mmd2_aligned": [],
                                 "projection_corr_mean": [], "nonlinearity_ratio": [],
                                 "z_norm_mean": [], "train_corr_mean": [],
                                 "pre_abs_mean": [], "fit_gain": [], "run_dirs": []})
        m = meta_of(row.get("run_dir"))
        d["seeds"].append(row.get("seed"))
        d["steps"].append(steps)
        d["param"].append(row.get("param_count") or m.get("param_count"))
        for k in ("pdf_js", "conv_mmd2", "conv_mmd2_aligned"):
            d[k].append(row.get(k))
        d["projection_corr_mean"].append(m.get("projection_corr_mean"))
        d["nonlinearity_ratio"].append(m.get("nonlinearity_ratio"))
        d["z_norm_mean"].append(m.get("z_norm_mean"))
        d["train_corr_mean"].append(m.get("train_corr_mean"))
        g = m.get("g_stats") or {}
        d["pre_abs_mean"].append(g.get("pre_abs_mean"))
        # how much of the fittable loss the latent map actually removed: 1 - data loss / L0,
        # where L0 is that same objective's loss at theta0 with z = 0 (meta's zpen_L0).
        L0, Ld = m.get("zpen_L0"), m.get("loss_final_data")
        d["fit_gain"].append((1.0 - float(Ld) / float(L0))
                             if (L0 not in (None, 0) and Ld is not None) else None)
        d["run_dirs"].append(row.get("run_dir"))

    for r in new_rows:
        arm = base_arm(r.get("arm"))
        rd = r.get("run_dir")
        snap = is_snapshot(r.get("arm"))
        for label, (want_arm, proto) in NEW_ARMS.items():
            if arm != want_arm:
                continue
            if proto == "fixed" and not snap:
                put(label, r.get("n_train"), r, r.get("steps") or 4000)
            elif proto == "selected" and snap and rd in selected:
                put(label, r.get("n_train"), r, selected[rd])
    for r in ref_rows:
        for label, want_arm in REF_LABEL.items():
            if r.get("arm") == want_arm:
                put(label, r.get("n_train"), r, r.get("selected_steps"))
    return out


def fmt_steps(steps):
    v = [int(s) for s in steps if s is not None]
    if not v:
        return "--"
    if len(set(v)) == 1:
        return "%d" % v[0]
    counts = {}
    for s in v:
        counts[s] = counts.get(s, 0) + 1
    return ", ".join("%d x%d" % (s, counts[s]) for s in sorted(counts))


def main():
    data = collect()
    if not data:
        raise SystemExit("make_seed_tables_d.py: no rows found -- run the grid first")
    order = ([l for l in NEW_ARMS] + [l for l in REF_LABEL])
    lines = []
    A = lines.append

    A("# Seed tables - part D: finer grid (12 qubits) and non-linear latent map")
    A("")
    A("Every number is mean +- sd over seeds 0-4, scored by `evaluate_v11.py` against the same")
    A("200 held-out frames Table 2 uses, on the same `data_lens_r3` split (`R3_SPLIT_SEED = 0`;")
    A("`data_lens_n12` takes its images from that same loader, so the 12-qubit arm sees exactly")
    A("the same frames).  The three Table 2 rows are read from")
    A("`results_seeds_table2_v11.jsonl`, not recomputed.  Lower is better for PDF-JS and both")
    A("Conv-MMD2 columns; higher is better for the projection correlation.")
    A("")
    A("| arm | what it changes | parameters (N=100) |")
    A("|---|---|---|")
    for label in order:
        for n in (100, 11):
            key = (label, n)
            if key in data and n == 100:
                p = [x for x in data[key]["param"] if x]
                what = {
                    "IQP-n12": "12 qubits, whole 64x64 frame on 4,096 codes, no 32x32 crop",
                    "IQP-mlpH32-M4 (f": "theta = theta0 + g(z), g a tanh MLP (H=32), magnitude calibrated to the affine map's at init",
                    "IQP-mlpH32-M4 (v": "the same, checkpoint chosen on the validation block",
                    "IQP-mlpH32-M4 z": "the same MLP with ||g(z)|| = PHI*||z|| enforced for every z",
                    "IQP-mlpH32-M2": "the same MLP with a 2-dimensional latent",
                    "IQP-n12-affine-M4 z-init": "diagnostic: the 12-qubit arm with z started at the 10-qubit arm's own latent scale (0.05)",
                }
                w = "-- (reference)"
                # longest prefix first: "IQP-n12-affine-M4 z-init" must beat "IQP-n12"
                for pref, txt in sorted(what.items(), key=lambda kv: -len(kv[0])):
                    if label.startswith(pref):
                        w = txt
                        break
                if label.startswith("IQP-n12") and "val-selected" in label:
                    w = "the same, checkpoint chosen on the validation block"
                A("| %s | %s | %s |" % (label, w, ("%d" % p[0]) if p else "--"))
    A("")

    # ---------------------------------------------------------------- main table
    A("## 1. Main table (v11 metrics, 5 seeds)")
    A("")
    A("| arm | N | params | PDF-JS | Conv-MMD2 | Conv-MMD2 (flux-aligned) | proj. corr | steps |")
    A("|---|---|---|---|---|---|---|---|")
    stats = {}
    for n in (11, 100):
        for label in order:
            key = (label, n)
            if key not in data:
                continue
            d = data[key]
            cells = []
            for mkey, _name, _dir in METRICS:
                txt, m, s = pm(d[mkey], 4 if mkey != "projection_corr_mean" else 4)
                stats[(label, n, mkey)] = (m, s)
                cells.append(txt)
            p = [x for x in d["param"] if x]
            A("| %s | %d | %s | %s | %s | %s | %s | %s |"
              % (label, n, ("%d" % p[0]) if p else "--", cells[0], cells[1], cells[2],
                 cells[3], fmt_steps(d["steps"])))
        A("| | | | | | | | |")
    A("")
    A("`steps` is the number of joint Adam steps behind the reported checkpoint: 4,000 for the")
    A("fixed protocol (Table 2's own number) and, for the val-selected rows, whatever")
    A("`select_checkpoint.py` picked per seed out of {1k, 2k, 4k, 8k, 16k} using only the")
    A("validation block, which is disjoint from the 200 held-out frames.")
    A("")

    # ---------------------------------------------------------------- overlap table
    A("## 2. Does either change beat the current best IQP arm by more than 1 sd?")
    A("")
    A("`separated` = the two mean +- 1 sd intervals do not overlap; `overlap` = they do, i.e.")
    A("the seed spread already covers the difference and the arms are not distinguishable at")
    A("this sample size.  `better`/`worse` is the direction of the means.")
    A("")
    ref_label = "IQP-affine-M4 (Table 2, fixed 4k)"
    A("| arm | N | metric | this arm | %s | verdict |" % ref_label)
    A("|---|---|---|---|---|---|")
    for n in (11, 100):
        for label in [l for l in NEW_ARMS]:
            for mkey, name, direction in METRICS:
                a = stats.get((label, n, mkey))
                b = stats.get((ref_label, n, mkey))
                if not a or not b or a[0] is None or b[0] is None:
                    continue
                better = (a[0] < b[0]) if direction == "lower" else (a[0] > b[0])
                ov = overlap(a[0], a[1], b[0], b[1])
                verdict = "%s, %s" % ("better" if better else "worse",
                                      "overlap (not distinguishable)" if ov else "separated by >1 sd")
                A("| %s | %d | %s | %.4f +- %.4f | %.4f +- %.4f | %s |"
                  % (label, n, name, a[0], a[1], b[0], b[1], verdict))
        A("| | | | | | |")
    A("")

    # ---------------------------------------------------------------- diagnostics
    A("## 3. Diagnostics: was the change actually exercised?")
    A("")
    A("`nonlinearity ratio` is the relative residual of the least-squares best AFFINE fit to")
    A("the learned map on the trained latents: 0 means the MLP collapsed to exactly the affine")
    A("map the reference arm already has, so a null result would be a null test rather than a")
    A("finding.  `|pre|` is the mean absolute tanh pre-activation (tanh is linear to a part in")
    A("10^3 below ~0.05).  `train corr` and `proj. corr` are the fit to the training frames and")
    A("to the held-out frames through the z-only projection.")
    A("")
    A("`fit gain` is 1 - (data loss at the reported checkpoint) / L0, where L0 is the same")
    A("objective's loss at theta0 with z = 0: the fraction of the fittable loss the latent map")
    A("removed.  Near 0 means the arm is barely doing better than one shared hologram.")
    A("")
    A("| arm | N | fit gain | nonlinearity ratio | mean \\|pre\\| | \\|\\|z\\|\\| | train corr | proj. corr |")
    A("|---|---|---|---|---|---|---|---|")
    for n in (11, 100):
        for label in order:
            key = (label, n)
            if key not in data:
                continue
            d = data[key]
            A("| %s | %d | %s | %s | %s | %s | %s | %s |"
              % (label, n, pm(d["fit_gain"], 4)[0], pm(d["nonlinearity_ratio"], 4)[0],
                 pm(d["pre_abs_mean"], 3)[0],
                 pm(d["z_norm_mean"], 4)[0], pm(d["train_corr_mean"], 4)[0],
                 pm(d["projection_corr_mean"], 4)[0]))
        A("| | | | | | | | |")
    A("")

    # ---------------------------------------------------------------- collapse check
    A("## 3b. Per-seed collapse check")
    A("")
    A("A cell whose seeds do not all behave the same way must not be read as a mean +- sd.")
    A("This lists every (arm, N) cell in which at least one seed ended with a training-set")
    A("correlation below 0.5, i.e. the run did not fit the data it was given at all.")
    A("")
    rows_c = []
    for n in (11, 100):
        for label in order:
            key = (label, n)
            if key not in data:
                continue
            tc = data[key]["train_corr_mean"]
            seeds = data[key]["seeds"]
            bad = [(sd, v) for sd, v in zip(seeds, tc) if v is not None and v < 0.5]
            if bad:
                rows_c.append((label, n, len(bad), len(tc),
                               ", ".join("seed %s: %.3f" % (sd, v) for sd, v in sorted(bad))))
    if rows_c:
        A("| arm | N | collapsed seeds | per-seed train corr of the collapsed runs |")
        A("|---|---|---|---|")
        for label, n, k, tot, detail in rows_c:
            A("| %s | %d | %d of %d | %s |" % (label, n, k, tot, detail))
    else:
        A("No cell has a seed with training correlation below 0.5.")
    A("")

    # ---------------------------------------------------------------- provenance
    A("## 4. Provenance")
    A("")
    A("| what | where |")
    A("|---|---|")
    A("| new arms, v11 metrics | `%s` |" % os.path.relpath(NEW_J, REPO))
    A("| checkpoint selection | `%s` |" % os.path.relpath(ES_J, REPO))
    A("| Table 2 reference rows | `%s` |" % os.path.relpath(REF_J, REPO))
    A("| trainer (A) | `experiments/13_iqp_turbulence_replication/train_iqp_affine_n12.py` |")
    A("| trainer (B) | `experiments/13_iqp_turbulence_replication/train_iqp_affine_mlp.py` |")
    A("| loader (A) | `experiments/13_iqp_turbulence_replication/data_lens_n12.py` |")
    A("| grid | `experiments/13_iqp_turbulence_replication/seeds/submit_seeds_grid_mlp.sbatch` |")
    A("")
    n_rows = sum(len(v["seeds"]) for v in data.values())
    A("%d scored runs across %d (arm, N) cells." % (n_rows, len(data)))
    A("")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines))
    print("[tables-d] wrote %s (%d lines, %d cells, %d runs)"
          % (OUT_MD, len(lines), len(data), n_rows))
    for key in sorted(data, key=lambda k: (k[1], order.index(k[0]) if k[0] in order else 99)):
        print("   %-40s N=%-4d seeds=%d" % (key[0], key[1], len(data[key]["seeds"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
