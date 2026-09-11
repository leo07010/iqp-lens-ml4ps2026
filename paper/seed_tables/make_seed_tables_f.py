# -*- coding: utf-8 -*-
"""Seed tables, part F: the full experiment suite for the non-linear (MLP) IQP arm.

theta_i = theta0 + g(z_i), g a tanh MLP -- train_iqp_affine_mlp.py.  Everything is mean +- sd
over seeds 0-4 at the fixed 4,000 steps unless a section says otherwise, scored by
evaluate_v11.py against the same 200 held-out frames Table 2 uses.

Reads (nothing typed in by hand):
    results_seeds_mlp_suite.jsonl      phases 1-2 of the suite (M/H sweeps, Table 2 targets,
                                       scaling, per-class, ablations, the N=20 sweep)
    results_seeds_mlp_suite_es.jsonl   which checkpoint the validation block selected
    results_seeds_table2_v11.jsonl     RBM-10bit, DCGAN-32 and the published affine IQP arm
    results_seeds_scaling_v11.jsonl    the three published data-scaling curves
    results_seeds_class_v11.jsonl      the published per-class affine rows
    results_seeds_gridmlp.jsonl        part D (the same MLP arm at M=4, H=32)
    results_seeds_mlpctrl.jsonl        part D2 (the same latent model with no circuit)
Python 3.9 compatible.
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
F = lambda name: os.path.join(EXP, name)
OUT_MD = os.path.join(HERE, "seed_tables_f.md")

SUITE, SUITE_ES = F("results_seeds_mlp_suite.jsonl"), F("results_seeds_mlp_suite_es.jsonl")
T2, SCALE, CLS = (F("results_seeds_table2_v11.jsonl"), F("results_seeds_scaling_v11.jsonl"),
                  F("results_seeds_class_v11.jsonl"))
PARTD, PARTD2 = F("results_seeds_gridmlp.jsonl"), F("results_seeds_mlpctrl.jsonl")

SNAP_RE = re.compile(r"\s+s(\d+)k$")
MLP_RE = re.compile(r"^IQP-mlpH(\d+)-M(\d+)-(MMDflat|auto3)-resample$")
METRICS = [("pdf_js", "PDF-JS", "lower"), ("conv_mmd2", "MMD2", "lower"),
           ("conv_mmd2_aligned", "MMD2_al", "lower"),
           ("projection_corr_mean", "proj corr", "higher")]


def read(path):
    out = []
    if os.path.isfile(path):
        for line in open(path):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


_META = {}


def meta_of(rd):
    if rd not in _META:
        p = os.path.join(rd or "", "meta.json")
        _META[rd] = json.load(open(p)) if os.path.isfile(p) else {}
    return _META[rd]


def enrich(r):
    """Add the meta-only fields evaluate_v11 does not carry."""
    m = meta_of(r.get("run_dir"))
    r = dict(r)
    for k in ("projection_corr_mean", "nonlinearity_ratio", "z_norm_mean", "train_corr_mean",
              "hidden", "M", "restart_pick", "theta0_objective_arg", "projection_restarts"):
        r.setdefault(k, m.get(k))
        if r.get(k) is None:
            r[k] = m.get(k)
    L0, Ld = m.get("zpen_L0"), m.get("loss_final_data")
    r["fit_gain"] = (1.0 - float(Ld) / float(L0)) if (L0 not in (None, 0) and Ld is not None) else None
    return r


def agg(rows, key):
    v = [r.get(key) for r in rows]
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return None, None
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def cell(rows, key, digits=4):
    m, s = agg(rows, key)
    return "--" if m is None else "%.*f +- %.*f" % (digits, m, digits, s)


def verdict(rows, ref_rows, key, direction):
    a, sa = agg(rows, key)
    b, sb = agg(ref_rows, key)
    if a is None or b is None:
        return "--"
    better = (a < b) if direction == "lower" else (a > b)
    ov = not (a + sa < b - sb or b + sb < a - sa)
    return "%s / %s" % ("better" if better else "worse", "overlap" if ov else "SEPARATED")


def group(rows, keyfn):
    g = OrderedDict()
    for r in rows:
        k = keyfn(r)
        if k is not None:
            g.setdefault(k, []).append(r)
    return g


def is_snap(arm):
    return bool(SNAP_RE.search(str(arm or "")))


def main():
    suite = [enrich(r) for r in read(SUITE)]
    if not suite:
        raise SystemExit("make_seed_tables_f.py: %s is empty -- run the suite first" % SUITE)
    fixed = [r for r in suite if not is_snap(r.get("arm"))]
    t2 = [enrich(r) for r in read(T2)]
    scale = [enrich(r) for r in read(SCALE)]
    cls = [enrich(r) for r in read(CLS)]
    partd = [enrich(r) for r in read(PARTD) if not is_snap(r.get("arm"))]
    partd2 = [enrich(r) for r in read(PARTD2)]
    es_sel = {}
    for r in read(SUITE_ES):
        if r.get("kind") == "mlpsuite_early_stopped" and r.get("run_dir"):
            es_sel[r["run_dir"]] = r.get("selected_steps")

    def pick(rows, arm=None, n=None, cl="any", pred=None):
        out = []
        for r in rows:
            if arm is not None and r.get("arm") != arm:
                continue
            if n is not None and int(r.get("n_train", -1)) != int(n):
                continue
            if cl != "any" and r.get("class_name") != cl:
                continue
            if pred is not None and not pred(r):
                continue
            out.append(r)
        return out

    L = []
    A = L.append
    A("# Seed tables - part F: the non-linear (MLP) IQP arm, full suite")
    A("")
    A("`theta_i = theta0 + g(z_i)`, g a tanh MLP (`train_iqp_affine_mlp.py`).  Protocol frozen")
    A("to v11 and part D: `data_lens_r3` split (`R3_SPLIT_SEED = 0`), the same 200 held-out")
    A("frames, seeds 0-4, `--prior resample`, `--gnorm rownorm`, `--g-input-scale 20`,")
    A("1,000-step theta0 KL warm start, 4,000 fixed joint Adam steps, `evaluate_v11.py`.")
    A("Lower is better for PDF-JS and both MMD2 columns; higher is better for the projection")
    A("correlation.  Verdict cells read `direction / 1-sd`: `SEPARATED` = the mean +- 1 sd")
    A("intervals do not overlap, `overlap` = they do and the arms are not distinguishable.")
    A("")

    # ---------------------------------------------------------------- 0. context
    A("## 0. Where this sits")
    A("")
    A("| arm | N | PDF-JS | MMD2 | MMD2_al | proj corr |")
    A("|---|---|---|---|---|---|")
    ctx = [("v11 affine IQP (Table 2)", t2, "IQP-affine-M4-MMDflat-resample"),
           ("part D MLP, M=4 H=32", partd, "IQP-mlpH32-M4-MMDflat-resample"),
           ("part D2 softmax control, H=32 (no circuit)", partd2, "MLPctrl-H32-M4-MMDflat-resample"),
           ("RBM-10bit (Table 2)", t2, "RBM-10bit"),
           ("DCGAN-32 (Table 2)", t2, "DCGAN-32")]
    for n in (11, 100):
        for label, src, arm in ctx:
            rows = pick(src, arm=arm, n=n, cl=None)
            if rows:
                A("| %s | %d | %s | %s | %s | %s |"
                  % (label, n, cell(rows, "pdf_js"), cell(rows, "conv_mmd2"),
                     cell(rows, "conv_mmd2_aligned"), cell(rows, "projection_corr_mean")))
        A("| | | | | | |")
    A("")

    ref11 = pick(t2, arm="IQP-affine-M4-MMDflat-resample", n=11, cl=None)
    ref100 = pick(t2, arm="IQP-affine-M4-MMDflat-resample", n=100, cl=None)
    rbm = {n: pick(t2, arm="RBM-10bit", n=n, cl=None) for n in (11, 100)}
    dcg = {n: pick(t2, arm="DCGAN-32", n=n, cl=None) for n in (11, 100)}

    # ---------------------------------------------------------------- 1. M sweep
    A("## 1. Latent-width sweep (M), flat MMD")
    A("")
    A("`nonlin` is the relative residual of the best affine fit to the learned map on the")
    A("trained latents.  It is meaningless where M >= N (the fit is then exact by")
    A("construction) and is printed as `n/a` there.")
    A("")
    A("| M | H | N | params | PDF-JS | MMD2 | MMD2_al | proj corr | \\|\\|z\\|\\| | nonlin | vs RBM (MMD2_al) | vs DCGAN (MMD2_al) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    sweep = [r for r in fixed if MLP_RE.match(str(r.get("arm") or "")) and r.get("class_name") is None]
    msweep = {}
    for n in (11, 100):
        for H in (32, 8):
            for M in (1, 2, 3, 4, 6, 8, 16):
                arm = "IQP-mlpH%d-M%d-MMDflat-resample" % (H, M)
                rows = pick(sweep, arm=arm, n=n, cl=None)
                if not rows:
                    continue
                msweep[(M, H, n)] = rows
                nl = "n/a" if M >= n else cell(rows, "nonlinearity_ratio")
                p = [r.get("param_count") for r in rows if r.get("param_count")]
                A("| %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                  % (M, H, n, ("%d" % p[0]) if p else "--", cell(rows, "pdf_js"),
                     cell(rows, "conv_mmd2"), cell(rows, "conv_mmd2_aligned"),
                     cell(rows, "projection_corr_mean"), cell(rows, "z_norm_mean"), nl,
                     verdict(rows, rbm[n], "conv_mmd2_aligned", "lower"),
                     verdict(rows, dcg[n], "conv_mmd2_aligned", "lower")))
        A("| | | | | | | | | | | | |")
    A("")

    # ---------------------------------------------------------------- 2. H sweep
    A("## 2. Hidden-width sweep (H) at M = 4, flat MMD")
    A("")
    A("| H | N | params | PDF-JS | MMD2 | MMD2_al | proj corr | nonlin | vs affine IQP (MMD2_al) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for n in (11, 100):
        for H in (4, 8, 16, 32):
            rows = pick(sweep, arm="IQP-mlpH%d-M4-MMDflat-resample" % H, n=n, cl=None)
            if not rows:
                continue
            p = [r.get("param_count") for r in rows if r.get("param_count")]
            A("| %d | %d | %s | %s | %s | %s | %s | %s | %s |"
              % (H, n, ("%d" % p[0]) if p else "--", cell(rows, "pdf_js"),
                 cell(rows, "conv_mmd2"), cell(rows, "conv_mmd2_aligned"),
                 cell(rows, "projection_corr_mean"), cell(rows, "nonlinearity_ratio"),
                 verdict(rows, ref11 if n == 11 else ref100, "conv_mmd2_aligned", "lower")))
        A("| | | | | | | | | |")
    A("")

    # ---------------------------------------------------------------- 3. Table 2 targets
    A("## 3. Table 2 targets: the best combination under both objectives")
    A("")
    A("`steps` is 4,000 (the protocol Table 2 reports).  `selected` is what")
    A("`seeds/select_checkpoint.py` picked per seed out of {1k, 2k, 4k, 8k, 16k} on the")
    A("validation block, the symmetry check the classical baselines get.")
    A("")
    A("| arm | objective | N | params | PDF-JS | MMD2 | MMD2_al | proj corr | selected steps | vs affine | vs RBM | vs DCGAN |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    t2arms = sorted({r["arm"] for r in fixed
                     if MLP_RE.match(str(r.get("arm") or ""))
                     and MLP_RE.match(str(r["arm"])).group(3) == "auto3"})
    # the flat counterpart of every auto3 arm, plus any flat arm that also has a 16k sweep
    for a in sorted({a.replace("-auto3-", "-MMDflat-") for a in t2arms}):
        t2arms.append(a)
    seen = []
    for n in (11, 100):
        for arm in sorted(set(t2arms)):
            rows = pick(fixed, arm=arm, n=n, cl=None)
            if not rows:
                continue
            mm = MLP_RE.match(arm)
            sel = []
            for r in suite:
                if is_snap(r.get("arm")) and SNAP_RE.sub("", r["arm"]) == arm \
                        and int(r.get("n_train", -1)) == n and r.get("run_dir") in es_sel:
                    sel.append(es_sel[r["run_dir"]])
            p = [r.get("param_count") for r in rows if r.get("param_count")]
            A("| M=%s H=%s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (mm.group(2), mm.group(1), mm.group(3), n, ("%d" % p[0]) if p else "--",
                 cell(rows, "pdf_js"), cell(rows, "conv_mmd2"), cell(rows, "conv_mmd2_aligned"),
                 cell(rows, "projection_corr_mean"),
                 (", ".join("%d" % s for s in sorted(sel)) if sel else "--"),
                 verdict(rows, ref11 if n == 11 else ref100, "conv_mmd2_aligned", "lower"),
                 verdict(rows, rbm[n], "conv_mmd2_aligned", "lower"),
                 verdict(rows, dcg[n], "conv_mmd2_aligned", "lower")))
            seen.append((arm, n))
        A("| | | | | | | | | | | | |")
    A("")
    A("Per-metric 1-sd verdicts for the same rows, against DCGAN-32:")
    A("")
    A("| arm | objective | N | " + " | ".join(nm for _k, nm, _d in METRICS) + " |")
    A("|---|---|---|" + "---|" * len(METRICS))
    for arm, n in seen:
        mm = MLP_RE.match(arm)
        rows = pick(fixed, arm=arm, n=n, cl=None)
        A("| M=%s H=%s | %s | %d | %s |"
          % (mm.group(2), mm.group(1), mm.group(3), n,
             " | ".join(verdict(rows, dcg[n], k, d) for k, _nm, d in METRICS)))
    A("")

    # ---------------------------------------------------------------- 4. scaling
    A("## 4. Data scaling")
    A("")
    A("| N | MLP arm PDF-JS | MLP MMD2 | MLP MMD2_al | affine MMD2_al | RBM MMD2_al | DCGAN MMD2_al | MLP vs DCGAN (MMD2_al) |")
    A("|---|---|---|---|---|---|---|---|")
    sc_arm = None
    for r in fixed:
        if r.get("n_train") in (10, 30, 50, 70, 90) and MLP_RE.match(str(r.get("arm") or "")):
            sc_arm = r["arm"]
            break
    for n in (10, 11, 20, 30, 50, 70, 90, 100):
        rows = pick(fixed, arm=sc_arm, n=n, cl=None) if sc_arm else []
        if not rows:
            continue
        aff = pick(scale, arm="IQP-affine-M4-MMDflat-resample", n=n, cl=None)
        rb = pick(scale, arm="RBM-10bit", n=n, cl=None)
        dc = pick(scale, arm="DCGAN-32", n=n, cl=None)
        A("| %d | %s | %s | %s | %s | %s | %s | %s |"
          % (n, cell(rows, "pdf_js"), cell(rows, "conv_mmd2"), cell(rows, "conv_mmd2_aligned"),
             cell(aff, "conv_mmd2_aligned"), cell(rb, "conv_mmd2_aligned"),
             cell(dc, "conv_mmd2_aligned"),
             verdict(rows, dc, "conv_mmd2_aligned", "lower") if dc else "--"))
    A("")
    A("The scaling arm is `%s`.  N = 11, 20 and 100 come from other sections of this suite and"
      % (sc_arm or "--"))
    A("are shown where the published curves also have a point.")
    A("")

    # ---------------------------------------------------------------- 5. per class
    A("## 5. Per class")
    A("")
    A("The published per-class rows exist only for the affine IQP arm, so RBM and DCGAN have")
    A("no per-class counterpart to compare against.")
    A("")
    A("| class | N | MLP PDF-JS | MLP MMD2 | MLP MMD2_al | affine MMD2_al | vs affine (MMD2_al) |")
    A("|---|---|---|---|---|---|---|")
    for c in ("axion", "cdm", "no_sub"):
        for n in (11, 100):
            rows = pick(fixed, n=n, cl=c)
            rows = [r for r in rows if MLP_RE.match(str(r.get("arm") or ""))]
            if not rows:
                continue
            aff = pick(cls, arm="IQP-affine-M4-MMDflat-resample-%s" % c, n=n, cl=c)
            A("| %s | %d | %s | %s | %s | %s | %s |"
              % (c, n, cell(rows, "pdf_js"), cell(rows, "conv_mmd2"),
                 cell(rows, "conv_mmd2_aligned"), cell(aff, "conv_mmd2_aligned"),
                 verdict(rows, aff, "conv_mmd2_aligned", "lower") if aff else "--"))
    A("")

    # ---------------------------------------------------------------- 6. ablations
    A("## 6. Ablations")
    A("")
    A("Baseline = the same combination with the suite's own settings (KL warm start, 8")
    A("restarts, restart picked by correlation).")
    A("")
    A("| ablation | N | PDF-JS | MMD2 | MMD2_al | proj corr | vs baseline (proj corr) | vs baseline (MMD2_al) |")
    A("|---|---|---|---|---|---|---|---|")
    abl_arms = sorted({r["arm"] for r in fixed if str(r.get("arm", "")).startswith("IQP-mlpabl-")})
    base_arm = sc_arm
    for n in (11, 100):
        base = pick(fixed, arm=base_arm, n=n, cl=None) if base_arm else []
        if base:
            A("| baseline (%s) | %d | %s | %s | %s | %s | -- | -- |"
              % (base_arm, n, cell(base, "pdf_js"), cell(base, "conv_mmd2"),
                 cell(base, "conv_mmd2_aligned"), cell(base, "projection_corr_mean")))
        for arm in abl_arms:
            rows = pick(fixed, arm=arm, n=n, cl=None)
            if not rows:
                continue
            A("| %s | %d | %s | %s | %s | %s | %s | %s |"
              % (arm, n, cell(rows, "pdf_js"), cell(rows, "conv_mmd2"),
                 cell(rows, "conv_mmd2_aligned"), cell(rows, "projection_corr_mean"),
                 verdict(rows, base, "projection_corr_mean", "higher") if base else "--",
                 verdict(rows, base, "conv_mmd2_aligned", "lower") if base else "--"))
        A("| | | | | | | | |")
    A("")

    # ---------------------------------------------------------------- 7. provenance
    A("## 7. Provenance")
    A("")
    A("| what | where |")
    A("|---|---|")
    for name, p in (("suite rows", SUITE), ("checkpoint selection", SUITE_ES),
                    ("Table 2 rows", T2), ("scaling rows", SCALE), ("per-class rows", CLS),
                    ("part D", PARTD), ("part D2 control", PARTD2)):
        A("| %s | `%s` |" % (name, os.path.relpath(p, REPO)))
    A("| trainer | `experiments/13_iqp_turbulence_replication/train_iqp_affine_mlp.py` |")
    A("| ablation trainer | `experiments/13_iqp_turbulence_replication/train_iqp_affine_mlp_abl.py` |")
    A("| per-class wrapper | `experiments/13_iqp_turbulence_replication/seeds/run_mlp_class.py` |")
    A("| phase 1 / 2 / 3 | `seeds/submit_mlp_suite_sweep.sbatch`, `seeds/submit_mlp_suite_phase2.sbatch`, `seeds/submit_mlp_suite_phase3.sbatch` |")
    A("| run outputs | `<WORK>/gsoc_iqp_mlpsuite/` |")
    A("")
    A("%d scored suite rows (%d at the fixed 4,000 steps, %d snapshots)."
      % (len(suite), len(fixed), len(suite) - len(fixed)))
    A("")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L))
    print("[tables-f] wrote %s (%d lines) from %d suite rows" % (OUT_MD, len(L), len(suite)))
    print("[tables-f] M-sweep cells: %d | ablation arms: %d | scaling arm: %s"
          % (len(msweep), len(abl_arms), sc_arm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
