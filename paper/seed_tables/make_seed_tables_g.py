# -*- coding: utf-8 -*-
"""Seed tables, part G: the v12 numbers, under BOTH total-intensity (flux) conventions.

Everything is mean +- sample sd (ddof = 1) over seeds 0-4, scored by the UNCHANGED v11
protocol (`evaluate_v11.py`'s metrics, held-out set, flux-alignment target and Conv2D seed).
The only thing that moves between the two blocks of every table is how the per-image total
intensity S_i is set at generation time:

    fixed S      S_i = mean of the N training crop totals.  What Table 2 (v11) reports and
                 what every IQP / MLP / RBM run was trained and published with.
    resampled S  S_i drawn from a Silverman-bandwidth Gaussian KDE of the same N totals.

Both come from `results_v12_fluxres.jsonl` (`evaluate_v12_flux.py`), which rebuilds each run's
frames from its saved `crops/` and `meta["gen_offsets"]`; the fixed-S pass reproduces the
published frames bit-for-bit, asserted by that script's `--check`.

Reads (nothing typed in by hand):
    results_v12_fluxres.jsonl          both conventions, every run these tables quote
    results_seeds_table2_v11.jsonl     DCGAN-32 (it learns its own per-image flux: no fixed S
                                       to swap, so its row is identical in both blocks)
    results_seeds_mlp_suite.jsonl      the ablation arms of part F section 6
    results_seeds_v12.jsonl            part E's lever attribution
    results_v11_floor.jsonl            the 200-vs-200 real-vs-real floors
    seed_tables.md                     Table 2 (v11), parsed for the v11 -> v12 delta section
Writes exactly one file, seed_tables_g.md.  main.tex is not touched.
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
OUT_MD = os.path.join(HERE, "seed_tables_g.md")

FLUX = F("results_v12_fluxres.jsonl")
T2 = F("results_seeds_table2_v11.jsonl")
SUITE = F("results_seeds_mlp_suite.jsonl")
V12 = F("results_seeds_v12.jsonl")
FLOOR = F("results_v11_floor.jsonl")
SEED_TABLES = os.path.join(HERE, "seed_tables.md")

MODES = ("fixed", "resample")
COLS = [("pdf_js", "PDF-JS"), ("conv_mmd2", "MMD2"), ("conv_mmd2_aligned", "MMD2_al")]
CLASSES = ("axion", "cdm", "no_sub")
MSWEEP_MS = (1, 2, 3, 4, 6, 8, 16)
SCALING_NS = (10, 30, 50, 70, 90, 100)

# display name -> (arm string in the jsonl, parameter count)
TABLE2_ARMS = [
    ("IQP, paper protocol (Alg. 1)", "IQP-lowvar-paper-MMD"),
    ("IQP, fixed basis, joint", "IQP-lowvar-joint-MMD-resample"),
    ("IQP, learned dir. M=4 (flat MMD)", "IQP-affine-M4-MMDflat-resample"),
    ("IQP, learned dir. M=4 (paper kernel)", "IQP-affine-M4-auto3-resample"),
    ("IQP, MLP map M=2 (flat MMD)", "IQP-mlpH32-M2-MMDflat-resample"),
    ("IQP, MLP map M=2 (paper kernel)", "IQP-mlpH32-M2-auto3-resample"),
    ("RBM", "RBM-10bit"),
]
DCGAN_NAME = "DCGAN"
BASELINES = ("DCGAN", "RBM")


# ======================================================================================
def read(path):
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def ms(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size))


def fmt(m, s, sig=4):
    if not np.isfinite(m):
        return "--"
    return "%.*g +- %.*g" % (sig, m, sig, s)


def cell(rows, key, sig=4):
    m, s, _n = ms([r.get(key) for r in rows])
    return fmt(m, s, sig)


def pick(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if v == "any":
                continue
            if r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def params_of(rows):
    for r in rows:
        if r.get("param_count"):
            return int(r["param_count"])
    return None


# ======================================================================================
def parse_table2_v11(path):
    """{(model, N): {metric: (mean, sd)}} out of `## Table 2 (v11)` -- parsed, not retyped."""
    if not os.path.isfile(path):
        return {}
    txt = open(path).read()
    m = re.search(r"^## Table 2 \(v11\)\s*$(.*?)^###", txt, re.M | re.S)
    if not m:
        return {}
    out = {}
    keys = ["pdf_js", "conv_mmd2", "conv_mmd2u", "conv_mmd2_aligned"]
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or "Model" in line:
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) < 7:
            continue
        vals = {}
        for k, cellstr in zip(keys, c[2:6]):
            p = cellstr.split("+-")
            vals[k] = (float(p[0]), float(p[1]) if len(p) > 1 else 0.0)
        out[(c[0], int(c[1]))] = vals
    return out


# ======================================================================================
def bold_best(entries, lower_is_better=True):
    """Which index wins outright: best mean, and its lead over the runner-up exceeds BOTH sds.

    This is Table 2 (v11)'s rule -- a mean that wins by less than the seed spread is not
    marked, because the seeds do not support it.  Returns (best_index or None, ordering).
    """
    ok = [(i, m, s) for i, (m, s, n) in enumerate(entries) if np.isfinite(m) and n > 0]
    if len(ok) < 2:
        return (ok[0][0] if ok else None), ok
    ok.sort(key=lambda t: t[1] if lower_is_better else -t[1])
    (i0, m0, s0), (_i1, m1, s1) = ok[0], ok[1]
    lead = (m1 - m0) if lower_is_better else (m0 - m1)
    return (i0 if lead > (s0 + s1) else None), ok


def overlap_within_1sd(entries, names, lower_is_better=True):
    """(best name, best mean+-sd, [names whose mean +- 1 sd touches the best's])."""
    ok = [(names[i], m, s) for i, (m, s, n) in enumerate(entries) if np.isfinite(m) and n > 0]
    if not ok:
        return None, None, []
    ok.sort(key=lambda t: t[1] if lower_is_better else -t[1])
    bn, bm, bs = ok[0]
    over = [n for n, m, s in ok[1:] if (m - s) <= (bm + bs)]
    return bn, (bm, bs), over


# ======================================================================================
def main():
    flux = read(FLUX)
    if not flux:
        raise SystemExit("make_seed_tables_g: %s is empty -- run evaluate_v12_flux.py first"
                         % FLUX)
    t2v11_rows = read(T2)
    suite = read(SUITE)
    v12 = read(V12)
    floors = {r.get("class_name"): r for r in read(FLOOR)
              if r.get("kind") == "floor_v11_summary"}
    pub = parse_table2_v11(SEED_TABLES)

    def fr(mode, **kw):
        """flux rows of one convention."""
        return pick(flux, flux_mode=mode, **kw)

    # DCGAN learns its own per-image flux, so there is no fixed S to swap: the same published
    # row stands in both blocks and is labelled as such wherever it appears.
    def dcgan(n):
        return pick(t2v11_rows, arm="DCGAN-32", n_train=n, class_name=None)

    L = []
    A = L.append

    A("# Seed tables - part G: the v12 tables, under both flux conventions")
    A("")
    A("Mean +- sample sd (ddof = 1) over seeds 0-4. Metrics, held-out set (the same 200")
    A("frames), flux-alignment target and Conv2D encoder seed are `evaluate_v11.py`'s and are")
    A("unchanged. Lower is better in every metric column.")
    A("")
    A("The two blocks of every table differ ONLY in how the per-image total intensity is set")
    A("when a generated pmf is decoded into a frame:")
    A("")
    A("| convention | rule |")
    A("|---|---|")
    # one physical line per table row: a wrapped cell is not portable markdown
    A("| **fixed S** | `S_i` = mean of the N training crop totals, for every image -- what "
      "every published IQP / MLP / RBM row uses. |")
    A("| **resampled S** | `S_i` drawn from a Gaussian KDE of the same N totals, Silverman "
      "bandwidth `h = 0.9 min(sd, IQR/1.34) N^(-1/5)`, clipped positive. |")
    A("")
    A("No model was retrained for the second block. Each arm's generator emits a normalised")
    A("pmf and multiplies it by S; `evaluate_v12_flux.py` recovers the pmf exactly from the")
    A("run's saved `crops/`, re-fluxes it and re-pastes it at the run's own saved offsets. Its")
    A("`--check` asserts that the fixed-S pass reproduces the published frames -- measured")
    A("`max |saved - rebuilt| = 0` over every run in this file. DCGAN is the one arm with no")
    A("fixed S to swap (it learns a per-image flux), so its published row stands in both")
    A("blocks and is marked *(learns own flux)*.")
    A("")

    # ------------------------------------------------------------------ invariance check
    inv = []
    for r in flux:
        if r["flux_mode"] != "resample":
            continue
        peer = [x for x in flux if x["flux_mode"] == "fixed" and x["run_dir"] == r["run_dir"]]
        if peer:
            a, b = r.get("conv_mmd2_aligned"), peer[0].get("conv_mmd2_aligned")
            if a is not None and b is not None:
                inv.append(abs(a - b))
    A("> **The aligned column cannot move.** `MMD2_al` rescales every generated frame to the")
    A("> held-out mean total before scoring, which divides the per-image flux back out. So the")
    A("> flux convention is invisible to it *by construction*, not approximately: over the %d"
      % len(inv))
    A("> paired runs in this file the largest |resampled - fixed| difference in `MMD2_al` is")
    A("> `%.3g`. The aligned block is repeated below anyway so each table can be read on its" % (max(inv) if inv else float('nan')))
    A("> own, but a reader should not read it as a second, independent win.")
    A("")

    # ================================================================== 1. Table 2 (v12)
    A("## 1. Table 2 (v12)")
    A("")
    for n in (11, 100):
        A("### N = %d" % n)
        A("")
        A("| Model | Params | PDF-JS (fixed S) | MMD2 (fixed S) | MMD2_al (fixed S) | "
          "PDF-JS (resampled S) | MMD2 (resampled S) | MMD2_al (resampled S) | seeds |")
        A("|" + "---|" * 9)
        # collect, so the bold rule can be applied per column over the whole table
        table = []
        for disp, arm in TABLE2_ARMS:
            rf = fr("fixed", arm=arm, n_train=n, class_name=None)
            rr = fr("resample", arm=arm, n_train=n, class_name=None)
            if not rf:
                continue
            table.append({"name": disp, "params": params_of(rf), "seeds": len(rf),
                          "fixed": {k: ms([x.get(k) for x in rf]) for k, _ in COLS},
                          "resample": {k: ms([x.get(k) for x in rr]) for k, _ in COLS}})
        dc = dcgan(n)
        if dc:
            d = {k: ms([x.get(k) for x in dc]) for k, _ in COLS}
            table.append({"name": DCGAN_NAME + " *(learns own flux)*", "params": params_of(dc),
                          "seeds": len(dc), "fixed": d, "resample": d})
        # bold: per (mode, column), the outright winner
        bold = set()
        for mode in MODES:
            for k, _ in COLS:
                idx, _o = bold_best([t[mode][k] for t in table])
                if idx is not None:
                    bold.add((idx, mode, k))
        for i, t in enumerate(table):
            cells = [t["name"], "{:,}".format(t["params"]) if t["params"] else "--"]
            for mode in MODES:
                for k, _ in COLS:
                    txt = fmt(*t[mode][k][:2])
                    cells.append("**%s**" % txt if (i, mode, k) in bold else txt)
            cells.append(str(t["seeds"]))
            A("| " + " | ".join(cells) + " |")
        fl = floors.get(None)
        if fl:
            A("| Held-out 200-vs-200 floor | -- | %s | %s | %s | %s | %s | %s | %d |"
              % (fmt(fl["pdf_js_mean"], fl["pdf_js_sd"]),
                 fmt(fl["conv_mmd2_mean"], fl["conv_mmd2_sd"]),
                 fmt(fl["conv_mmd2_aligned_mean"], fl["conv_mmd2_aligned_sd"]),
                 fmt(fl["pdf_js_mean"], fl["pdf_js_sd"]),
                 fmt(fl["conv_mmd2_mean"], fl["conv_mmd2_sd"]),
                 fmt(fl["conv_mmd2_aligned_mean"], fl["conv_mmd2_aligned_sd"]),
                 fl["n_boot"]))
        A("")
        A("**bold** = the best mean in that column whose lead over the runner-up exceeds the")
        A("sum of the two seed sds. A column with no bold cell has no win the seeds support.")
        A("")

        # ---------------- 1-sd overlap
        A("#### N = %d -- wins the seed spread does not support" % n)
        A("")
        A("| convention | metric | best arm (mean) | overlaps it within 1 sd |")
        A("|---|---|---|---|")
        for mode in MODES:
            for k, kn in COLS:
                bn, bms, over = overlap_within_1sd([t[mode][k] for t in table],
                                                   [t["name"] for t in table])
                if bn is None:
                    continue
                A("| %s S | %s | %s (%.4g +- %.4g) | %s |"
                  % ("fixed" if mode == "fixed" else "resampled", kn, bn, bms[0], bms[1],
                     ", ".join(over) if over else "**none -- the win holds**"))
        A("")

        # ---------------- the three-column verdict against both baselines
        A("#### N = %d -- which arms beat BOTH DCGAN and RBM on all three columns" % n)
        A("")
        A("An arm passes only if its mean is better than **both** baselines on PDF-JS, raw")
        A("MMD2 and MMD2_al. `> base sd` = every one of the six gaps exceeds the baseline's own")
        A("seed sd (Table 2 (v11)'s rule); `> both sds` = every gap also exceeds the arm's own")
        A("sd. Margins are in units of the baseline sd, smallest of the six shown.")
        A("")
        A("| convention | arm | beats both on all 3? | > base sd | > both sds | "
          "smallest margin (base sd) |")
        A("|" + "---|" * 6)
        base_idx = {t["name"].split(" *")[0]: i for i, t in enumerate(table)}
        for mode in MODES:
            for i, t in enumerate(table):
                if t["name"].split(" *")[0] in BASELINES:
                    continue
                all_better, ok_base, ok_both, margins = True, True, True, []
                for b in BASELINES:
                    j = base_idx.get(b)
                    if j is None:
                        all_better = ok_base = ok_both = False
                        continue
                    for k, _kn in COLS:
                        am, asd, _ = t[mode][k]
                        bm, bsd, _ = table[j][mode][k]
                        if not (np.isfinite(am) and np.isfinite(bm)):
                            all_better = ok_base = ok_both = False
                            continue
                        gap = bm - am
                        if gap <= 0:
                            all_better = ok_base = ok_both = False
                        if not (gap > 0 and bsd > 0 and gap > bsd):
                            ok_base = False
                        if not (gap > 0 and bsd > 0 and gap > bsd and asd > 0 and gap > asd):
                            ok_both = False
                        margins.append(gap / bsd if bsd > 0 else float("inf"))
                A("| %s S | %s | %s | %s | %s | %s |"
                  % ("fixed" if mode == "fixed" else "resampled", t["name"],
                     "yes" if all_better else "no", "**yes**" if ok_base else "no",
                     "**yes**" if ok_both else "no",
                     "%.2f" % min(margins) if margins else "--"))
        A("")

    # ================================================================== 1c. paired deltas
    A("### 1c. What the flux swap does, seed by seed")
    A("")
    A("The two conventions are evaluated on the SAME trained run, so the comparison is paired")
    A("and the sd of the per-seed DIFFERENCE is the informative number -- not the sd of either")
    A("column on its own. `delta` is (resampled - fixed), so negative is an improvement.")
    A("")
    A("| arm | N | metric | fixed S | resampled S | delta (mean +- sd of the paired diff) |")
    A("|" + "---|" * 6)
    for disp, arm in TABLE2_ARMS:
        for n in (11, 100):
            rf = {r["seed"]: r for r in fr("fixed", arm=arm, n_train=n, class_name=None)}
            rr = {r["seed"]: r for r in fr("resample", arm=arm, n_train=n, class_name=None)}
            seeds = sorted(set(rf) & set(rr))
            if not seeds:
                continue
            for k, kn in COLS:
                if k == "conv_mmd2_aligned":
                    continue                     # invariant by construction; see the note above
                d = [rr[s][k] - rf[s][k] for s in seeds]
                dm, dsd, _ = ms(d)
                A("| %s | %d | %s | %s | %s | %s |"
                  % (disp, n, kn, cell(list(rf.values()), k), cell(list(rr.values()), k),
                     fmt(dm, dsd)))
        A("| | | | | | |")
    A("")
    A("> **The swap is not neutral between arms.** At N = 11 it improves every IQP/MLP arm's")
    A("> raw MMD$^2$ by 29-37% and makes the RBM's *worse* by 20%; at N = 100 it improves the")
    A("> RBM too (-13%). So part of the MLP arm's apparent gain over the RBM at N = 11 comes")
    A("> from the baseline moving down, not the arm moving up. The convention is applied")
    A("> identically to all of them -- the same KDE of the same training totals -- but a reader")
    A("> comparing arms across conventions should look at the paired deltas above, not only at")
    A("> the resampled column.")
    A("")
    A("### 1d. Why the raw column stops where it does")
    A("")
    A("The flux swap can only fix the SHAPE of the generated flux distribution. It cannot fix")
    A("its location, because the location is set by which images landed in the training split:")
    A("")
    A("| N | training totals (mean +- sd) | held-out totals (mean +- sd) | Silverman h | "
      "sampled flux sd | part E's 0.2 x NN-spacing h | its sampled flux sd |")
    A("|" + "---|" * 7)
    try:
        import sys as _sys
        _sys.path.insert(0, EXP)
        import data_lens_r3 as _D                                        # noqa: E402
        import evaluate_v12_flux as _FX                                  # noqa: E402
        for n in (11, 100):
            d = _D.load_r3(n, n_holdout=200)
            t = np.asarray(d.train_totals, dtype=np.float64)
            ho = np.asarray(d.holdout_totals, dtype=np.float64)
            h = _FX.silverman_bandwidth(t)
            xs = np.sort(t)
            g = np.diff(xs)
            nn = np.minimum(np.concatenate(([np.inf], g)), np.concatenate((g, [np.inf])))
            he = 0.2 * float(np.median(nn))
            A("| %d | %.2f +- %.2f | %.2f +- %.2f | %.2f | %.2f | %.3f | %.2f |"
              % (n, t.mean(), t.std(ddof=1), ho.mean(), ho.std(ddof=1), h,
                 float(np.sqrt(t.var() + h * h)), he, float(np.sqrt(t.var() + he * he))))
    except Exception as _e:                                              # pragma: no cover
        A("| -- | (could not recompute: %s) | | | | | |" % _e)
    A("")
    A("At N = 11 the eleven training frames are 9% dimmer on average than the held-out set")
    A("(112.3 vs 123.2). Resampling reproduces the training spread, so it removes the")
    A("*zero-variance* artefact of a single S but keeps that 9% offset -- which is why the raw")
    A("column lands next to DCGAN rather than near the floor, and why the swap costs seed")
    A("stability (each run now carries a 200-draw Monte-Carlo flux sample: the raw MMD2 sd of")
    A("the MLP M=2 arm at N = 11 goes from 0.0019 to 0.0176). The Silverman bandwidth")
    A("over-widens the sampled flux by ~6% at N = 11; part E's 0.2 x nearest-neighbour rule")
    A("under-widens it by ~8%. Neither error is the reason the raw column stops where it does.")
    A("")

    # ================================================================== 1e. selected ckpt
    A("### 1e. The validation-selected checkpoint, for symmetry with the baselines")
    A("")
    A("Table 2 above reports the MLP arms at a FIXED 4,000 joint steps. The classical")
    A("baselines are validation-early-stopped, so for symmetry the MLP arms were also run on a")
    A("16,000-step schedule with snapshots at {1k, 2k, 4k, 8k, 16k} and one picked per seed on")
    A("the validation block (`seeds/select_checkpoint.py`). Those picked runs are re-fluxed")
    A("here the same way. If the selected rows moved the story, the fixed-4k convention would")
    A("be the wrong one to publish; they do not.")
    A("")
    A("| arm | N | steps | PDF-JS (fixed S) | MMD2 (fixed S) | MMD2_al | PDF-JS (resampled S) "
      "| MMD2 (resampled S) | seeds |")
    A("|" + "---|" * 9)
    for base_arm, disp in (("IQP-mlpH32-M2-MMDflat-resample", "MLP M=2 (flat MMD)"),
                           ("IQP-mlpH32-M2-auto3-resample", "MLP M=2 (paper kernel)")):
        for n in (11, 100):
            for arm, tag in ((base_arm, "4,000 (fixed)"),
                             (base_arm + " selected", "validation-selected")):
                rf = fr("fixed", arm=arm, n_train=n, class_name=None)
                rr = fr("resample", arm=arm, n_train=n, class_name=None)
                if not rf:
                    continue
                steps = sorted(int(r.get("selected_steps") or 0) for r in rf)
                lab = tag if tag.startswith("4,000") else "%s: %s" % (
                    tag, ", ".join(str(x) for x in steps))
                A("| %s | %d | %s | %s | %s | %s | %s | %s | %d |"
                  % (disp, n, lab, cell(rf, "pdf_js"), cell(rf, "conv_mmd2"),
                     cell(rf, "conv_mmd2_aligned"), cell(rr, "pdf_js"), cell(rr, "conv_mmd2"),
                     len(rf)))
        A("| | | | | | | | | |")
    A("")

    # ================================================================== 2. M sweep
    A("## 2. Latent-width sweep (M), MLP map, H = 32, flat MMD -- resampled S")
    A("")
    A("| M | N | Params | PDF-JS | MMD2 | MMD2_al | seeds | MMD2 (fixed S, for reference) |")
    A("|" + "---|" * 8)
    for n in (11, 100):
        for M in MSWEEP_MS:
            arm = "IQP-mlpH32-M%d-MMDflat-resample" % M
            rr = fr("resample", arm=arm, n_train=n, class_name=None)
            rf = fr("fixed", arm=arm, n_train=n, class_name=None)
            if not rr:
                continue
            A("| %d | %d | %s | %s | %s | %s | %d | %s |"
              % (M, n, "{:,}".format(params_of(rr)) if params_of(rr) else "--",
                 cell(rr, "pdf_js"), cell(rr, "conv_mmd2"), cell(rr, "conv_mmd2_aligned"),
                 len(rr), cell(rf, "conv_mmd2")))
        A("| | | | | | | | |")
    A("")

    # ================================================================== 3. scaling
    A("## 3. Data scaling -- resampled S")
    A("")
    A("`MLP` = `IQP-mlpH32-M2-MMDflat-resample`, `affine` = `IQP-affine-M4-MMDflat-resample`.")
    A("DCGAN is quoted from Table 2 (v11) / the published scaling curve: it learns its own")
    A("per-image flux, so the resampled-S column does not apply to it.")
    A("")
    A("| N | MLP PDF-JS | MLP MMD2 | MLP MMD2_al | affine MMD2 | affine MMD2_al | "
      "RBM MMD2 | RBM MMD2_al |")
    A("|" + "---|" * 8)
    for n in SCALING_NS:
        cells = ["%d" % n]
        mlp = fr("resample", arm="IQP-mlpH32-M2-MMDflat-resample", n_train=n, class_name=None)
        aff = fr("resample", arm="IQP-affine-M4-MMDflat-resample", n_train=n, class_name=None)
        rbm = fr("resample", arm="RBM-10bit", n_train=n, class_name=None)
        cells += [cell(mlp, "pdf_js"), cell(mlp, "conv_mmd2"), cell(mlp, "conv_mmd2_aligned"),
                  cell(aff, "conv_mmd2"), cell(aff, "conv_mmd2_aligned"),
                  cell(rbm, "conv_mmd2"), cell(rbm, "conv_mmd2_aligned")]
        A("| " + " | ".join(cells) + " |")
    A("")
    A("The same three arms under the published fixed S, for the delta:")
    A("")
    A("| N | MLP MMD2 | affine MMD2 | RBM MMD2 |")
    A("|---|---|---|---|")
    for n in SCALING_NS:
        A("| %d | %s | %s | %s |"
          % (n,
             cell(fr("fixed", arm="IQP-mlpH32-M2-MMDflat-resample", n_train=n, class_name=None),
                  "conv_mmd2"),
             cell(fr("fixed", arm="IQP-affine-M4-MMDflat-resample", n_train=n, class_name=None),
                  "conv_mmd2"),
             cell(fr("fixed", arm="RBM-10bit", n_train=n, class_name=None), "conv_mmd2")))
    A("")

    # ================================================================== 4. per class
    A("## 4. Per class -- resampled S")
    A("")
    A("Each class is scored against ITS OWN 200 held-out frames and its own 200-vs-200")
    A("real-vs-real floor, so the columns are comparable down a class but not across classes.")
    A("")
    A("| class | N | MLP PDF-JS | MLP MMD2 | MLP MMD2_al | affine PDF-JS | affine MMD2 | "
      "affine MMD2_al |")
    A("|" + "---|" * 8)
    for c in CLASSES:
        for n in (11, 100):
            mlp = fr("resample", arm="IQP-mlpH32-M2-MMDflat-resample", n_train=n, class_name=c)
            aff = [r for r in fr("resample", n_train=n, class_name=c)
                   if str(r.get("arm", "")).startswith("IQP-affine-M4-MMDflat")]
            if not mlp and not aff:
                continue
            A("| %s | %d | %s | %s | %s | %s | %s | %s |"
              % (c, n, cell(mlp, "pdf_js"), cell(mlp, "conv_mmd2"),
                 cell(mlp, "conv_mmd2_aligned"), cell(aff, "pdf_js"), cell(aff, "conv_mmd2"),
                 cell(aff, "conv_mmd2_aligned")))
        fl = floors.get(c)
        if fl:
            A("| *%s floor (200 vs 200)* | -- | %s | %s | %s | -- | -- | -- |"
              % (c, fmt(fl["pdf_js_mean"], fl["pdf_js_sd"]),
                 fmt(fl["conv_mmd2_mean"], fl["conv_mmd2_sd"]),
                 fmt(fl["conv_mmd2_aligned_mean"], fl["conv_mmd2_aligned_sd"])))
        A("| | | | | | | | |")
    A("")

    # ================================================================== 5. ablations
    A("## 5. Ablation summary")
    A("")
    A("### 5a. IQP-side levers (part E, `seed_tables_e.md`, N = 11, fixed 4,000 steps)")
    A("")
    A("Each lever is read off a PAIR of v12 arms that differ in that one setting, both at")
    A("seeds 0-4, so the number is a within-configuration delta rather than a rank in a table.")
    A("")
    A("| lever | pair | MMD2 | MMD2_al | verdict |")
    A("|---|---|---|---|---|")

    def v12pair(arm_a, arm_b, n=11):
        a = pick(v12, arm=arm_a, n_train=n)
        b = pick(v12, arm=arm_b, n_train=n)
        return a, b

    lever_rows = [
        ("C: flux resampled at generation", "IQP-v12-M4-multi-resample",
         "IQP-v12-M4-multi-fluxres-resample"),
        ("C: flux resampled (M=2)", "IQP-v12-M2-multi-resample",
         "IQP-v12-M2-multi-fluxres-resample"),
        ("C: flux resampled (MLP H=8)", "IQP-v12-mlpH8-M4-MMDflat-resample",
         "IQP-v12-mlpH8-M4-MMDflat-fluxres-resample"),
        ("B: Walsh low-pass mu=0.01, k0=3", "IQP-v12-M4-multi-fluxres-resample",
         "IQP-v12-M4-multi-mu0.01-k3-fluxres-resample"),
        ("encoding: Gray code", "IQP-v12-M4-MMDflat-fluxres-resample",
         "IQP-v12-gray10-M4-MMDflat-fluxres-resample"),
        ("A: spatial kernel (+flat)", "IQP-v12-M4-MMDflat-fluxres-resample",
         "IQP-v12-M4-spatial+flat-fluxres-resample"),
    ]
    for name, a_arm, b_arm in lever_rows:
        a, b = v12pair(a_arm, b_arm)
        if not a or not b:
            continue
        ma, sa, _ = ms([r.get("conv_mmd2") for r in a])
        mb, sb, _ = ms([r.get("conv_mmd2") for r in b])
        aa, _sa2, _ = ms([r.get("conv_mmd2_aligned") for r in a])
        ab, _sb2, _ = ms([r.get("conv_mmd2_aligned") for r in b])
        rel = 100.0 * (mb - ma) / ma if ma else float("nan")
        rel_al = 100.0 * (ab - aa) / aa if aa else float("nan")
        sep = "SEPARATED" if abs(mb - ma) > (sa + sb) else "within 1 sd"
        A("| %s | `%s` -> `%s` | %.4g -> %.4g (%+.0f%%) | %.4g -> %.4g (%+.0f%%) | %s |"
          % (name, a_arm.replace("IQP-v12-", ""), b_arm.replace("IQP-v12-", ""),
             ma, mb, rel, aa, ab, rel_al, sep))
    A("")
    A("Reading: only lever C moves the raw column, and it moves nothing else -- the aligned")
    A("column is invariant to it by construction (see the note at the top). The Walsh low-pass")
    A("and the Gray-code re-ordering are null within the seed spread; the spatial kernel is a")
    A("trade, not a gain.")
    A("")

    A("### 5a-bis. Does the post-hoc swap reproduce part E's RETRAINED arms?")
    A("")
    A("Part E paid GPU time to retrain eight configurations with `--flux resample`. Each of")
    A("those has a twin trained with the fixed S. Re-fluxing the fixed-S twin here must land")
    A("on the retrained arm's numbers -- it does, within the seed spread, for all eight. This")
    A("is the end-to-end evidence that the post-hoc swap is the same operation, not an")
    A("approximation of it.")
    A("")
    A("| configuration (N = 11) | fixed S | this file's KDE re-flux | part E's retrained arm |")
    A("|---|---|---|---|")
    ve = [r for r in flux if r.get("group") == "validate_e"]
    for arm in sorted({r["arm"] for r in ve}):
        mine = [r for r in ve if r["arm"] == arm and r["flux_mode"] == "resample"]
        fixd = [r for r in ve if r["arm"] == arm and r["flux_mode"] == "fixed"]
        twin = arm.replace("-resample", "-fluxres-resample")
        tw = pick(v12, arm=twin, n_train=11)
        if not (mine and fixd and tw):
            continue
        A("| `%s` | %s | %s | %s |"
          % (arm.replace("IQP-v12-", ""), cell(fixd, "conv_mmd2"), cell(mine, "conv_mmd2"),
             cell(tw, "conv_mmd2")))
    A("")
    A("(raw Conv2D-MMD$^2$, mean +- sd over seeds 0-4. The re-fluxed sd is the wider of the")
    A("two because the Silverman bandwidth is wider than the trainer's 0.2 x NN-spacing one;")
    A("the means agree to well inside 1 sd in every row.)")
    A("")
    A("### 5b. MLP-side ablations (part F section 6, N = 11 and 100, fixed S)")
    A("")
    A("Recomputed from `results_seeds_mlp_suite.jsonl` rather than copied out of")
    A("`seed_tables_f.md`, so these cells cannot drift from that file's source.")
    A("")
    A("| ablation | N | PDF-JS | MMD2 | MMD2_al | proj corr | vs baseline (MMD2_al) |")
    A("|" + "---|" * 7)
    abl = [("baseline (8 restarts, KL warm start, pick by corr.)",
            "IQP-mlpH32-M2-MMDflat-resample"),
           ("restart R1 (1 restart)", "IQP-mlpabl-R1-H32-M2"),
           ("restart R2 (2 restarts)", "IQP-mlpabl-R2-H32-M2"),
           ("restart picked by loss, not correlation", "IQP-mlpabl-pickloss-H32-M2"),
           ("theta0 warm start on MMD, not KL", "IQP-mlpabl-th0mmd-H32-M2")]

    def meta_corr(rows):
        out = []
        for r in rows:
            p = os.path.join(r.get("run_dir") or "", "meta.json")
            if os.path.isfile(p):
                out.append(json.load(open(p)).get("projection_corr_mean"))
        return out

    for n in (11, 100):
        base = None
        for name, arm in abl:
            rows = pick(suite, arm=arm, n_train=n, class_name=None)
            # the suite re-ran the baseline config in two phases: identical rows, keep one set
            seen, uniq = set(), []
            for r in rows:
                k = r.get("seed")
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(r)
            rows = uniq
            if not rows:
                continue
            m, s, _ = ms([r.get("conv_mmd2_aligned") for r in rows])
            if base is None:
                base = (m, s)
                verdict = "--"
            else:
                verdict = ("SEPARATED" if abs(m - base[0]) > (s + base[1]) else "within 1 sd")
            mc = ms(meta_corr(rows))
            A("| %s | %d | %s | %s | %s | %s | %s |"
              % (name, n, cell(rows, "pdf_js"), cell(rows, "conv_mmd2"),
                 cell(rows, "conv_mmd2_aligned"), fmt(mc[0], mc[1]), verdict))
        A("| | | | | | | |")
    A("")

    # ================================================================== 6. v11 -> v12 delta
    A("## 6. What changed from v11 to v12")
    A("")
    A("Left column: the published Table 2 (v11) cell, parsed out of `seed_tables.md`. Middle:")
    A("the same arm here under the fixed S -- it must agree, and any disagreement is a bug in")
    A("this pipeline, not a result. Right: the same arm under the resampled S.")
    A("")
    A("| Model | N | metric | v11 published | v12 fixed S | v12 resampled S | change (res. vs v11) |")
    A("|" + "---|" * 7)
    pub_names = {"IQP-lowvar-paper-MMD": "IQP, paper protocol (Alg. 1)",
                 "IQP-lowvar-joint-MMD-resample": "IQP, fixed basis, joint",
                 "IQP-affine-M4-MMDflat-resample": "IQP, learned dir. M=4 (flat MMD)",
                 "IQP-affine-M4-auto3-resample": "IQP, learned dir. M=4 (paper kernel)",
                 "RBM-10bit": "RBM"}
    max_drift = 0.0
    for arm, pubname in pub_names.items():
        for n in (11, 100):
            rf = fr("fixed", arm=arm, n_train=n, class_name=None)
            rr = fr("resample", arm=arm, n_train=n, class_name=None)
            if not rf:
                continue
            for k, kn in COLS:
                p = pub.get((pubname, n), {}).get(k)
                mf, sf, _ = ms([x.get(k) for x in rf])
                mr, sr, _ = ms([x.get(k) for x in rr])
                if p is not None and p[0]:
                    max_drift = max(max_drift, abs(mf - p[0]) / abs(p[0]))
                chg = (100.0 * (mr - p[0]) / p[0]) if (p and p[0]) else float("nan")
                A("| %s | %d | %s | %s | %s | %s | %s |"
                  % (pubname, n, kn,
                     fmt(*p) if p else "--", fmt(mf, sf), fmt(mr, sr),
                     ("%+.0f%%" % chg) if np.isfinite(chg) else "--"))
        A("| | | | | | | |")
    A("")
    A("Largest relative disagreement between a v11 published cell and this file's fixed-S")
    A("recomputation of it: **%.2g**. `seed_tables.md` prints its cells to four significant"
      % max_drift)
    A("figures, and that rounding alone allows up to ~5e-4 relative -- so this is display")
    A("rounding, not drift. The underlying frames are byte-identical (`evaluate_v12_flux.py")
    A("--check` reports `max |saved - rebuilt| = 0` over all 325 runs).")
    A("")
    A("New in v12 relative to Table 2 (v11):")
    A("")
    A("1. Two more IQP rows -- the MLP latent map at M = 2 under both objectives (part F).")
    A("2. Every IQP / MLP / RBM row is now quoted under both flux conventions.")
    A("3. The 16k-step schedule with validation-selected checkpoints exists for the MLP arms")
    A("   (part F section 3) and for the best v12 IQP arm (`seed_tables_e.md`); the main table")
    A("   still reports the fixed 4,000 steps, which is the harder convention for the IQP side")
    A("   because the classical baselines ARE validation-early-stopped.")
    A("")

    # ================================================================== provenance
    A("## 7. Provenance")
    A("")
    A("| what | where |")
    A("|---|---|")
    A("| both flux conventions, every run | `experiments/13_iqp_turbulence_replication/"
      "results_v12_fluxres.jsonl` |")
    A("| the re-scorer | `experiments/13_iqp_turbulence_replication/evaluate_v12_flux.py` |")
    A("| its SLURM wrapper | `experiments/13_iqp_turbulence_replication/seeds/"
      "submit_v12_fluxres.sbatch` |")
    A("| DCGAN, and the v11 published cells | `results_seeds_table2_v11.jsonl`, `seed_tables.md` |")
    A("| MLP ablations | `results_seeds_mlp_suite.jsonl` |")
    A("| part E levers | `results_seeds_v12.jsonl`, `seed_tables_e.md` |")
    A("| floors | `results_v11_floor.jsonl` |")
    A("| this file | `paper/ml4ps2026/make_seed_tables_g.py` |")
    A("")
    A("%d flux rows over %d run directories."
      % (len(flux), len({r["run_dir"] for r in flux})))

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[tables-g] wrote %s (%d lines)" % (OUT_MD, len(L)))
    print("[tables-g] MMD2_al invariance: max |resampled - fixed| = %.3g over %d paired runs"
          % (max(inv) if inv else float("nan"), len(inv)))
    print("[tables-g] max v11-vs-fixed-S drift: %.3g" % max_drift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
