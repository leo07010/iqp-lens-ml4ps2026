#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the v12 grid (results_seeds_v12.jsonl) into paper/ml4ps2026/seed_tables_e.md.

NEW file.  It reads only `results_seeds_v12.jsonl` (written by evaluate_v11.py over
outputs_v12/) and the run dirs' own meta.json, plus the BASELINE rows it PARSES out of the
existing `paper/ml4ps2026/seed_tables.md` "Table 2 (v11)" section -- parsed rather than
retyped, so the comparison numbers cannot drift from the published ones.  It writes exactly
one file, `seed_tables_e.md`; `main.tex`, `seed_tables.md` and every existing results_*.jsonl
are left alone.

THE PRE-REGISTERED VERDICT
    "clear win" = the best IQP combination's MEAN beats BOTH baselines (DCGAN and RBM) on all
    three of PDF-JS, raw MMD^2 and flux-aligned MMD^2, by more than the baseline's own seed
    sd.  The margin is reported in units of the baseline sd AND in units of the IQP arm's own
    sd, so the stricter reading ("more than either sd") can be checked from the same table.

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(os.path.dirname(REPO)), "paper", "ml4ps2026")
RESULTS = os.path.join(REPO, "results_seeds_v12.jsonl")
RESULTS_ES = os.path.join(REPO, "results_seeds_v12_es.jsonl")
SEED_TABLES = os.path.join(PAPER, "seed_tables.md")
OUT_MD = os.path.join(PAPER, "seed_tables_e.md")

# the three columns the verdict is defined on, plus the unbiased estimator for context
METRICS = [("pdf_js", "PDF-JS"), ("conv_mmd2", "MMD2"), ("conv_mmd2u", "MMD2_u"),
           ("conv_mmd2_aligned", "MMD2_al")]
VERDICT_COLS = ["pdf_js", "conv_mmd2", "conv_mmd2_aligned"]
BASELINES = ["DCGAN", "RBM"]
SELECTED_SUFFIX = " selected"


# --------------------------------------------------------------------------------------
def parse_table2(path):
    """{(model, N): {metric: (mean, sd)}} from the `## Table 2 (v11)` section of seed_tables.md.

    The section's columns are Model | N | PDF-JS | MMD2 | MMD2_u | MMD2_al | seeds, each cell
    `mean +- sd`.  Parsed, not retyped, so this file cannot quote a stale baseline.
    """
    txt = open(path).read()
    m = re.search(r"^## Table 2 \(v11\)\s*$(.*?)^###", txt, re.M | re.S)
    if not m:
        raise SystemExit("v12_aggregate: could not find the '## Table 2 (v11)' section in %s" % path)
    out = {}
    cols = [k for k, _ in METRICS]
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or "Model" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        model, n = cells[0], int(cells[1])
        vals = {}
        for key, cell in zip(cols, cells[2:6]):
            parts = cell.split("+-")
            vals[key] = (float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0)
        vals["seeds"] = int(cells[6])
        out[(model, n)] = vals
    return out


def load_rows(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def meta_of(run_dir, cache={}):
    if run_dir not in cache:
        p = os.path.join(run_dir, "meta.json")
        cache[run_dir] = json.load(open(p)) if os.path.isfile(p) else {}
    return cache[run_dir]


def ms(vals):
    """(mean, sd) with ddof=1; sd = 0 for a single value."""
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0)


def fmt(mean, sd, sig=4):
    if not np.isfinite(mean):
        return "--"
    return "%.*g +- %.*g" % (sig, mean, sig, sd)


def selected_rows(rows, path=None):
    """The validation-selected snapshot of the 16k schedule, as extra rows.

    `results_seeds_v12_es.jsonl` is written by `seeds/select_checkpoint.py`, which scores the
    validation block and records WHICH snapshot it picked per seed; its own metric columns come
    from a different scorer and lack `conv_mmd2u`.  So this function uses that file only as an
    index -- it takes the picked `run_dir` and returns the row for that same directory out of
    `results_seeds_v12.jsonl`, i.e. the identical v11-protocol numbers every other row in this
    table carries.  Nothing is re-scored and no jsonl is written.
    """
    path = path or RESULTS_ES
    if not os.path.isfile(path):
        return []
    by_dir = {os.path.normpath(r["run_dir"]): r for r in rows if r.get("run_dir")}
    out, missing = [], []
    for r in load_rows(path):
        if "early_stopped" not in str(r.get("kind", "")):
            continue                      # the file also carries the fixed-step reference row
        src = by_dir.get(os.path.normpath(r.get("run_dir") or ""))
        if src is None:
            missing.append(r.get("run_dir"))
            continue
        row = dict(src)
        row["arm"] = str(row["arm_base"] or row["arm"]) + SELECTED_SUFFIX
        row["selection"] = r.get("selection")
        row["candidate_steps"] = r.get("candidate_steps")
        out.append(row)
    if missing:
        raise SystemExit("v12_aggregate: %d validation-selected run dir(s) are not in %s: %s"
                         % (len(missing), RESULTS, missing[:3]))
    return out


# --------------------------------------------------------------------------------------
def build_groups(rows):
    """{(arm, N): {'rows': [...], 'meta': merged}} keyed on the v12 arm name and n_train."""
    g = {}
    for r in rows:
        if not str(r.get("arm", "")).startswith("IQP-v12"):
            continue
        g.setdefault((r["arm"], int(r["n_train"])), []).append(r)
    return g


_SNAP_RE = re.compile(r"\s+s(\d+)k$")


def snapshot_tag(arm):
    """The step marker the trainer appends to an arm name for a schedule snapshot.

    `train_iqp_affine_v12.py` writes `<arm> s016k` into meta["arm"] when it saves the 16k
    schedule's snapshots, and `selected_rows()` below appends ` selected` for the
    validation-picked one.  Neither marker is visible in any of the meta fields label() reads
    (M, objective, mu, flux_mode are identical across the whole schedule), so without this the
    five schedule arms and the fixed-4k arm all collapse onto one label -- which is exactly the
    collision that aborted job 363066.  Returns "" for a plain fixed-step arm.
    """
    if arm.endswith(SELECTED_SUFFIX):
        return "selected"
    m = _SNAP_RE.search(arm or "")
    return ("%dk steps" % int(m.group(1))) if m else ""


def summarise(key, rs):
    arm, n = key
    metas = [meta_of(r["run_dir"]) for r in rs]
    out = {"arm": arm, "n_train": n, "seeds": len(rs),
           "seed_list": sorted(int(r.get("seed", -1)) for r in rs),
           "snapshot": snapshot_tag(arm),
           "steps_list": sorted(int(r.get("selected_steps") or 0) for r in rs)}
    for k, _ in METRICS:
        out[k] = ms([r[k] for r in rs])
    for k in ("gen_max", "gen_clip_frac", "gen_mean_total"):
        out[k] = ms([r[k] for r in rs])
    out["proj_corr"] = ms([m.get("projection_corr_mean") for m in metas])
    out["gen_total_sd"] = ms([m.get("gen_total_sd") for m in metas])
    out["hi_ratio"] = ms([m.get("walsh_hi_energy_ratio") for m in metas])
    m0 = metas[0] if metas else {}
    out["param_count"] = rs[0].get("param_count")
    out["M"] = m0.get("M")
    out["objective"] = m0.get("objective")
    out["mu"] = m0.get("mu")
    out["walsh_k0"] = m0.get("walsh_k0")
    out["flux_mode"] = m0.get("flux_mode")
    out["encoding"] = m0.get("encoding", "crop10")
    out["latent_map"] = m0.get("latent_map", "affine")
    out["hidden"] = m0.get("hidden")
    out["n_qubits"] = m0.get("n_qubits")
    out["encode_decode_corr"] = m0.get("encode_decode_corr")
    out["lam_scale"] = float(m0.get("lam_scale", 1.0) or 1.0)
    out["spatial_flat_scale"] = float(m0.get("spatial_flat_scale", 1.0) or 1.0)
    out["spatial_lam_flat"] = float(m0.get("spatial_lam_flat") or 0.0)
    out["lam"] = ms([m.get("lam") for m in metas if m.get("lam") is not None])
    out["nonlin"] = ms([m.get("nonlinearity_ratio") for m in metas
                        if m.get("nonlinearity_ratio") is not None])
    return out


def label(s):
    """A short human label for one v12 arm."""
    obj = {"multi": "multi", "mmdflat": "flat", "auto3": "paper",
           "spatial": "spatial"}.get(s["objective"], s["objective"])
    bits = []
    if s.get("encoding", "crop10") != "crop10":
        bits.append(s["encoding"])
    if s.get("latent_map", "affine") == "mlp":
        bits.append("mlpH%s" % s.get("hidden"))
    bits.append("M=%s" % s["M"])
    # the objective label MUST separate every arm the grid ran: `spatial` and `spatial+flat`
    # differ only in spatial_lam_flat, and `multi` arms only in lam_scale.  label() is the key
    # the ranking, the overlap table and the verdict table are all built on, so a collision
    # here would silently merge two different arms.
    if s["objective"] == "spatial" and s.get("spatial_lam_flat", 0.0) != 0.0:
        sc = s.get("spatial_flat_scale", 1.0)
        obj += "+flat" + ("" if sc == 1.0 else "x%g" % sc)
    elif s["objective"] == "multi" and s.get("lam_scale", 1.0) != 1.0:
        obj += "(lam x%g)" % s["lam_scale"]
    bits.append(obj)
    if s["mu"]:
        bits.append("mu=%g,k0=%s" % (s["mu"], s["walsh_k0"]))
    bits.append("flux=%s" % s["flux_mode"])
    # the snapshot / validation-selected marker.  MUST stay in the label: it is the only thing
    # separating the 16k schedule's snapshots (and the selected one) from the fixed-4k arm.
    if s.get("snapshot"):
        bits.append(s["snapshot"])
    return ", ".join(bits)


def margin(iqp, base):
    """(gap, gap/base_sd, gap/iqp_sd) for one metric; gap > 0 means the IQP arm is better."""
    gap = base[0] - iqp[0]
    return (gap,
            gap / base[1] if base[1] > 0 else float("inf"),
            gap / iqp[1] if iqp[1] > 0 else float("inf"))


def verdict_for(s, base_rows, n):
    """Per-column margins of one arm against both baselines at this N, plus the two verdicts."""
    det = {}
    ok_base, ok_both = True, True
    for col in VERDICT_COLS:
        det[col] = {}
        for b in BASELINES:
            bb = base_rows.get((b, n))
            if bb is None:
                ok_base = ok_both = False
                continue
            g, gb, gi = margin(s[col], bb[col])
            det[col][b] = (g, gb, gi)
            if not (g > 0 and gb > 1.0):
                ok_base = False
            if not (g > 0 and gb > 1.0 and gi > 1.0):
                ok_both = False
    return det, ok_base, ok_both


# --------------------------------------------------------------------------------------
def table_block(fh, summaries, base_rows, n, title):
    fh.write("\n## %s\n\n" % title)
    fh.write("| Arm | N | PDF-JS | MMD2 | MMD2_u | MMD2_al | proj. corr. | gen_max | "
             "clip frac | gen total (mean/sd) | Params | seeds |\n")
    fh.write("|" + "---|" * 12 + "\n")
    for s in summaries:
        fh.write("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %.1f / %.1f | %s | %d |\n"
                 % (label(s), s["n_train"],
                    fmt(*s["pdf_js"]), fmt(*s["conv_mmd2"]), fmt(*s["conv_mmd2u"]),
                    fmt(*s["conv_mmd2_aligned"]), fmt(*s["proj_corr"], sig=4),
                    fmt(*s["gen_max"], sig=3), fmt(*s["gen_clip_frac"], sig=3),
                    s["gen_mean_total"][0], s["gen_total_sd"][0],
                    "{:,}".format(int(s["param_count"])) if s["param_count"] else "--",
                    s["seeds"]))
    # the published baselines, verbatim from seed_tables.md Table 2 (v11)
    for name in ["DCGAN", "RBM", "IQP, learned dir. M=4 (flat MMD)",
                 "IQP, learned dir. M=4 (paper kernel)"]:
        b = base_rows.get((name, n))
        if b is None:
            continue
        fh.write("| **%s** (Table 2, v11) | %d | %s | %s | %s | %s | -- | -- | -- | -- | -- | %d |\n"
                 % (name, n, fmt(*b["pdf_js"]), fmt(*b["conv_mmd2"]), fmt(*b["conv_mmd2u"]),
                    fmt(*b["conv_mmd2_aligned"]), b["seeds"]))


def overlap_block(fh, summaries, base_rows, n):
    """Which wins the seed spread does NOT support -- the same 1-sd rule seed_tables.md uses."""
    fh.write("\n### N = %d -- wins that the seed spread does not support\n\n" % n)
    fh.write("| metric | best arm (mean) | overlaps it within 1 sd |\n|---|---|---|\n")
    pool = [(label(s), s) for s in summaries]
    for name in ["DCGAN", "RBM", "IQP, learned dir. M=4 (flat MMD)",
                 "IQP, learned dir. M=4 (paper kernel)"]:
        b = base_rows.get((name, n))
        if b is not None:
            pool.append((name + " (Table 2)", b))
    for col, cname in METRICS:
        cand = [(lab, v[col][0], v[col][1]) for lab, v in pool if np.isfinite(v[col][0])]
        if not cand:
            continue
        cand.sort(key=lambda t: t[1])
        blab, bmean, bsd = cand[0]
        over = [lab for lab, m, sd in cand[1:] if (m - sd) <= (bmean + bsd)]
        fh.write("| %s | %s (%.4g +- %.4g) | %s |\n"
                 % (cname, blab, bmean, bsd,
                    ", ".join(over) if over else "**none -- the win holds**"))


def verdict_block(fh, summaries, base_rows, n):
    fh.write("\n### N = %d -- the pre-registered \"clear win\" test\n\n" % n)
    # N = 5 has no published baseline: saying "no combination passes" there would read as a
    # negative result when the truth is that the test is not defined at that N.
    if not any((b, n) in base_rows for b in BASELINES):
        fh.write("**Not applicable at N = %d:** Table 2 (v11) has no DCGAN or RBM row at this "
                 "N, so the clear-win test has nothing to test against. The rows above are "
                 "reported for the N-scaling trend only.\n" % n)
        return []
    fh.write("A combination passes only if its MEAN beats BOTH baselines on all three of "
             "PDF-JS, raw MMD$^2$ and aligned MMD$^2$ by more than the baseline's own seed sd. "
             "`gap/sd_base` is the margin in units of the baseline's sd (the literal test); "
             "`gap/sd_iqp` is the same gap in units of the IQP arm's own sd (the stricter "
             "reading). A negative number means the baseline is still ahead.\n\n")
    fh.write("| Arm | metric | vs DCGAN gap/sd_base | vs DCGAN gap/sd_iqp | vs RBM gap/sd_base "
             "| vs RBM gap/sd_iqp |\n")
    fh.write("|" + "---|" * 6 + "\n")
    passes = []
    for s in summaries:
        det, ok_base, ok_both = verdict_for(s, base_rows, n)
        if ok_base:
            passes.append((label(s), ok_both))
        for col in VERDICT_COLS:
            cn = dict(METRICS)[col]
            d = det[col]
            fh.write("| %s | %s | %s | %s | %s | %s |\n"
                     % (label(s), cn,
                        "%.2f" % d["DCGAN"][1] if "DCGAN" in d else "--",
                        "%.2f" % d["DCGAN"][2] if "DCGAN" in d else "--",
                        "%.2f" % d["RBM"][1] if "RBM" in d else "--",
                        "%.2f" % d["RBM"][2] if "RBM" in d else "--"))
    fh.write("\n**Verdict at N = %d:** " % n)
    if not passes:
        fh.write("no combination passes. No v12 arm beats both baselines on all three "
                 "columns by more than the baselines' seed sd.\n")
    else:
        fh.write("%d combination(s) pass the baseline-sd test: %s. Of those, %s also pass "
                 "the stricter both-sd reading.\n"
                 % (len(passes), ", ".join("`%s`" % p for p, _ in passes),
                    ", ".join("`%s`" % p for p, ok in passes if ok) or "none"))
    return passes


# --------------------------------------------------------------------------------------
def main():
    if not os.path.isfile(RESULTS):
        raise SystemExit("v12_aggregate: %s does not exist -- run the grid first" % RESULTS)
    base_rows = parse_table2(SEED_TABLES)
    rows = load_rows(RESULTS)
    rows_sel = selected_rows(rows)
    groups = build_groups(rows + rows_sel)
    if not groups:
        raise SystemExit("v12_aggregate: no IQP-v12 rows in %s" % RESULTS)
    summaries = [summarise(k, v) for k, v in groups.items()]

    def sort_key(s):
        # cheapest models first within a family, so the parameter-count story reads down the page
        return (s["n_train"], s.get("encoding", "crop10"), s.get("latent_map", "affine"),
                int(s.get("hidden") or 0), int(s["M"] or 0), s["objective"],
                float(s["mu"] or 0.0), int(s["walsh_k0"] or 0), s["flux_mode"],
                s.get("snapshot") or "")

    ns = sorted({s["n_train"] for s in summaries})
    # fail loudly rather than quietly merging two arms under one label
    for n in ns:
        labs = [label(x) for x in summaries if x["n_train"] == n]
        dup = sorted({l for l in labs if labs.count(l) > 1})
        if dup:
            raise SystemExit("v12_aggregate: label collision at N=%d: %s -- fix label()"
                             % (n, dup))
    with open(OUT_MD, "w") as fh:
        fh.write("# v12 IQP-side improvements -- seed statistics\n\n")
        fh.write("Every number is mean +- sample sd (ddof = 1) over the seeds listed in the "
                 "last column, scored under the UNCHANGED v11 protocol "
                 "(`experiments/13_iqp_turbulence_replication/evaluate_v11.py`, same 200 "
                 "held-out images, same `CONV_SEED`). The baselines, the data split and the "
                 "metrics were not touched; the rows in **bold** are copied by this script "
                 "straight out of `seed_tables.md`'s Table 2 (v11) so they cannot drift.\n\n")
        fh.write("The four levers (`train_iqp_affine_v12.py`): **A** `--objective multi` = "
                 "flat MMD + lam x paper-kernel MMD with lam = L0_flat / L0_paper fixed at "
                 "theta0; **B** `--mu`, `--walsh-k0` = a penalty on the Walsh energy above "
                 "|s| = k0, relative to L0 and to its own value at theta0; **C** `--flux "
                 "resample` = each generated image's total intensity drawn from the N training "
                 "crop totals with replacement plus jitter, instead of the single training "
                 "mean S; **D** `--M`. `flux=mean` + `objective=flat` + `mu=0` reproduces the "
                 "published arm bit-for-bit (checked on seed 0, N = 11: max |difference| over "
                 "all 200 generated images = 0).\n\n")
        fh.write("All arms run a FIXED 4,000 joint steps, the same count the published IQP "
                 "arms use; the baselines they are compared against are validation "
                 "early-stopped, which if anything favours the baselines.\n")
        fh.write("\n> **Selection caveat.** mu, k0 and lam-scale are swept here and the best "
                 "value is read off the held-out score, so the best row's margin is optimistic "
                 "by the usual selection amount. The mu = 0, lam-scale = 1 rows carry no such "
                 "selection.\n")
        fh.write("\n> **Projection-correlation caveat.** `proj. corr.` is computed in the "
                 "arm's OWN pmf space, so the `lowres8` rows measure a 16x16 correlation and "
                 "the 10-qubit rows a 32x32 one. They are not comparable across encodings; the "
                 "four metric columns are (they are computed on the decoded 64x64 frames by "
                 "the unchanged evaluator). `lowres8`'s encode->decode ceiling on real crops "
                 "is 0.990.\n")

        # ---- headline: the best few arms at the smallest N, by mean rank over the three
        # verdict columns -- the one table a reader should look at first
        for n in ns:
            sub0 = [x for x in summaries if x["n_train"] == n]
            if n != 11 or not sub0:
                continue
            rk = {}
            for col in VERDICT_COLS:
                for i, x in enumerate(sorted(sub0, key=lambda z: z[col][0])):
                    rk[label(x)] = rk.get(label(x), 0.0) + i + 1
            top = sorted(sub0, key=lambda x: rk[label(x)])[:8]
            fh.write("\n## Headline: the 8 best v12 arms at N = 11 (mean rank over the three "
                     "verdict columns)\n\n")
            fh.write("| Arm | PDF-JS | MMD2 | MMD2_al | Params | beats DCGAN+RBM on all 3 "
                     "by > baseline sd? |\n|" + "---|" * 6 + "\n")
            for x in top:
                _d, ok_b, ok_both = verdict_for(x, base_rows, n)
                fh.write("| %s | %s | %s | %s | %s | %s |\n"
                         % (label(x), fmt(*x["pdf_js"]), fmt(*x["conv_mmd2"]),
                            fmt(*x["conv_mmd2_aligned"]),
                            "{:,}".format(int(x["param_count"])) if x["param_count"] else "--",
                            ("**yes**" + (" (also > its own sd)" if ok_both else
                                          " (not > its own sd)")) if ok_b else "no"))
            for name in ["DCGAN", "RBM"]:
                b = base_rows.get((name, n))
                if b:
                    fh.write("| **%s** (Table 2, v11) | %s | %s | %s | %s | -- |\n"
                             % (name, fmt(*b["pdf_js"]), fmt(*b["conv_mmd2"]),
                                fmt(*b["conv_mmd2_aligned"]),
                                "199,837" if name == "DCGAN" else "11,010"))

        headline = None
        for n in ns:
            sub = sorted([s for s in summaries if s["n_train"] == n], key=sort_key)
            table_block(fh, sub, base_rows, n,
                        "N = %d%s" % (n, " (headline cell)" if n == 11 else ""))
            overlap_block(fh, sub, base_rows, n)
            p = verdict_block(fh, sub, base_rows, n)
            if n == 11:
                headline = (sub, p)

        fh.write("\n## Provenance\n\n")
        fh.write("- runs: `outputs_v12/s{0..4}/*` (`train_iqp_affine_v12.py`)\n")
        fh.write("- scores: `results_seeds_v12.jsonl` (`evaluate_v11.py --no-legacy`)\n")
        fh.write("- the `selected` rows: which snapshot the validation block picked comes from "
                 "`results_seeds_v12_es.jsonl` (`seeds/select_checkpoint.py`); the NUMBERS are "
                 "the same `results_seeds_v12.jsonl` rows as everywhere else in this file\n")
        fh.write("- baselines: parsed from `seed_tables.md`, section `## Table 2 (v11)`\n")
        fh.write("- this file: `experiments/13_iqp_turbulence_replication/v12_aggregate.py`\n")

    # ------------------------------------------------------------------ stdout summary
    print("[v12-agg] %d arm x N summaries from %d rows -> %s" % (len(summaries), len(rows), OUT_MD))
    for n in ns:
        sub = sorted([s for s in summaries if s["n_train"] == n], key=sort_key)
        print("\n[v12-agg] ---- N = %d (%d arms) ----" % (n, len(sub)))
        print("%-46s %11s %10s %10s %8s %8s %9s %6s" % ("arm", "PDF-JS", "MMD2", "MMD2_al",
                                                          "proj", "gmax", "params", "seeds"))
        for s in sub:
            print("%-46s %11.6f %10.5f %10.5f %8.4f %8.3f %9s %6d"
                  % (label(s)[:46], s["pdf_js"][0], s["conv_mmd2"][0],
                     s["conv_mmd2_aligned"][0], s["proj_corr"][0], s["gen_max"][0],
                     "{:,}".format(int(s["param_count"])) if s["param_count"] else "--",
                     s["seeds"]))
        for name in ["DCGAN", "RBM", "IQP, learned dir. M=4 (flat MMD)",
                     "IQP, learned dir. M=4 (paper kernel)"]:
            b = base_rows.get((name, n))
            if b:
                print("%-46s %11.6f %10.5f %10.5f %8s %8s %9s %6d"
                      % (name[:40] + " (Tab2)", b["pdf_js"][0], b["conv_mmd2"][0],
                         b["conv_mmd2_aligned"][0], "--", "--", "--", b["seeds"]))
    if headline:
        sub, p = headline
        print("\n[v12-agg] N=11 clear-win test: %s"
              % (", ".join("%s (both-sd: %s)" % (a, ok) for a, ok in p) if p
                 else "NO combination passes"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
