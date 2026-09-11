#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate every Appendix-B (turbulence-replication) quantity to mean +/- sd over 5 seeds.

Scope: ONLY the reference paper's own turbulence dataset (D2).  The main-paper lens tables
are a separate track with its own aggregator; nothing here reads or writes those files.

Reads (never writes) the existing 3-seed results plus the new seed-3/4 results, and emits

    results_d2_seeds.jsonl              one flat record per (seed, arm, aggregation, quantity)
    paper/ml4ps2026/seed_tables_c.md    the Appendix-B seed tables

Pitfall handled explicitly (docs/IQP_TURBULENCE_REPLICATION_RESULTS.md, evaluate.py summary):
evaluate.py's printed table groups by meta["arm"], so the six single-seed `iqp_sens_*`
sensitivity runs and the single-seed `dcgan_scaling_*` runs are folded into the IQP-31k /
DCGAN rows.  This script groups by RUN-DIR BASENAME and keeps only the primary multi-seed
arms, so those runs can never contaminate a seed statistic.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import json
import math
import os
import re

REPO = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(REPO, "..", "..", "paper", "ml4ps2026"))
OUT_JSONL = os.path.join(REPO, "results_d2_seeds.jsonl")
OUT_MD = os.path.join(PAPER, "seed_tables_c.md")

SEEDS_ALL = [0, 1, 2, 3, 4]

_PRIMARY = re.compile(
    r"^(?P<base>iqp_K6|iqp_K7|rbm|ctrl_marginal|"
    r"dcgan_51k_n11|dcgan_51k_n100|dcgan_0\.2m_n11|dcgan_0\.2m_n100|"
    r"dcgan_1\.2m_n11|dcgan_1\.2m_n100)_seed(?P<seed>\d+)(?P<suffix>__last)?$"
)
_DETERMINISTIC_ARMS = {"ctrl_interp": "CTRL-interp"}

LABEL = {
    "iqp_K6": ("IQP-31k", 31180, 11),
    "iqp_K7": ("IQP-63k", 63004, 11),
    "rbm": ("RBM-12k", 11988, 11),
    "ctrl_marginal": ("CTRL-marginal", 198, 11),
    "dcgan_51k_n11": ("DCGAN-51k", 55339, 11),
    "dcgan_51k_n100": ("DCGAN-51k", 55339, 100),
    "dcgan_0.2m_n11": ("DCGAN-0.2M", 202303, 11),
    "dcgan_0.2m_n100": ("DCGAN-0.2M", 202303, 100),
    "dcgan_1.2m_n11": ("DCGAN-1.2M", 1213889, 11),
    "dcgan_1.2m_n100": ("DCGAN-1.2M", 1213889, 100),
}
MAIN_ORDER = ["iqp_K6", "iqp_K7", "rbm", "dcgan_1.2m_n11", "dcgan_0.2m_n11",
              "dcgan_51k_n11", "ctrl_marginal",
              "dcgan_1.2m_n100", "dcgan_0.2m_n100", "dcgan_51k_n100"]
LAST_ORDER = ["iqp_K6", "iqp_K7", "rbm", "ctrl_marginal"]

AF_CONFIGS = ["auto3", "sigma0.3", "sigma0.15", "flat", "flat_lr1e-3",
              "flat_K7", "sigma0.3_K7_lr1e-3", "auto3_K7"]
AFP_CONFIGS = ["paper_stage1", "paper_stage1_plus_stage2",
               "allparams_subsample5e4", "paper_stage1_subsample5e4"]

# The 3-seed / single-seed values Appendix B cites today, for the drift check.
PUBLISHED = {
    ("main", "iqp_K6", "pdf_js"): 0.2700, ("main", "iqp_K6", "conv_mmd2"): 0.8237,
    ("main", "iqp_K7", "pdf_js"): 0.2366, ("main", "iqp_K7", "conv_mmd2"): 0.8161,
    ("main", "rbm", "pdf_js"): 0.2139, ("main", "rbm", "conv_mmd2"): 0.7282,
    ("main", "dcgan_1.2m_n11", "pdf_js"): 0.05474, ("main", "dcgan_1.2m_n11", "conv_mmd2"): 0.3819,
    ("last", "iqp_K6", "pdf_js"): 0.0322, ("last", "iqp_K6", "conv_mmd2"): 0.7096,
    ("last", "iqp_K7", "pdf_js"): 0.0476, ("last", "iqp_K7", "conv_mmd2"): 0.7390,
    ("last", "rbm", "pdf_js"): 0.02246, ("last", "rbm", "conv_mmd2"): 0.6384,
}

EMD = "—"      # em dash
PM = "±"
SQ = "²"


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mean_sd(vals):
    """(mean, sample sd with ddof=1, n)."""
    vals = [float(v) for v in vals
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var), n


def sd0(s, n):
    """The ddof=0 sd that numpy.std (and hence evaluate.py's printed summary) reports."""
    if s is None or n is None or n < 2:
        return None
    return s * math.sqrt((n - 1) / float(n))


def fmt(m, s, n, digits=4):
    """Render mean +/- sd.  A spread of exactly zero over >=2 seeds is not a small sd, it
    means the quantity does not depend on the seed at all (e.g. anchor_fit_experiment's corr:
    full-pool pmf + data-dependent init leaves nothing for the seed to change), so it is
    labelled rather than printed as '+/- 0.0000'."""
    if m is None:
        return "n/a"
    if n <= 1:
        return "%.*f (n=1)" % (digits, m)
    if s is not None and s < 1e-12:
        return "%.*f (seed-independent, n=%d)" % (digits, m, n)
    return "%.*f %s %.*f" % (digits, m, PM, digits, s)


def fmt0(s, n, digits=4):
    v = sd0(s, n)
    return EMD if v is None else "%.*f" % (digits, v)


def sig(x, digits=4):
    return "n/a" if x is None else "%.*f" % (digits, x)


# --------------------------------------------------------------------------------------
def collect_table2():
    sources = {
        "main": [os.path.join(REPO, "results_d2_main.jsonl"),
                 os.path.join(REPO, "results_d2_seeds_main.jsonl")],
        "last": [os.path.join(REPO, "results_d2_last.jsonl"),
                 os.path.join(REPO, "results_d2_seeds_last.jsonl")],
    }
    table, determ = {}, {}
    for agg, paths in sources.items():
        for p in paths:
            for row in read_jsonl(p):
                base = os.path.basename(os.path.normpath(row.get("run_dir", "")))
                if base in _DETERMINISTIC_ARMS:
                    row = dict(row, _source=os.path.basename(p))
                    determ[(agg, base)] = row
                    continue
                m = _PRIMARY.match(base)
                if not m:
                    continue
                if (agg == "last") != bool(m.group("suffix")):
                    continue
                table.setdefault((agg, m.group("base")), {})[int(m.group("seed"))] = dict(
                    pdf_js=row.get("pdf_js"), conv_mmd2=row.get("conv_mmd2"),
                    run_dir=row.get("run_dir"), param_count=row.get("param_count"),
                    train_snapshots=row.get("train_snapshots"), source=os.path.basename(p))
    return table, determ


def collect_anchor_corr(extras):
    out = {}
    for r in extras:
        if r.get("task") != "anchor_corr":
            continue
        m = re.match(r"^(iqp_K[67])_seed(\d+)$", r["arm"])
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = r
    return out


def _af_root(kind, seed):
    if kind == "experiment":
        return os.path.join(REPO, "outputs_anchor_fit" if seed == 0
                            else "outputs_anchor_fit_s%d" % seed)
    return os.path.join(REPO, "outputs_anchor_fit_protocol" if seed == 0
                        else "outputs_anchor_fit_protocol_s%d" % seed)


def collect_anchor_fit(kind, configs):
    out = {}
    for seed in SEEDS_ALL:
        root = _af_root(kind, seed)
        for cfg in configs:
            meta_path = os.path.join(root, cfg, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as fh:
                meta = json.load(fh)
            fin = meta.get("final") or {}
            out.setdefault(cfg, {})[seed] = dict(
                corr=fin.get("corr"), pdf_js=fin.get("pdf_js"), mmd2=fin.get("mmd2"),
                step=fin.get("step"), meta_path=meta_path,
                corr_after_stage2=meta.get("corr_after_stage2"))
    return out


# --------------------------------------------------------------------------------------
def main():
    table, determ = collect_table2()
    extras = read_jsonl(os.path.join(REPO, "results_d2_seeds_extras.jsonl"))
    anchor = collect_anchor_corr(extras)
    af = collect_anchor_fit("experiment", AF_CONFIGS)
    afp = collect_anchor_fit("protocol", AFP_CONFIGS)
    # (5) `turb_seed_extras.py` opens --results in append mode, so a manual re-run of
    # `--task calib` against the same file would otherwise be counted twice (table2 and
    # anchor_corr are dict-keyed by seed and immune; calib is a flat list).  Keep the last
    # record per (reference, shuffle_seed).
    _calib = {}
    for r in extras:
        if r.get("task") == "calib":
            _calib[(r.get("reference"), r.get("shuffle_seed"))] = r
    calib = list(_calib.values())

    recs = []
    for (agg, base), per_seed in sorted(table.items()):
        label, params, n_train = LABEL[base]
        for seed, v in sorted(per_seed.items()):
            recs.append(dict(quantity="table2", aggregation=agg, arm=label, arm_key=base,
                             seed=seed, param_count=params, train_snapshots=n_train,
                             pdf_js=v["pdf_js"], conv_mmd2=v["conv_mmd2"],
                             run_dir=v["run_dir"], source=v["source"], deterministic=False))
    for (agg, base), row in sorted(determ.items()):
        recs.append(dict(quantity="table2", aggregation=agg, arm=_DETERMINISTIC_ARMS[base],
                         arm_key=base, seed=None, param_count=row.get("param_count"),
                         train_snapshots=row.get("train_snapshots"),
                         pdf_js=row.get("pdf_js"), conv_mmd2=row.get("conv_mmd2"),
                         run_dir=row.get("run_dir"), source=row.get("_source"),
                         deterministic=True))
    for base, per_seed in sorted(anchor.items()):
        for seed, r in sorted(per_seed.items()):
            recs.append(dict(quantity="anchor_corr", arm=LABEL[base][0], arm_key=base,
                             seed=seed, anchor_corr_sampled=r["anchor_corr_sampled_mean"],
                             anchor_corr_exact=r["anchor_corr_exact_mean"],
                             anchor_corr_sampled_t1=r["anchor_corr_sampled_t1"],
                             anchor_corr_exact_t1=r["anchor_corr_exact_t1"],
                             run_dir=r["run_dir"], deterministic=False))
    for cfg, per_seed in sorted(af.items()):
        for seed, r in sorted(per_seed.items()):
            recs.append(dict(quantity="anchor_fit_ceiling", config=cfg, seed=seed,
                             corr=r["corr"], pdf_js=r["pdf_js"], mmd2=r["mmd2"],
                             meta_path=r["meta_path"], deterministic=False))
    for cfg, per_seed in sorted(afp.items()):
        for seed, r in sorted(per_seed.items()):
            recs.append(dict(quantity="anchor_fit_protocol", config=cfg, seed=seed,
                             corr=r["corr"], pdf_js=r["pdf_js"], mmd2=r["mmd2"],
                             corr_after_stage2=r["corr_after_stage2"],
                             meta_path=r["meta_path"], deterministic=False))
    for r in calib:
        recs.append(dict(quantity="metric_calibration", reference=r["reference"],
                         shuffle_seed=r.get("shuffle_seed"), pdf_js=r.get("pdf_js"),
                         conv_mmd2=r.get("conv_mmd2"), corr_mean=r.get("corr_mean"),
                         corr_min=r.get("corr_min"), deterministic=r.get("deterministic")))

    with open(OUT_JSONL, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print("[aggregate] wrote %d records to %s" % (len(recs), OUT_JSONL))

    L = []
    A = L.append
    A("# Appendix B seed tables (5 seeds)")
    A("")
    A("Auto-generated by `experiments/13_iqp_turbulence_replication/turb_aggregate_seeds.py`; "
      "per-seed records in `experiments/13_iqp_turbulence_replication/results_d2_seeds.jsonl`. "
      "Covers only Appendix B (the replication on the reference paper's own turbulence "
      "dataset); the main-paper lens tables are a separate track.")
    A("")
    A("Seeds 0-2 come from job 342220 (`results_d2_main.jsonl`) and the last-write job "
      "(`results_d2_last.jsonl`); seeds 3-4 from `results_d2_seeds_main.jsonl` and "
      "`results_d2_seeds_last.jsonl`. Quantities marked **deterministic** carry no random "
      "seed: re-running them cannot change the value, so no spread is reported.")
    A("")
    A("**sd convention.** The `mean %s sd` columns use the sample standard deviation "
      "(ddof = 1). The %s values Appendix B cites today came from `evaluate.py`'s printed "
      "summary, which uses `numpy.std` (ddof = 0), smaller by a factor sqrt((n-1)/n), i.e. "
      "0.894 at n = 5. Both are given for the two Table-2 blocks so the paper can pick one "
      "convention and apply it everywhere." % (PM, PM))
    A("")
    A("Grouping is by run-directory basename, not by `meta[\"arm\"]`: `evaluate.py`'s printed "
      "summary folds the six single-seed `iqp_sens_*` runs into the IQP-31k row and the "
      "single-seed `dcgan_scaling_*` runs into the DCGAN rows, which is why its "
      "`IQP-31k (11)` line reads 0.2486 %s 0.0299 rather than the 0.2700 of the three seeded "
      "runs." % PM)
    A("")

    A("## C1. Table 2 under the pre-registered (mean) aggregation, 11 training snapshots")
    A("")
    A("| Model | Params | N | PDF-JS (mean %s sd) | Conv2D-MMD%s (mean %s sd) | n_seeds | sd, ddof=0 (JS / MMD%s) | value cited in Appendix B |"
      % (PM, SQ, PM, SQ))
    A("|---|---|---|---|---|---|---|---|")
    for base in MAIN_ORDER:
        if ("main", base) not in table:
            continue
        label, params, n_train = LABEL[base]
        per = table[("main", base)]
        seeds = sorted(per)
        mj, sj, nj = mean_sd([per[s]["pdf_js"] for s in seeds])
        mm, sm, nm = mean_sd([per[s]["conv_mmd2"] for s in seeds])
        pj = PUBLISHED.get(("main", base, "pdf_js"))
        pm_ = PUBLISHED.get(("main", base, "conv_mmd2"))
        pub = EMD if pj is None else "%.4f / %.4f" % (pj, pm_)
        A("| %s | %s | %d | %s | %s | %d | %s / %s | %s |"
          % (label, "{:,}".format(params), n_train, fmt(mj, sj, nj), fmt(mm, sm, nm), nj,
             fmt0(sj, nj), fmt0(sm, nm), pub))
    for (agg, base), row in sorted(determ.items()):
        if agg != "main":
            continue
        A("| %s | 0 | 11 | %.5f (**deterministic**) | %.6f (**deterministic**) | %s | %s | same |"
          % (_DETERMINISTIC_ARMS[base], row.get("pdf_js"), row.get("conv_mmd2"), EMD, EMD))
    A("")
    A("Seeds present per arm: " + "; ".join(
        "%s(%d) %s" % (LABEL[b][0], LABEL[b][2], sorted(table[("main", b)]))
        for b in MAIN_ORDER if ("main", b) in table))
    A("")

    A("## C2. Table 2 under the last-write-wins aggregation")
    A("")
    A("Tuple-based arms only. DCGAN and CTRL-interp have no per-cell aggregation step, so "
      "their C1 values carry over unchanged.")
    A("")
    A("| Model | Params | PDF-JS (mean %s sd) | Conv2D-MMD%s (mean %s sd) | n_seeds | sd, ddof=0 (JS / MMD%s) | value cited in Appendix B |"
      % (PM, SQ, PM, SQ))
    A("|---|---|---|---|---|---|---|")
    for base in LAST_ORDER:
        if ("last", base) not in table:
            continue
        label, params, _ = LABEL[base]
        per = table[("last", base)]
        seeds = sorted(per)
        mj, sj, nj = mean_sd([per[s]["pdf_js"] for s in seeds])
        mm, sm, nm = mean_sd([per[s]["conv_mmd2"] for s in seeds])
        pj = PUBLISHED.get(("last", base, "pdf_js"))
        pm_ = PUBLISHED.get(("last", base, "conv_mmd2"))
        pub = EMD if pj is None else "%.4f / %.4f" % (pj, pm_)
        A("| %s | %s | %s | %s | %d | %s / %s | %s |"
          % (label, "{:,}".format(params), fmt(mj, sj, nj), fmt(mm, sm, nm), nj,
             fmt0(sj, nj), fmt0(sm, nm), pub))
    A("")
    A("Reference paper's own reported IQP PDF-JS on this dataset: 0.0238 and 0.0221 "
      "(**deterministic**, a citation). Its Table 2 Conv2D-MMD%s column spans 0.896-1.472 "
      "(**deterministic**, a citation)." % SQ)
    A("")

    A("## C3. Anchor-fit correlation of the trained IQP on its own 11 training anchors")
    A("")
    A("corr(generated field, ground-truth field) on the model's own training anchors, once "
      "through the sampled aggregation path (10^5 samples) and once through the exact "
      "conditional mean E[z | x, y]. Recomputed per seed by "
      "`turb_seed_extras.py --task anchor-corr`. Two reductions are given because Appendix B "
      "cites both: the mean over all 11 anchors, and anchor t = 1 alone (the anchor the "
      "protocol-attribution experiment of C4 uses).")
    A("")
    A("| Model | reduction | corr, sampled (mean %s sd) | corr, exact (mean %s sd) | n_seeds |" % (PM, PM))
    A("|---|---|---|---|---|")
    for base in ["iqp_K6", "iqp_K7"]:
        if base not in anchor:
            continue
        per = anchor[base]
        seeds = sorted(per)
        for red, ks, ke in (("mean of 11 anchors", "anchor_corr_sampled_mean", "anchor_corr_exact_mean"),
                            ("anchor t = 1", "anchor_corr_sampled_t1", "anchor_corr_exact_t1")):
            ms, ss, ns = mean_sd([per[s][ks] for s in seeds])
            me, se, ne = mean_sd([per[s][ke] for s in seeds])
            A("| %s | %s | %s | %s | %d |"
              % (LABEL[base][0], red, fmt(ms, ss, ns), fmt(me, se, ne), ns))
    A("")
    k6 = anchor.get("iqp_K6", {}).get(0)
    k7 = anchor.get("iqp_K7", {}).get(0)
    if k6 and k7:
        A("**Provenance correction.** Appendix B cites \"correlation 0.69 sampled, 0.77 exact\" "
          "for the IQP-31k main run, taken from `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md` "
          "4(2). Recomputing that pair per arm shows it is the **IQP-63k** seed-0 11-anchor "
          "mean (%.4f / %.4f), not IQP-31k's (%.4f / %.4f). The diagnostic script it came "
          "from, `diag_scripts/q2_anchor_fit.py`, loops `[(\"iqp_K7_seed0\", 7), "
          "(\"iqp_K6_seed0\", 6)]` and prints the K=7 block first. What *is* the IQP-31k main "
          "run is the anchor t = 1 exact correlation, %.4f, and that one is independently "
          "confirmed by `anchor_fit_protocol.py --summarize` in job 342685 (\"main run "
          "iqp_K6_seed0 (ref) 0.7738\")."
          % (k7["anchor_corr_sampled_mean"], k7["anchor_corr_exact_mean"],
             k6["anchor_corr_sampled_mean"], k6["anchor_corr_exact_mean"],
             k6["anchor_corr_exact_t1"]))
        A("")

    A("## C4. Model-family ceiling and protocol attribution")
    A("")
    A("`anchor_fit_experiment.py`: all parameters trainable, full-pool pmf, 10^5 steps, one "
      "config per kernel/K setting, scored on anchor t = 1. The ceiling is the best config of "
      "the sweep.")
    A("")
    _af_seeds = sorted({sd for c in af for sd in af[c]})
    A("**This sweep carries no seed.** Run at seeds %s it returns bit-identical `corr` "
      "for every config, because nothing in the optimisation consumes the RNG: the pmf is the "
      "anchor's full native code pool (no subsample draw), the initialisation is "
      "data-dependent (`train_iqp.data_init_theta`), and the `auto3` bandwidths are resolved "
      "from that same pool. The seed does reach the run, and does move `pdf_js`, which is "
      "scored on a sampled field. So the model-family ceiling is a deterministic quantity and "
      "the `0.88-0.89` the paper cites is a range across configs, not a seed spread."
      % ", ".join(str(x) for x in _af_seeds))
    A("")
    A("`sigma0.15` and `flat` are the same experiment: at sigma = 0.15 the Walsh weights "
      "collapse to the uniform `flat` weights to ten digits, so the two rows agree bit for "
      "bit by construction rather than by coincidence.")
    A("")
    A("| Config | corr (mean %s sd) | n_seeds | seed-0 value |" % PM)
    A("|---|---|---|---|")
    for cfg in AF_CONFIGS:
        if cfg not in af:
            continue
        per = af[cfg]
        m, s, n = mean_sd([per[x]["corr"] for x in sorted(per)])
        A("| `%s` | %s | %d | %s |" % (cfg, fmt(m, s, n), n, sig(per.get(0, {}).get("corr"))))
    # (3) a max over a PARTIAL config set is biased low, so a seed only contributes its
    # ceiling once all AF_CONFIGS landed for it.  Incomplete seeds are named, not averaged in.
    ceil_per_seed, ceil_partial = {}, {}
    for seed in SEEDS_ALL:
        vals = [af[c][seed]["corr"] for c in AF_CONFIGS
                if c in af and seed in af[c] and af[c][seed]["corr"] is not None]
        if len(vals) == len(AF_CONFIGS):
            ceil_per_seed[seed] = max(vals)
        elif vals:
            ceil_partial[seed] = len(vals)
    mc, sc, nc = mean_sd(list(ceil_per_seed.values()))
    A("| **best config of the sweep (ceiling)** | **%s** | %d | %s |"
      % (fmt(mc, sc, nc), nc, sig(ceil_per_seed.get(0))))
    if ceil_partial:
        A("")
        A("Seeds excluded from the ceiling row because not all %d configs landed: %s. A max "
          "over a partial sweep is biased low, so those seeds are named rather than averaged in."
          % (len(AF_CONFIGS),
             ", ".join("seed %d (%d/%d)" % (k, v, len(AF_CONFIGS)) for k, v in sorted(ceil_partial.items()))))
    A("")
    A("`anchor_fit_protocol.py`: same anchor, 3x10^4 steps, isolating the two ingredients of "
      "the reference paper's Algorithm 1 stage 1.")
    A("")
    A("| Config | corr (mean %s sd) | n_seeds | seed-0 value |" % PM)
    A("|---|---|---|---|")
    for cfg in AFP_CONFIGS:
        if cfg not in afp:
            continue
        per = afp[cfg]
        m, s, n = mean_sd([per[x]["corr"] for x in sorted(per)])
        A("| `%s` | %s | %d | %s |" % (cfg, fmt(m, s, n), n, sig(per.get(0, {}).get("corr"))))
    A("")

    def delta(cfg_a, cfg_b):
        vals = []
        for seed in SEEDS_ALL:
            a = afp.get(cfg_a, {}).get(seed, {}).get("corr")
            b = afp.get(cfg_b, {}).get(seed, {}).get("corr")
            if a is not None and b is not None:
                vals.append(b - a)
        return mean_sd(vals)

    A("| Attributed factor | delta corr (mean %s sd) | n_seeds | value cited in Appendix B |" % PM)
    A("|---|---|---|---|")
    m, s, n = delta("allparams_subsample5e4", "paper_stage1")
    A("| stage 1 freezes theta_lat (as attributed in the 3-seed write-up) | %s | %d | -0.08 |"
      % (fmt(m, s, n), n))
    m, s, n = delta("allparams_subsample5e4", "paper_stage1_subsample5e4")
    A("| stage 1 freezes theta_lat, *unconfounded* (both cells at subsample = 5x10^4) | %s | %d | not reported |"
      % (fmt(m, s, n), n))
    m, s, n = delta("paper_stage1", "paper_stage1_subsample5e4")
    A("| training pmf is a 5x10^4 subsample, not the full pool | %s | %d | -0.04 |" % (fmt(m, s, n), n))
    m, s, n = delta("paper_stage1", "paper_stage1_plus_stage2")
    A("| Algorithm 1 stage 2 (latent adaptation) | %s | %d | +0.0001 |" % (fmt(m, s, n), n))
    A("")
    d_conf = delta("allparams_subsample5e4", "paper_stage1")
    d_unconf = delta("allparams_subsample5e4", "paper_stage1_subsample5e4")
    d_sub = delta("paper_stage1", "paper_stage1_subsample5e4")
    A("Two rows are given for the freeze because the first is a confounded contrast: "
      "`allparams_subsample5e4` is (freeze = off, subsample = on) while `paper_stage1` is "
      "(freeze = on, subsample = off), so both factors move at once. Holding subsample = on in "
      "both cells isolates the freeze.")
    A("")
    A("**This block is where 5 seeds change the story.** At seed 0 alone the freeze read "
      "-0.078 and the pmf subsample read -0.038, which is the -0.08 / -0.04 split Appendix B "
      "cites. Over 5 seeds the freeze is %s and the subsample is %s, i.e. the subsample term "
      "is indistinguishable from zero and the freeze is correspondingly larger. The seed-0 "
      "split was an artefact of `paper_stage1` landing high at that seed (%.4f against a "
      "5-seed mean of %s): that cell freezes theta_lat at a random N(0, 0.01) draw, so it is "
      "the most seed-sensitive cell in the table. The confounded and unconfounded freeze "
      "contrasts now agree (%s vs %s) precisely because the factor separating them is null. "
      "The honest 5-seed statement is that stage 1 freezing theta_lat is essentially the whole "
      "protocol penalty, and the 5x10^4-sample pmf costs nothing measurable."
      % (fmt(*d_conf), fmt(*d_sub),
         afp.get("paper_stage1", {}).get(0, {}).get("corr") or float("nan"),
         fmt(*mean_sd([afp["paper_stage1"][x]["corr"] for x in sorted(afp.get("paper_stage1", {}))])),
         fmt(*d_conf), fmt(*d_unconf)))
    A("")
    A("The factors are still not fully crossed (there is no all-parameters + full-pool cell at "
      "3x10^4 steps), so these remain approximate attributions rather than a variance "
      "decomposition, the same caveat the earlier write-up carries.")
    A("")

    A("## C5. Metric-calibration references")
    A("")
    A("The Conv2D encoder seed is fixed at `evaluate.py`'s `CONV_SEED = 0` for every row, so "
      "these live in the same feature space as every arm above; only the shuffle draw varies.")
    A("")
    A("`linear_interp_of_bracketing_anchors` here is **not** the same object as C1's "
      "CTRL-interp arm, and the two are expected to disagree. This row interpolates the raw "
      "anchor fields, reproducing the 0.9998 correlation the paper cites (source: "
      "`diag_scripts/q4_task_triviality.py`). The CTRL-interp arm interpolates the "
      "*quantised* anchor fields, because every generative arm is capped at 6-bit z and the "
      "zero-parameter control is held to the same cap (`controls.run_interp`). That is why "
      "CTRL-interp scores PDF-JS 0.00119 in C1 while this row scores 0.00010: the quantisation "
      "is a deliberate handicap, not a discrepancy.")
    A("")
    A("| Reference | PDF-JS (mean %s sd) | Conv2D-MMD%s (mean %s sd) | corr | n_seeds | value cited in Appendix B |"
      % (PM, SQ, PM))
    A("|---|---|---|---|---|---|")
    by_ref = {}
    for r in calib:
        by_ref.setdefault(r["reference"], []).append(r)
    ref_order = ["gt_spatially_shuffled_px", "gt_shuffled_across_tau",
                 "linear_interp_of_bracketing_anchors", "nearest_anchor_copy",
                 "identical_to_gt"]
    cited = {"gt_spatially_shuffled_px": "0.704 (Conv2D-MMD%s)" % SQ,
             "linear_interp_of_bracketing_anchors": "0.9998 (corr)"}
    for ref in ref_order:
        rows = by_ref.get(ref)
        if not rows:
            continue
        det = bool(rows[0].get("deterministic"))
        mj, sj, nj = mean_sd([r["pdf_js"] for r in rows])
        mm, sm, nm = mean_sd([r["conv_mmd2"] for r in rows])
        cm = rows[0].get("corr_mean")
        corr_s = EMD if cm is None else ("%.5f (**deterministic**)" % cm)
        if det:
            A("| `%s` | %.5f (**deterministic**) | %.5f (**deterministic**) | %s | %s | %s |"
              % (ref, mj, mm, corr_s, EMD, cited.get(ref, EMD)))
        else:
            A("| `%s` | %s | %s | %s | %d | %s |"
              % (ref, fmt(mj, sj, nj, 5), fmt(mm, sm, nm, 5), corr_s, nj, cited.get(ref, EMD)))
    A("")

    A("## C5b. Does the Appendix-B verdict survive 5 seeds?")
    A("")
    A("Appendix B rests on two comparative claims. Both are re-evaluated here on the 5-seed "
      "means, with the seed spread shown so a claim resting inside the noise is visible as "
      "such.")
    A("")

    def _ms(agg, base, metric):
        per = table.get((agg, base))
        if not per:
            return None, None, 0
        return mean_sd([per[x][metric] for x in sorted(per)])

    A("| Claim | Metric | IQP-31k | challenger | verdict on 5-seed means |")
    A("|---|---|---|---|---|")
    for challenger, cl_label in (("rbm", "RBM-12k"), ("dcgan_1.2m_n11", "DCGAN-1.2M")):
        for metric, met_label in (("pdf_js", "PDF-JS"), ("conv_mmd2", "Conv2D-MMD" + SQ)):
            mi, si, ni = _ms("main", "iqp_K6", metric)
            mc, sc, nc = _ms("main", challenger, metric)
            if mi is None or mc is None:
                continue
            wins = mc < mi
            overlap = abs(mi - mc) < (si or 0) + (sc or 0)
            verdict = ("%s beats IQP" % cl_label) if wins else ("IQP beats %s" % cl_label)
            if overlap:
                verdict += ", but the gap is inside 1 sd + 1 sd"
            A("| %s beats our IQP on both metrics (mean aggregation, N = 11) | %s | %s | %s | %s |"
              % (cl_label, met_label, fmt(mi, si, ni), fmt(mc, sc, nc), verdict))
    mi, si, ni = _ms("last", "iqp_K6", "pdf_js")
    mr, sr, nr = _ms("last", "rbm", "pdf_js")
    if mi is not None and mr is not None:
        verdict = "RBM still wins" if mr < mi else "RBM no longer wins"
        if abs(mi - mr) < (si or 0) + (sr or 0):
            verdict += ", but the gap is inside 1 sd + 1 sd"
        A("| under last-write-wins the RBM still wins | PDF-JS | %s | %s | %s |"
          % (fmt(mi, si, ni), fmt(mr, sr, nr), verdict))
    sat = [r["conv_mmd2"] for r in calib if r["reference"] == "gt_spatially_shuffled_px"]
    if sat:
        msat, ssat, nsat = mean_sd(sat)
        rows = []
        for agg, base in (("main", "iqp_K6"), ("main", "rbm"), ("main", "ctrl_marginal")):
            m, _, _ = _ms(agg, base, "conv_mmd2")
            if m is not None:
                rows.append((LABEL[base][0], m))
        above = [lab for lab, m in rows if m > msat]
        A("| Conv2D-MMD%s is saturated: the pixel-shuffled ground truth already scores %s | Conv2D-MMD%s | %s | shuffled GT %s | arms at or above that line: %s |"
          % (SQ, "%.4f" % msat, SQ,
             fmt(*_ms("main", "iqp_K6", "conv_mmd2")),
             fmt(msat, ssat, nsat), ", ".join(above) if above else "none"))
    A("")

    A("## C6. Quantities that carry no seed")
    A("")
    A("| Quantity | Value | Why it is deterministic |")
    A("|---|---|---|")
    A("| Reference paper's IQP PDF-JS (0.0238, 0.0221) | as published | a citation |")
    A("| Reference paper's Table 2 Conv2D-MMD%s column (0.896-1.472) | as published | a citation |" % SQ)
    A("| Held-out split (11 anchors, 200 tau) | fixed | `data_d2.HOLDOUT_SEED = 0` is a constant of the split, not a run seed |")
    A("| Ground-truth fields | fixed | derived from the dataset by that split |")
    A("| Conv2D encoder feature space | fixed | `evaluate.py` `CONV_SEED = 0` for every arm |")
    A("| CTRL-interp (0 params) | see C1 | piecewise-linear interpolation of quantised anchor fields; `controls.run_interp` draws no random numbers |")
    A("| Zero-parameter interpolation correlation | see C5 | same reason |")
    A("| Parameter counts (31,180 / 63,004 / 11,988 / 1,213,889) | exact | fixed by n, K and the architecture |")
    A("")

    if not os.path.isdir(PAPER):
        os.makedirs(PAPER)
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[aggregate] wrote %s (%d lines)" % (OUT_MD, len(L)))

    print("\n[drift] 5-seed mean vs the value Appendix B currently cites")
    for (agg, base, metric), pub in sorted(PUBLISHED.items()):
        per = table.get((agg, base))
        if not per:
            continue
        m, s, n = mean_sd([per[x][metric] for x in sorted(per)])
        if m is None:
            continue
        flag = "  <== differs by >3%" if (pub and abs(m - pub) / abs(pub) > 0.03) else ""
        print("  %-5s %-16s %-10s published %.5f  ->  5-seed %.5f +/- %.5f (ddof=1) "
              "+/- %.5f (ddof=0) (n=%d)%s"
              % (agg, base, metric, pub, m, s, sd0(s, n) or 0.0, n, flag))


if __name__ == "__main__":
    main()
