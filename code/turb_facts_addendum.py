#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit the facts.md Addendum for the Appendix-B 5-seed pass, filled from
results_d2_seeds.jsonl.  Prints to stdout; the caller appends it to
paper/ml4ps2026/facts.md (append-only, because other tracks edit that file too).

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import math
import os

REPO = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(REPO, "results_d2_seeds.jsonl")


def mean_sd(vals):
    vals = [float(v) for v in vals
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, 1
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)), n


def f(m, s, n, d=4):
    """Mirrors turb_aggregate_seeds.fmt.  A single seed has no spread to report, and a spread of
    exactly zero over >=2 seeds means the quantity does not depend on the seed at all -- printing
    either as '± 0.0000' would read as 'five runs agreed exactly'."""
    if m is None:
        return "n/a"
    if n <= 1:
        return "%.*f (n=1, no spread available)" % (d, m)
    if s is not None and s < 1e-12:
        return "%.*f (seed-independent, n=%d)" % (d, m, n)
    return "%.*f ± %.*f (n=%d)" % (d, m, d, s, n)


def main():
    rows = [json.loads(l) for l in open(JSONL) if l.strip()]

    def t2(agg, key, metric):
        return mean_sd([r[metric] for r in rows
                        if r["quantity"] == "table2" and r.get("aggregation") == agg
                        and r.get("arm_key") == key and r.get("seed") is not None])

    def anch(key, field):
        return mean_sd([r[field] for r in rows
                        if r["quantity"] == "anchor_corr" and r.get("arm_key") == key])

    def calib(ref, metric):
        return mean_sd([r[metric] for r in rows
                        if r["quantity"] == "metric_calibration" and r.get("reference") == ref])

    # a max over a partial config sweep is biased low, so only seeds with the full config set
    # contribute (same rule as turb_aggregate_seeds.py)
    cfg_by_seed = {}
    for r in rows:
        if r["quantity"] == "anchor_fit_ceiling" and r.get("corr") is not None:
            cfg_by_seed.setdefault(r["seed"], {})[r["config"]] = r["corr"]
    n_cfg = max((len(v) for v in cfg_by_seed.values()), default=0)
    ceil_by_seed = {s: max(v.values()) for s, v in cfg_by_seed.items() if len(v) == n_cfg}
    ceil = mean_sd(list(ceil_by_seed.values()))

    prot = {}
    for r in rows:
        if r["quantity"] == "anchor_fit_protocol":
            prot.setdefault(r["config"], {})[r["seed"]] = r.get("corr")

    def delta(a, b):
        vals = []
        for s in sorted(set(prot.get(a, {})) & set(prot.get(b, {}))):
            va, vb = prot[a][s], prot[b][s]
            if va is not None and vb is not None:
                vals.append(vb - va)
        return mean_sd(vals)

    interp = [r for r in rows if r["quantity"] == "metric_calibration"
              and r.get("reference") == "linear_interp_of_bracketing_anchors"]
    ctrl_interp = [r for r in rows if r["quantity"] == "table2" and r.get("arm_key") == "ctrl_interp"
                   and r.get("aggregation") == "main"]

    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A("## Addendum 2026-09-09: Appendix B (turbulence replication) at 5 seeds")
    A("")
    A("Every Appendix-B number is now a 5-seed mean ± sd, or is explicitly marked as carrying "
      "no seed. Full tables: `paper/ml4ps2026/seed_tables_c.md`; per-seed records: "
      "`experiments/13_iqp_turbulence_replication/results_d2_seeds.jsonl`. Seeds 0-2 are the "
      "existing runs (job 342220 and the last-write job); seeds 3-4 are new (jobs 360790 / "
      "360773 / 360791, dev partition). `sd` is the sample sd (ddof = 1); the ± values the "
      "paper cited before came from `evaluate.py`'s `numpy.std` (ddof = 0) and are smaller by "
      "√((n−1)/n) = 0.894 at n = 5.")
    A("")
    A("### Table 2, pre-registered (mean) aggregation, N = 11")
    A("")
    A("| Arm | PDF–JS | Conv2D-MMD² |")
    A("|---|---|---|")
    for key, lab in (("iqp_K6", "IQP-31k"), ("iqp_K7", "IQP-63k"), ("rbm", "RBM-12k"),
                     ("dcgan_1.2m_n11", "DCGAN-1.2M"), ("ctrl_marginal", "CTRL-marginal")):
        A("| %s | %s | %s |" % (lab, f(*t2("main", key, "pdf_js")),
                                f(*t2("main", key, "conv_mmd2"))))
    if ctrl_interp:
        A("| CTRL-interp (0 params) | %.5f **deterministic** | %.6f **deterministic** |"
          % (ctrl_interp[0]["pdf_js"], ctrl_interp[0]["conv_mmd2"]))
    A("")
    A("### Table 2, last-write-wins aggregation")
    A("")
    A("| Arm | PDF–JS | Conv2D-MMD² |")
    A("|---|---|---|")
    for key, lab in (("iqp_K6", "IQP-31k"), ("iqp_K7", "IQP-63k"), ("rbm", "RBM-12k"),
                     ("ctrl_marginal", "CTRL-marginal")):
        A("| %s | %s | %s |" % (lab, f(*t2("last", key, "pdf_js")),
                                f(*t2("last", key, "conv_mmd2"))))
    A("")
    A("### Anchor fit, model-family ceiling, protocol attribution")
    A("")
    A("| Quantity | 5-seed value | previously cited |")
    A("|---|---|---|")
    A("| IQP-31k anchor corr, 11-anchor mean, sampled / exact | %s / %s | 0.69 / 0.77 |"
      % (f(*anch("iqp_K6", "anchor_corr_sampled")), f(*anch("iqp_K6", "anchor_corr_exact"))))
    A("| IQP-31k anchor corr, t = 1, sampled / exact | %s / %s | 0.774 (exact) |"
      % (f(*anch("iqp_K6", "anchor_corr_sampled_t1")), f(*anch("iqp_K6", "anchor_corr_exact_t1"))))
    A("| IQP-63k anchor corr, 11-anchor mean, sampled / exact | %s / %s | — |"
      % (f(*anch("iqp_K7", "anchor_corr_sampled")), f(*anch("iqp_K7", "anchor_corr_exact"))))
    A("| model-family ceiling (best config of the 8-config sweep) | %s | 0.88–0.89 |" % f(*ceil))
    A("| Δ corr from stage 1 freezing θ_lat (as attributed before) | %s | −0.08 |"
      % f(*delta("allparams_subsample5e4", "paper_stage1")))
    A("| Δ corr from stage 1 freezing θ_lat, unconfounded (both cells at 5×10⁴ subsample) | %s | not reported |"
      % f(*delta("allparams_subsample5e4", "paper_stage1_subsample5e4")))
    A("| Δ corr from the 5×10⁴-sample pmf | %s | −0.04 |" % f(*delta("paper_stage1", "paper_stage1_subsample5e4")))
    A("| Δ corr from Algorithm 1 stage 2 | %s | +0.0001 |" % f(*delta("paper_stage1", "paper_stage1_plus_stage2")))
    A("")
    A("**Provenance correction on the 0.69 / 0.77 pair.** `docs/IQP_TURBULENCE_REPLICATION_"
      "RESULTS.md` §4(2) attributes the 11-anchor mean \"0.691 sampled / 0.776 exact\" to the "
      "IQP-31k main run. Recomputing per arm shows that pair is the **IQP-63k** seed-0 mean "
      "(0.6913 / 0.7755); IQP-31k seed 0 is 0.6653 / 0.7538. The diagnostic it came from, "
      "`diag_scripts/q2_anchor_fit.py`, loops `[(\"iqp_K7_seed0\", 7), (\"iqp_K6_seed0\", 6)]` "
      "and prints the K=7 block first. What *is* IQP-31k is the anchor t = 1 exact correlation "
      "0.7738, independently confirmed by `anchor_fit_protocol.py --summarize` in job 342685. "
      "Appendix B should cite either the IQP-31k 11-anchor mean (%s exact) or the IQP-31k "
      "t = 1 value (%s exact), not the current pair."
      % (f(*anch("iqp_K6", "anchor_corr_exact")), f(*anch("iqp_K6", "anchor_corr_exact_t1"))))
    A("")
    A("**The ceiling sweep carries no seed.** Run at seeds %s, `anchor_fit_experiment.py` "
      "returns bit-identical `corr` for all %d configs: the pmf is the anchor's full native "
      "code pool (no subsample draw), the init is data-dependent, and the `auto3` bandwidths "
      "come from that same pool, so nothing in the optimisation consumes the RNG. The seed does "
      "reach the run and does move `pdf_js`, which is scored on a sampled field. So `0.88–0.89` "
      "is a range across configs, not a seed spread. The protocol rows are different: the two "
      "`*subsample5e4` cells draw a 5×10⁴ subsample and `paper_stage1` freezes θ_lat at a random "
      "N(0, 0.01) init, so those do vary with the seed."
      % (", ".join(str(x) for x in sorted(cfg_by_seed)), n_cfg))
    A("")
    A("**The protocol attribution is where 5 seeds change the story.** At seed 0 alone the "
      "freeze read −0.078 and the pmf subsample read −0.038, which is the −0.08 / −0.04 split "
      "the paper cites. Over 5 seeds the freeze is %s and the subsample is %s: the subsample "
      "term is indistinguishable from zero and the freeze absorbs it. The seed-0 split was an "
      "artefact of `paper_stage1` landing high at that seed (0.8074 against a 5-seed mean of "
      "%s) — that cell freezes θ_lat at a random N(0, 0.01) draw and is the most seed-sensitive "
      "cell in the table. The 5-seed statement is that stage 1 freezing θ_lat is essentially "
      "the whole protocol penalty and the 5×10⁴-sample pmf costs nothing measurable. The two "
      "freeze rows now agree because the factor separating them is null."
      % (f(*delta("allparams_subsample5e4", "paper_stage1")),
         f(*delta("paper_stage1", "paper_stage1_subsample5e4")),
         f(*mean_sd([prot["paper_stage1"][k] for k in sorted(prot.get("paper_stage1", {}))]))))
    A("")
    A("### Metric-calibration references")
    A("")
    A("| Reference | 5-seed value | previously cited |")
    A("|---|---|---|")
    A("| pixel-shuffled ground truth, Conv2D-MMD² | %s | 0.704 |"
      % f(*calib("gt_spatially_shuffled_px", "conv_mmd2")))
    A("| τ-shuffled ground truth, PDF–JS / Conv2D-MMD² | %s / %s | 0 / 0 |"
      % (f(*calib("gt_shuffled_across_tau", "pdf_js"), d=5),
         f(*calib("gt_shuffled_across_tau", "conv_mmd2"), d=5)))
    if interp:
        A("| zero-parameter linear interpolation, corr | %.5f **deterministic** | 0.9998 |"
          % interp[0]["corr_mean"])
    A("")
    A("The `linear interpolation` row interpolates the **raw** anchor fields, which is what "
      "`diag_scripts/q4_task_triviality.py` did to get the cited 0.9998. It is a different "
      "object from the CTRL-interp arm above, which interpolates the **quantised** anchor "
      "fields so the zero-parameter control carries the same 6-bit z cap as every generative "
      "arm. That is why CTRL-interp scores PDF–JS 0.00119 and this row scores about 0.0001.")
    A("")
    A("### Quantities that carry no seed")
    A("")
    A("The reference paper's own reported values (IQP PDF–JS 0.0238 / 0.0221, its Table 2 "
      "Conv2D-MMD² column 0.896–1.472) are citations. The 11-anchor / 200-τ split is fixed by "
      "`data_d2.HOLDOUT_SEED = 0`, a constant of the split rather than a run seed, so the "
      "ground-truth fields are fixed too. The Conv2D encoder is re-derived from "
      "`evaluate.py`'s `CONV_SEED = 0` for every arm, so the feature space is shared and "
      "fixed. CTRL-interp and the zero-parameter interpolation correlation draw no random "
      "numbers at all (`controls.run_interp`). Parameter counts are exact.")
    A("")
    print("\n".join(L))


if __name__ == "__main__":
    main()
