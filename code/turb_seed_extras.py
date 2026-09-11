#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-seed extras for the 5-seed Appendix-B table (NEW FILE -- touches no existing output).

Two independent tasks, both writing one JSON line per record to --results:

  --task anchor-corr
      For each trained IQP run dir (outputs_*/iqp_K{K}_seed{s}), recompute the anchor-fit
      correlation the way docs/IQP_TURBULENCE_REPLICATION_RESULTS.md 3(d)/(5) does:
      for every one of the 11 training anchors, take theta = concat(theta_lat_anchors[j],
      theta_core) and score the generated field against that anchor's ground-truth field,
      once through the SAMPLED aggregation path (1e5 samples, codec.aggregate -- the "0.69"
      number) and once through the EXACT conditional mean E_model[z|x,y] (the "0.77"
      number).  Same machinery as diag_scripts/q2_anchor_fit.py and
      anchor_fit_experiment.exact_conditional_field, but parameterised by run dir instead of
      hardcoded to seed 0.

  --task calib
      The metric-calibration references of RESULTS.md 5 (1)-(4): the spatially-shuffled
      ground truth (the "0.704" Conv-MMD^2 saturation floor), the tau-shuffled ground truth,
      and the zero-parameter linear interpolation / nearest-anchor copy correlations.  The
      shuffle references are re-run over --calib-seeds shuffle seeds; the interpolation
      references take no randomness at all and are emitted once, flagged deterministic=true.
      The Conv2D encoder seed stays fixed at evaluate.py's CONV_SEED = 0 for every record,
      so all arms and all references live in the same feature space (varying it would change
      the metric, not the model).

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

import codec
import data_d2

# Same guard as diag_scripts/common.py:3 / anchor_fit_experiment.py / train_iqp.py: an FWHT at
# n = 18 under torch's default (often 100+) intra-op pool is dominated by sync overhead and runs
# ~1000x slower.  The sbatch also exports OMP_NUM_THREADS=4; this makes a manual run safe too.
torch.set_num_threads(min(4, os.cpu_count() or 1))
from anchor_fit_experiment import exact_conditional_field, sampled_field
from iqp_exact import IQPExact
from metrics import conv_mmd2, pdf_js

CONV_SEED = 0            # must match evaluate.py's CONV_SEED
SAMPLE_GEN_SEED = 999    # must match diag_scripts/q2_anchor_fit.py's gg.manual_seed(999)
N_SYNTH_ANCHOR = 100000  # must match q2_anchor_fit.py's 1e5 sampled path
_GT_RE = re.compile(r"gt_t(-?\d+)\.npy$")


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


# --------------------------------------------------------------------------------------
# task: anchor-corr
# --------------------------------------------------------------------------------------
def anchor_corr_one(run_dir, plane, device):
    with open(os.path.join(run_dir, "meta.json")) as fh:
        meta = json.load(fh)
    theta_core = torch.as_tensor(np.load(os.path.join(run_dir, "theta_core.npy")),
                                 dtype=torch.float32)
    lat = np.load(os.path.join(run_dir, "theta_lat_anchors.npy"))
    K = int(meta["K"])
    anchors = [int(t) for t in meta["anchors"]]
    a_z, b_z = data_d2.get_z_range(plane)
    zlut = codec.dequantize_z(np.arange(64), a_z, b_z)

    sim = IQPExact(n=codec.N_QUBITS, K=K, device=device, dtype=torch.complex64)
    gen = torch.Generator(device=device)
    gen.manual_seed(SAMPLE_GEN_SEED)

    assert anchors and anchors[0] == 1, (
        "anchor_corr_*_t1 reports per[0] as anchor t=1; this run's anchors start at %r" % (anchors[:1],))
    per = []
    for j, t in enumerate(anchors):
        theta = torch.cat((torch.as_tensor(lat[j], dtype=torch.float32), theta_core)).to(device)
        gt_field = data_d2.load_d2_snapshot(t, plane=plane).field.astype(np.float32)
        f_sampled = sampled_field(sim, theta, a_z, b_z, N_SYNTH_ANCHOR, gen)
        f_exact = exact_conditional_field(sim, theta, zlut)
        per.append(dict(t=t,
                        corr_sampled=_corr(f_sampled, gt_field),
                        corr_exact=_corr(f_exact, gt_field)))
    return meta, per


def task_anchor_corr(args, sink):
    for run_dir in args.run_dirs:
        run_dir = os.path.abspath(run_dir.rstrip("/"))
        if not os.path.isdir(run_dir):
            raise SystemExit("[FATAL] not a directory: %s" % run_dir)
        meta, per = anchor_corr_one(run_dir, args.plane, args.device)
        cs = [r["corr_sampled"] for r in per]
        ce = [r["corr_exact"] for r in per]
        rec = dict(
            task="anchor_corr",
            run_dir=run_dir,
            arm=os.path.basename(run_dir),
            seed=int(meta.get("seed", -1)),
            K=int(meta["K"]),
            param_count=int(meta.get("param_count", -1)),
            n_anchors=len(per),
            anchor_corr_sampled_mean=float(np.mean(cs)),
            anchor_corr_exact_mean=float(np.mean(ce)),
            anchor_corr_sampled_t1=float(per[0]["corr_sampled"]),
            anchor_corr_exact_t1=float(per[0]["corr_exact"]),
            per_anchor=per,
            deterministic=False,
        )
        sink(rec)
        print("[anchor-corr] %-20s seed=%s  mean corr sampled=%.4f  exact=%.4f  (t=1: %.4f / %.4f)"
              % (rec["arm"], rec["seed"], rec["anchor_corr_sampled_mean"],
                 rec["anchor_corr_exact_mean"], rec["anchor_corr_sampled_t1"],
                 rec["anchor_corr_exact_t1"]))


# --------------------------------------------------------------------------------------
# task: calib
# --------------------------------------------------------------------------------------
def load_ground_truth(gt_dir):
    taus, fields = [], []
    for path in sorted(glob.glob(os.path.join(gt_dir, "gt_t*.npy"))):
        m = _GT_RE.match(os.path.basename(path))
        if m:
            taus.append(int(m.group(1)))
            fields.append(np.load(path))
    order = np.argsort(taus)
    taus = [taus[i] for i in order]
    return taus, np.stack([fields[i] for i in order]).astype(np.float32)


def task_calib(args, sink):
    taus, gts = load_ground_truth(args.ground_truth)
    print("[calib] %d held-out ground-truth fields from %s" % (len(taus), args.ground_truth))

    def score(gen):
        return float(pdf_js(gts, gen)), float(conv_mmd2(gts, gen, seed=CONV_SEED))

    # ---- deterministic reference: identical to GT
    js, mm = score(gts)
    sink(dict(task="calib", reference="identical_to_gt", deterministic=True,
              shuffle_seed=None, pdf_js=js, conv_mmd2=mm))
    print("[calib] identical_to_gt              PDF-JS=%.5f  Conv-MMD2=%.5f" % (js, mm))

    # ---- seeded references: spatial pixel shuffle / tau shuffle
    for s in args.calib_seeds:
        rng = np.random.default_rng(int(s))
        spat = np.stack([g.ravel()[rng.permutation(g.size)].reshape(g.shape) for g in gts])
        js, mm = score(spat.astype(np.float32))
        sink(dict(task="calib", reference="gt_spatially_shuffled_px", deterministic=False,
                  shuffle_seed=int(s), pdf_js=js, conv_mmd2=mm))
        print("[calib] gt_spatially_shuffled_px  seed=%d  PDF-JS=%.5f  Conv-MMD2=%.5f" % (s, js, mm))

    for s in args.calib_seeds:
        rng = np.random.default_rng(int(s))
        tsh = gts[rng.permutation(gts.shape[0])]
        js, mm = score(tsh.astype(np.float32))
        sink(dict(task="calib", reference="gt_shuffled_across_tau", deterministic=False,
                  shuffle_seed=int(s), pdf_js=js, conv_mmd2=mm))
        print("[calib] gt_shuffled_across_tau    seed=%d  PDF-JS=%.5f  Conv-MMD2=%.5f" % (s, js, mm))

    # ---- deterministic references: zero-parameter interpolation / nearest-anchor copy
    anchors = [int(t) for t in data_d2.ANCHOR_TS]
    a_z, b_z = data_d2.get_z_range(args.plane)
    af = {t: data_d2.load_d2_snapshot(t, plane=args.plane).field.astype(np.float64)
          for t in anchors}
    A = np.asarray(anchors)
    interp, near = [], []
    ci, cn = [], []
    for k, t in enumerate(taus):
        j = int(np.searchsorted(A, t))
        lo = anchors[max(j - 1, 0)]
        hi = anchors[min(j, len(anchors) - 1)]
        if hi == lo:
            pred = af[lo]
        else:
            w = (t - lo) / float(hi - lo)
            pred = (1.0 - w) * af[lo] + w * af[hi]
        nearest = af[lo] if abs(t - lo) <= abs(t - hi) else af[hi]
        interp.append(pred)
        near.append(nearest)
        ci.append(_corr(pred, gts[k]))
        cn.append(_corr(nearest, gts[k]))
    interp = np.stack(interp).astype(np.float32)
    near = np.stack(near).astype(np.float32)
    js_i, mm_i = score(interp)
    js_n, mm_n = score(near)
    sink(dict(task="calib", reference="linear_interp_of_bracketing_anchors", deterministic=True,
              shuffle_seed=None, corr_mean=float(np.mean(ci)), corr_min=float(np.min(ci)),
              pdf_js=js_i, conv_mmd2=mm_i))
    sink(dict(task="calib", reference="nearest_anchor_copy", deterministic=True,
              shuffle_seed=None, corr_mean=float(np.mean(cn)), corr_min=float(np.min(cn)),
              pdf_js=js_n, conv_mmd2=mm_n))
    print("[calib] linear_interp   corr mean=%.5f min=%.5f  PDF-JS=%.5f  Conv-MMD2=%.5f"
          % (np.mean(ci), np.min(ci), js_i, mm_i))
    print("[calib] nearest_anchor  corr mean=%.5f min=%.5f  PDF-JS=%.5f  Conv-MMD2=%.5f"
          % (np.mean(cn), np.min(cn), js_n, mm_n))
    print("[calib] (a_z, b_z) = %.6f, %.6f" % (a_z, b_z))


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["anchor-corr", "calib"], required=True)
    p.add_argument("--run-dirs", nargs="*", default=[],
                   help="anchor-corr: trained IQP run directories")
    p.add_argument("--ground-truth", default="outputs_d2_main/ground_truth")
    p.add_argument("--results", required=True, help="output jsonl (appended to)")
    p.add_argument("--plane", default="mid")
    p.add_argument("--device", default="cpu")
    p.add_argument("--calib-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = os.path.abspath(args.results)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fh = open(out, "a")

    def sink(rec):
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()

    try:
        if args.task == "anchor-corr":
            task_anchor_corr(args, sink)
        else:
            task_calib(args, sink)
    finally:
        fh.close()
    print("[seed_extras] wrote records to %s" % out)


if __name__ == "__main__":
    main()
