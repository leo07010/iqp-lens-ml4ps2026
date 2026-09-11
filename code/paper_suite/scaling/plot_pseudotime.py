#!/usr/bin/env python3
"""Paper Table 2 analogue: unseen-time synthesis by latent interpolation.

docs/IQP_LENS_PAPER_SUITE_PLAN.md, row "Table 2 qualitative samples + unseen-time synthesis by
latent interpolation" -> "pseudo-time arm (ellipticity e1 as the time axis): generated crops
along e1 vs the nearest real crops; latent trajectory plot".

The pseudo-time arm already ran in round 3 (train_iqp_r3.py --protocol pseudotime, arm
`IQP-lowvar-pseudotime-MMD`): it visits the N=100 training images in ascending e1, trains a
per-image latent, and at generation time draws ONE sample per held-out image by interpolating
the trained latents piecewise-linearly in e1 (clamped outside the training range) at that
held-out image's own e1 -- see train_iqp_r3.py's module docstring and `interp_latents_in_e1`.
So `crops/generated_{i:03d}.npy` is already paired with held-out image i at (as close as the
model gets to) that image's own e1; this script does not train or generate anything new, it
only re-renders that pairing as a figure.

NEW file, reads but does not modify outputs_lens_r3/iqp_pseudotime_n100/, results_r3.jsonl,
data_lens_r3.py or train_iqp_r3.py.

What is recomputed vs read
---------------------------
e1 of every training / held-out crop : recomputed by train_iqp_r3.e1_of on the crops from
    data_lens_r3.load_r3(n_train, seed, n_holdout) -- a pure function of the meta.json's own
    (n_train, seed, n_holdout), so this reproduces train_iqp_r3.py's e1_train / e1_hold bit for
    bit (same cache, same deterministic slicing, no randomness involved).
the latent trajectory theta_lat(e1) used for generation : NOT saved by train_iqp_r3.py (only
    the N=100 PER-TRAINING-IMAGE latents are saved, in theta_lat.npy) -- recomputed here by
    calling the SAME train_iqp_r3.interp_latents_in_e1(e1_train, theta_lat, targets) the
    training script called internally, which is deterministic given theta_lat.npy and the two
    e1 arrays above. This is therefore the EXACT (n_gen, 50) trajectory that was decoded into
    the generated crops, not an approximation.
mean |Delta e1| in the figure title : read from results_r3.jsonl's `pair_de1` field for the
    matching run_dir (evaluate_r3.py's paired_block, mean over ALL n_pairs pairs) rather than
    recomputed, per the task order ("read from results_r3.jsonl or meta"); falls back to a
    recompute over the shown crops only if no matching row is found, and says so in the title.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../paper_suite/scaling -> .../13_iqp_turbulence_replication
sys.path.insert(0, ROOT)

import data_lens_r3                                     # noqa: E402
from train_iqp_r3 import e1_of, interp_latents_in_e1     # noqa: E402

# ---------------------------------------------------------------------------------------
# Palette: validated categorical slots (blue/orange/aqua/yellow/magenta, fixed adjacent-pair
# order per the dataviz skill's reference palette) and the same chart chrome / SEQ colormap
# tokens as analysis_lens/make_figures.py, so this figure sits visually with the others under
# docs/figures/.
# ---------------------------------------------------------------------------------------
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec",
                                                "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "lines.linewidth": 1.8, "axes.titlelocation": "left",
})

DEFAULT_PSEUDOTIME_DIR = os.path.join(ROOT, "outputs_lens_r3", "iqp_pseudotime_n100")
DEFAULT_RESULTS = os.path.join(ROOT, "results_r3.jsonl")
DEFAULT_OUT = "<REPO_ROOT>/docs/figures/iqp_lens_pseudotime.png"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pseudo-time unseen-time synthesis figure.")
    p.add_argument("--pseudotime-dir", default=DEFAULT_PSEUDOTIME_DIR,
                   help="a train_iqp_r3.py --protocol pseudotime output dir (conditional=true)")
    p.add_argument("--results", default=DEFAULT_RESULTS,
                   help="jsonl carrying this run's pair_de1 (evaluate_r3.py schema)")
    p.add_argument("--n-show", type=int, default=12, help="held-out crops shown, spread over e1")
    p.add_argument("--n-latent-dims", type=int, default=5,
                   help="top-variance latent dims plotted in the trajectory panel")
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args(argv)


def find_result_row(results_path, run_dir):
    """The results_r3(.b).jsonl row whose run_dir matches `run_dir`, or None."""
    if not os.path.isfile(results_path):
        return None
    want = os.path.normpath(os.path.abspath(run_dir))
    with open(results_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rd = row.get("run_dir")
            if rd and os.path.normpath(os.path.abspath(rd)) == want:
                return row
    return None


def main(argv=None):
    args = parse_args(argv)
    pt_dir = args.pseudotime_dir
    meta_path = os.path.join(pt_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit("plot_pseudotime.py: no meta.json under %r" % pt_dir)
    with open(meta_path) as fh:
        meta = json.load(fh)
    if not meta.get("conditional"):
        raise SystemExit("plot_pseudotime.py: %r has conditional=%r -- this is not a "
                          "--protocol pseudotime run (or an equivalent conditional arm)"
                          % (pt_dir, meta.get("conditional")))

    n_train = int(meta["n_train"])
    n_holdout = int(meta["n_holdout"])
    n_gen = int(meta["n_gen"])
    seed = int(meta.get("seed", 0))

    d = data_lens_r3.load_r3(n_train, seed=seed, n_holdout=n_holdout)
    e1_train = np.asarray([e1_of(d.train_crops[i]) for i in range(n_train)], dtype=np.float64)
    e1_hold = np.asarray([e1_of(d.holdout_crops[i]) for i in range(d.holdout_crops.shape[0])],
                         dtype=np.float64)
    targets = e1_hold[:n_gen]                            # exactly what train_iqp_r3.py targeted

    theta_lat_path = os.path.join(pt_dir, "theta_lat.npy")
    if not os.path.isfile(theta_lat_path):
        raise SystemExit("plot_pseudotime.py: %r missing (needed to recompute the latent "
                          "trajectory)" % theta_lat_path)
    theta_lat = np.load(theta_lat_path)
    if theta_lat.shape[0] != n_train:
        raise SystemExit("plot_pseudotime.py: theta_lat.npy has %d rows, meta.json n_train=%d"
                          % (theta_lat.shape[0], n_train))
    lat_gen = interp_latents_in_e1(e1_train, theta_lat, targets)      # (n_gen, d_lat), exact

    gen_crops = np.stack([np.load(os.path.join(pt_dir, "crops", "generated_%03d.npy" % i))
                          for i in range(n_gen)], axis=0)
    e1_gen = np.asarray([e1_of(gen_crops[i]) for i in range(n_gen)], dtype=np.float64)

    # 12 held-out targets spread across the e1 range (evenly spaced in SORTED e1 position, not
    # in raw index), so the panel reads left-to-right as "low e1 -> high e1".
    order = np.argsort(targets)
    n_show = max(2, min(int(args.n_show), n_gen))
    pos = np.unique(np.linspace(0, n_gen - 1, n_show).round().astype(int))
    sel = order[pos]

    row = find_result_row(args.results, pt_dir)
    if row is not None and row.get("pair_de1") is not None:
        mean_de1 = float(row["pair_de1"])
        n_pairs = int(row.get("n_pairs", n_gen))
        de1_source = "results row, all %d pairs" % n_pairs
    else:
        mean_de1 = float(np.abs(e1_gen - targets).mean())
        n_pairs = n_gen
        de1_source = "recomputed here (no matching results row found), all %d pairs" % n_pairs

    # latent trajectory: top-variance dims of the n_gen x d_lat INTERPOLATED trajectory itself
    # (the thing that was actually decoded), not of the N training-image latents.
    var = lat_gen.var(axis=0)
    k = min(int(args.n_latent_dims), lat_gen.shape[1], len(S))
    top_dims = np.argsort(var)[::-1][:k]

    # ---------------------------------------------------------------------------- figure
    n_cols = len(sel)
    fig = plt.figure(figsize=(max(1.15 * n_cols, 9.0) + 1.0, 7.4))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1.0, 1.0, 1.55], hspace=0.65, wspace=0.10,
                          top=0.87, bottom=0.075, left=0.055, right=0.99)

    vmax = float(max(d.holdout_crops[sel].max(), gen_crops[sel].max(), 1e-12))
    for c, idx in enumerate(sel):
        axt = fig.add_subplot(gs[0, c])
        axt.imshow(d.holdout_crops[idx], cmap=SEQ, vmin=0.0, vmax=vmax, origin="lower")
        axt.set_xticks([]); axt.set_yticks([])
        axt.grid(False)
        axt.set_title("e1=%.3f" % targets[idx], fontsize=7.3, color=INK2, pad=3)
        if c == 0:
            axt.set_ylabel("held-out\ntarget", fontsize=8.3, color=INK)

        axg = fig.add_subplot(gs[1, c])
        axg.imshow(gen_crops[idx], cmap=SEQ, vmin=0.0, vmax=vmax, origin="lower")
        axg.set_xticks([]); axg.set_yticks([])
        axg.grid(False)
        panel_de1 = abs(float(e1_gen[idx]) - float(targets[idx]))
        axg.set_title(u"Δe1=%.3f" % panel_de1, fontsize=7.3, color=INK2, pad=3)
        if c == 0:
            axg.set_ylabel("pseudo-time\ngenerated", fontsize=8.3, color=INK)

    axl = fig.add_subplot(gs[2, :])
    order_all = np.argsort(targets)
    for j, dim in enumerate(top_dims):
        axl.plot(targets[order_all], lat_gen[order_all, dim], color=S[j], lw=1.7,
                marker="o", ms=2.8, label="latent[%d]  var=%.3g" % (int(dim), float(var[dim])))
    for e1v in targets[sel]:
        axl.axvline(e1v, color=GRID, lw=0.8, zorder=0)
    if float(e1_train.min()) > float(targets.min()):
        axl.axvspan(float(targets.min()), float(e1_train.min()), color=BASE, alpha=0.22, lw=0,
                   label="_nolegend_")
    if float(e1_train.max()) < float(targets.max()):
        axl.axvspan(float(e1_train.max()), float(targets.max()), color=BASE, alpha=0.22, lw=0,
                   label="_nolegend_")
    axl.set_xlabel("held-out e1 (ellipticity, the pseudo-time axis)", fontsize=9)
    axl.set_ylabel("interpolated latent coordinate", fontsize=9)
    axl.legend(fontsize=6.6, ncol=min(k, 5), loc="best")
    n_clamped = int(np.sum((targets < e1_train.min()) | (targets > e1_train.max())))
    axl.set_title("latent trajectory theta_lat(e1): top %d of %d dims by variance across the "
                  "%d held-out targets (shaded = outside the N=%d training e1 range "
                  "[%.3f, %.3f], clamped there -- %d/%d targets)"
                  % (k, lat_gen.shape[1], n_gen, n_train, e1_train.min(), e1_train.max(),
                     n_clamped, n_gen), fontsize=8.2, color=INK2)

    fig.suptitle(u"IQP pseudo-time synthesis -- %s (N_train=%d, %d held-out, seed %d)\n"
                u"mean |Δe1| generated vs target = %.4f (%s); %d of %d shown here, "
                u"spread across the held-out e1 range"
                % (meta.get("arm", "?"), n_train, n_holdout, seed, mean_de1, de1_source,
                   n_cols, n_gen),
                fontsize=10.3, color=INK, y=0.995)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)

    shown_de1 = float(np.abs(e1_gen[sel] - targets[sel]).mean())
    print("[plot_pseudotime] wrote %s (%d KB)" % (args.out, os.path.getsize(args.out) // 1024))
    print("[plot_pseudotime] %d panels, e1 range shown [%.4f, %.4f] of held-out [%.4f, %.4f]; "
          "training e1 range [%.4f, %.4f] (%d/%d targets clamped outside it)"
          % (n_cols, targets[sel].min(), targets[sel].max(), targets.min(), targets.max(),
             e1_train.min(), e1_train.max(), n_clamped, n_gen))
    print("[plot_pseudotime] mean |delta e1| over the %d SHOWN pairs = %.4f "
          "(figure title reports the full-set mean = %.4f from %s)"
          % (n_cols, shown_de1, mean_de1, de1_source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
