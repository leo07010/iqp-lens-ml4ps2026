# -*- coding: utf-8 -*-
"""fig_samples_mlp.png -- held-out frames above, unconditional draws from the best MLP
combination at N = 11 and N = 100 below.  Same colour scale in every row.  The generated rows
are prior draws, not reconstructions of the frames above them.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mlpsuite_common import EXP, read, is_snap, save                    # noqa: E402
sys.path.insert(0, EXP)
from data_lens_r3 import load_r3                                         # noqa: E402

SUITE = os.path.join(EXP, "results_seeds_mlp_suite.jsonl")
MLP_RE = re.compile(r"^IQP-mlpH(\d+)-M(\d+)-MMDflat-resample$")
NSHOW = 5


def pick_run(rows, n):
    """seed 0 of the arm with the most rows at this N (the suite's primary combination)."""
    cand = [r for r in rows if int(r.get("n_train", -1)) == n and r.get("class_name") is None
            and MLP_RE.match(str(r.get("arm") or ""))]
    if not cand:
        return None
    counts = {}
    for r in cand:
        counts[r["arm"]] = counts.get(r["arm"], 0) + 1
    arm = max(counts, key=counts.get)
    for r in sorted(cand, key=lambda r: r.get("seed", 99)):
        if r["arm"] == arm and os.path.isdir(r.get("run_dir") or ""):
            return r
    return None


def stack(run_dir, k):
    fs = sorted(f for f in os.listdir(run_dir)
                if f.startswith("generated_") and f.endswith(".npy"))
    return np.stack([np.load(os.path.join(run_dir, f)) for f in fs[:k]], axis=0)


def main():
    rows = [r for r in read(SUITE) if not is_snap(r.get("arm"))]
    picks = [(n, pick_run(rows, n)) for n in (11, 100)]
    picks = [(n, r) for n, r in picks if r is not None]
    if not picks:
        raise SystemExit("fig_samples_mlp: no runs found")
    d = load_r3(n_train=100, seed=0, n_holdout=200)
    ho = d.holdout_images[:NSHOW]
    vmax = float(max(ho.max(), 1e-9))
    nrow = 1 + len(picks)
    fig, axes = plt.subplots(nrow, NSHOW, figsize=(5.2, 1.15 * nrow + 0.2))
    axes = np.atleast_2d(axes)
    for j in range(NSHOW):
        axes[0, j].imshow(ho[j], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    axes[0, 0].set_ylabel("held-out", fontsize=7)
    for i, (n, r) in enumerate(picks, start=1):
        gen = stack(r["run_dir"], NSHOW)
        meta = json.load(open(os.path.join(r["run_dir"], "meta.json")))
        for j in range(NSHOW):
            axes[i, j].imshow(gen[j], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        axes[i, 0].set_ylabel("N=%d\n(unpaired)" % n, fontsize=7)
        print("[figS] N=%d arm=%s seed=%s params=%s" % (n, meta.get("arm"), meta.get("seed"),
                                                        meta.get("param_count")))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MLP arm $\\theta=\\theta_0+g(z)$: held-out and unconditional draws",
                 fontsize=7.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.06, wspace=0.04)
    save(fig, "fig_samples_mlp", also_pdf=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
