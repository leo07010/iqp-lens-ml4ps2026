# -*- coding: utf-8 -*-
"""fig_samples_v12 -- held-out frames against the v12 headline arm, at N = 11 and N = 100.

Arm: the MLP latent map, M = 2, H = 32, flat MMD, seed 0, the fixed 4,000-step protocol.
Two generated rows per N: the published fixed-S decode and the resampled-S decode of the SAME
run, rebuilt from that run's saved crops/ by evaluate_v12_flux.py's own functions -- so the
picture and the table cannot disagree.

The generated rows are unconditional prior draws.  They are NOT the model's reconstructions
of the held-out frames above them and must not be read as pairs; the figure says so.

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _mlpsuite_common import EXP, OUT, read, save                        # noqa: E402

sys.path.insert(0, EXP)
import evaluate_v12_flux as FX                                           # noqa: E402
import data_lens_r3 as D                                                 # noqa: E402

FLUX = os.path.join(EXP, "results_v12_fluxres.jsonl")
ARM = "IQP-mlpH32-M2-MMDflat-resample"
NSHOW = 6
SEED = 0


def run_dir_for(n):
    for r in read(FLUX):
        if (r.get("arm") == ARM and int(r.get("n_train", -1)) == n
                and r.get("seed") == SEED and r.get("class_name") is None):
            return r["run_dir"]
    return None


def both_decodes(run_dir, n):
    """(fixed-S frames, resampled-S frames) for one run, rebuilt the way the tables were."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    crops = FX.load_crops(run_dir)
    offs = np.asarray(meta["gen_offsets"], dtype=np.int64)
    totals = FX.train_totals_for(n, None)
    S_fix = np.full(crops.shape[0], float(totals.mean()), dtype=np.float64)
    S_res, h = FX.sample_flux_kde(totals, crops.shape[0], SEED)
    return (FX.frames_from_crops(crops, offs, S_fix),
            FX.frames_from_crops(crops, offs, S_res), meta, h)


def main():
    d = D.load_r3(100, n_holdout=200)
    ho = d.holdout_images[:NSHOW]
    vmax = float(max(ho.max(), 1e-9))

    blocks = []
    for n in (11, 100):
        rd = run_dir_for(n)
        if rd is None:
            print("[figs] no run for N=%d -- skipped" % n)
            continue
        gf, gr, meta, h = both_decodes(rd, n)
        blocks.append((n, gf[:NSHOW], gr[:NSHOW], meta, h))
    if not blocks:
        raise SystemExit("fig_samples_v12: nothing to draw")

    nrow = 1 + 2 * len(blocks)
    fig, axes = plt.subplots(nrow, NSHOW, figsize=(5.4, 0.95 * nrow))
    for j in range(NSHOW):
        axes[0, j].imshow(ho[j], cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
    axes[0, 0].set_ylabel("held-out\n(real)", fontsize=6.2)
    row = 1
    for n, gf, gr, meta, h in blocks:
        for frames, tag in ((gf, "gen. $N$=%d\nfixed $S$" % n),
                            (gr, "gen. $N$=%d\nresampled $S$" % n)):
            for j in range(NSHOW):
                axes[row, j].imshow(frames[j], cmap="magma", vmin=0.0, vmax=vmax,
                                    interpolation="nearest")
            axes[row, 0].set_ylabel(tag, fontsize=6.2)
            row += 1
    for r in range(nrow):
        for c in range(NSHOW):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    m0 = blocks[0][3]
    fig.suptitle("IQP + MLP latent map, $M$=2, $H$=32, flat MMD, seed 0, %d steps, "
                 "%s params  |  same colour scale throughout"
                 % (int(m0.get("steps", 0)), "{:,}".format(int(m0.get("param_count", 0)))),
                 fontsize=7, y=0.995)
    fig.text(0.0, -0.02,
             "The generated rows are unconditional prior draws, NOT reconstructions of the "
             "frames above them: read each row as a sample, not as a column-wise pair.\n"
             "Within one $N$ the two generated rows are the SAME trained model and the SAME "
             "circuit output, decoded with a single training-mean total intensity and with a "
             "per-image one drawn from a Silverman KDE of the training totals.",
             fontsize=5.0, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.subplots_adjust(hspace=0.06, wspace=0.04)
    path = os.path.join(OUT, "fig_samples_v12.png")
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print("[figs] wrote %s (%d bytes)" % (path, os.path.getsize(path)))
    for n, gf, gr, meta, h in blocks:
        print("[figs] N=%-4d fixed total %.2f | resampled totals %.2f +- %.2f (bandwidth %.2f)"
              % (n, gf.reshape(gf.shape[0], -1).sum(1).mean(),
                 gr.reshape(gr.shape[0], -1).sum(1).mean(),
                 gr.reshape(gr.shape[0], -1).sum(1).std(ddof=1), h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
