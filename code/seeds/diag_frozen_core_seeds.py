#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed replicate of the FROZEN-CORE reconstruction number of paper section 3.2.

The paper says: "With the core frozen on one anchor, the 50 fixed latent coordinates
reconstruct every other training image at correlation 0.47."

That is exactly what Algorithm 1 (`train_iqp_r3.py --protocol paper`) does: stage 1 fits all
1,024 phases to the FIRST training image, then the core is frozen and stage 2 fits only the
50 latent coordinates per remaining image (train_iqp_r3.py:481-519).  Its run directory
therefore already contains everything the claim needs -- `theta_core.npy` (974,) and
`theta_lat.npy` (N, 50) -- so this script needs no training of its own: it reads one or more
`iqp_paper_n*` run dirs, replays the frozen-core forward pass, and reports

    corr_anchor        the stage-1 image's own reconstruction correlation
    corr_others_mean   the mean over images 1..N-1  <- the paper's 0.47
    corr_others_sd/min/max

The seed comes from each run dir's meta.json, so running this over the five seeds' paper-
protocol run dirs gives the mean +- sd of the claim.  The data does not move between seeds:
data_lens_r3.load_r3 fixes its split at R3_SPLIT_SEED (data_lens_r3.py:265-266).

Provenance note.  The published 0.47 is the ROUND-2 diagnostic
(mech_tests/diag_r2/run_diag.py, 12-bit codec on the full 64x64 field, one shared S).  This
script is its round-3 counterpart: 10-bit codec on the 32x32 aligned crop with each image's
own S, i.e. the encoding every number in the paper's tables uses.  Both measure the same
thing -- how well a frozen core plus 50 free latent coordinates reproduces an image the core
was not fitted to -- so the seed spread computed here is the spread of that claim.

    python3 seeds/diag_frozen_core_seeds.py --run-dir outputs_lens_r3/iqp_paper_n11 \
        --results results_seeds_diag.jsonl

Rows are appended, one per run dir.  Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
sys.path.insert(0, REPO)

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

import codec10                                          # noqa: E402
import data_lens_r3                                     # noqa: E402
import latent_select_r3                                 # noqa: E402
import train_iqp_r3                                     # noqa: E402
from iqp_exact import IQPExact                          # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", nargs="+", required=True,
                   help="one or more iqp_paper_n* run directories")
    p.add_argument("--results", default=os.path.join(REPO, "results_seeds_diag.jsonl"))
    p.add_argument("--latent-dir", default=None)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a @ b) / (na * nb))


def score_run(run_dir, latent_dir=None):
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit("diag_frozen_core: no meta.json in %s" % run_dir)
    with open(meta_path) as fh:
        meta = json.load(fh)
    if meta.get("protocol") != "paper":
        raise SystemExit("diag_frozen_core: %s is protocol=%r, expected 'paper'"
                         % (run_dir, meta.get("protocol")))

    theta_core = np.load(os.path.join(run_dir, "theta_core.npy")).astype(np.float64)
    theta_lat = np.load(os.path.join(run_dir, "theta_lat.npy")).astype(np.float64)
    n = int(theta_lat.shape[0])

    d = data_lens_r3.load_r3(n, n_holdout=0)
    latent_masks = latent_select_r3.load_latent_masks(meta.get("latent", "lowvar"),
                                                      out_dir=latent_dir)
    sim = IQPExact(train_iqp_r3.N_QUBITS, train_iqp_r3.K_MAX, device="cpu",
                   dtype=torch.complex128)
    layout = train_iqp_r3.ParamLayout(sim, latent_masks)
    if layout.d_core != theta_core.size or layout.d_lat != theta_lat.shape[1]:
        raise SystemExit("diag_frozen_core: layout (%d core, %d lat) does not match the saved "
                         "arrays (%d, %d) in %s"
                         % (layout.d_core, layout.d_lat, theta_core.size,
                            theta_lat.shape[1], run_dir))

    core_t = torch.as_tensor(theta_core, dtype=sim.rdtype)
    lat_t = torch.as_tensor(theta_lat, dtype=sim.rdtype)
    with torch.no_grad():
        th = layout.assemble(core_t, lat_t, "cpu")
        q = train_iqp_r3.probs_batch(sim, th).to(torch.float64).numpy()

    cs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rec = codec10.pmf_to_image(q[i], float(d.train_totals[i])).astype(np.float64)
        cs[i] = corr(rec, np.asarray(d.train_crops[i], dtype=np.float64))

    others = cs[1:]
    row = {
        "diag": "frozen_core_recon",
        "script": "seeds/diag_frozen_core_seeds.py",
        "run_dir": os.path.abspath(run_dir),
        "arm": meta.get("arm"), "seed": meta.get("seed"), "n_train": n,
        "objective": meta.get("objective"), "latent": meta.get("latent"),
        "corr_anchor": float(cs[0]),
        "corr_others_mean": float(others.mean()) if others.size else None,
        "corr_others_sd": float(others.std(ddof=0)) if others.size else None,
        "corr_others_min": float(others.min()) if others.size else None,
        "corr_others_max": float(others.max()) if others.size else None,
        "corr_all_mean": float(cs.mean()),
        "n_others": int(others.size),
        "codec": "codec10 32x32 crop, per-image S",
    }
    print("[diag-b] %s  seed=%s N=%d  anchor %.4f | others mean %.4f sd %.4f "
          "[min %.4f, max %.4f]"
          % (os.path.basename(run_dir), row["seed"], n, row["corr_anchor"],
             row["corr_others_mean"], row["corr_others_sd"],
             row["corr_others_min"], row["corr_others_max"]), flush=True)
    return row


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    rows = [score_run(rd, args.latent_dir) for rd in args.run_dir]
    with open(args.results, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("[diag-b] appended %d rows to %s (%.1f s)"
          % (len(rows), args.results, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
