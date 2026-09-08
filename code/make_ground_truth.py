#!/usr/bin/env python3
"""Write the ground-truth fields every arm's evaluate.py comparison reads from.

INTERFACES.md: `outputs/ground_truth/gt_t{tau}.npy` (64, 64) float32, one per held-out tau.
`tau` and the a_z/b_z quantization bounds come from the SAME single sources every model
arm uses (synth_data.make_split_labels / synth_data.anchor_a_z_b_z for --data synthetic;
data_d2.load_d2()'s own pre-fixed split for --data d2), so the file names line up exactly
with what train_iqp.py / baseline_rbm.py / controls.py write, and evaluate.py can just
glob-match by tau.

The written field is the QUANTIZED ground truth (code-pool aggregate, i.e. z rounded to
the 6-bit bin every generative arm is capped at -- see synth_data.pool_and_quantized_field
and the audit fix in controls.py/baseline_dcgan.py), NOT the raw continuous field: a model
that can only ever emit 64 distinct z levels should not be scored against ground truth at
infinite precision.

Python 3.9 compatible (no `X | None`).
"""
import argparse
import json
import os

import numpy as np

import codec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Write outputs/ground_truth/gt_t{tau}.npy for evaluate.py.")
    p.add_argument("--data", choices=("synthetic", "d2"), default="synthetic")
    p.add_argument("--out", type=str, default="outputs/ground_truth")
    p.add_argument("--anchors-max", type=int, default=0, help=">0: use only the first N anchors (smoke tests)")
    p.add_argument("--n-holdout", type=int, default=200, help="number of held-out tau to write")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.data == "synthetic":
        import synth_data
        all_snaps = synth_data.make_synthetic_snapshots(synth_data.N_SNAPSHOTS_SYNTH, seed=synth_data.SYNTH_DATA_SEED)
        anchors, unseen = synth_data.make_split_labels(
            all_snaps.shape[0], seed=synth_data.SYNTH_DATA_SEED, n_holdout=args.n_holdout,
            anchors_max=args.anchors_max if args.anchors_max > 0 else None,
        )
        a_z, b_z = synth_data.anchor_a_z_b_z(all_snaps, anchors)

        def raw_field(t):
            return all_snaps[int(t) - 1]

        def quantized(t):
            return synth_data.pool_and_quantized_field(raw_field(t), a_z, b_z)[1]
    else:
        try:
            from data_d2 import load_d2, get_z_range
        except Exception as exc:  # pragma: no cover - depends on a file we do not own
            raise SystemExit("--data d2 needs data_d2.py with load_d2(); not available yet (%s)" % exc)
        snapshots, anchors, unseen = load_d2()
        anchors = [int(a) for a in anchors]
        unseen = [int(t) for t in unseen]
        if args.anchors_max > 0:
            anchors = anchors[: args.anchors_max]
        if args.n_holdout > 0:
            unseen = unseen[: args.n_holdout]
        # a_z/b_z MUST be the exact bounds load_d2() used to build snap.field
        # (data_d2.get_z_range, native z_val over the 11 anchors) -- NOT re-derived from
        # `.field`'s own min/max, which is narrower (~18.8% here) since `.field` is already
        # a 64x64-bin aggregate that averages away the extremes.
        a_z, b_z = get_z_range("mid")

        def quantized(t):
            # snap.field is ALREADY codec.aggregate(pool_codes, a_z, b_z) computed by
            # data_d2._build_snapshot with these exact bounds (data_d2.py docstring) --
            # re-quantizing/re-aggregating here would be redundant, so write it out as-is.
            return np.asarray(snapshots[int(t)].field, dtype=np.float64)

    anchors = [int(a) for a in (anchors.tolist() if hasattr(anchors, "tolist") else anchors)]
    unseen = [int(t) for t in (unseen.tolist() if hasattr(unseen, "tolist") else unseen)]

    n_written = 0
    for tau in unseen:
        field = quantized(tau).astype(np.float32)
        np.save(os.path.join(args.out, "gt_t{}.npy".format(tau)), field)
        n_written += 1

    meta = {
        "data": args.data,
        "anchors": anchors,
        "unseen": unseen,
        "n_holdout": n_written,
        "a_z": a_z,
        "b_z": b_z,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("[ground_truth] wrote %d fields to %s (a_z=%.6g b_z=%.6g)" % (n_written, args.out, a_z, b_z))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
