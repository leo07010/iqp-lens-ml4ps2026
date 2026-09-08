#!/usr/bin/env python3
"""CTRL-copy and CTRL-mean: the two no-model controls of the lens pilot.

docs/IQP_LENS_PILOT_PREREGISTRATION.md, Arms table:

  CTRL-copy : the N QUANTIZED training images, cycled to `--n-gen` outputs. A pure
              memorisation control -- prediction P2 is that this scores best or near-best
              on PDF-JS and Conv2D-MMD^2 in the N = 100 regime, i.e. that those two
              metrics reward copying the training set. Any generative arm that does not
              beat it has not demonstrated anything a lookup table could not.
  CTRL-mean : the mean of the N quantized training images, repeated `--n-gen` times. A
              "class-mean blur" control -- prediction P2/P3 is that it does well on
              Conv2D-MMD^2 and badly on PDF-JS, and that IQP's images end up close to it.

Both write the unconditional output contract (generated_{i:03d}.npy + meta.json), the
same one DCGAN and the gaussian-latent arms use, so evaluate_lens.py scores them
identically. Neither has any randomness beyond the fixed training-image draw, so `--seed`
only lands in meta.json.

The images written are the QUANTIZED fields (data_lens.Snapshot.field = the 6-bit-z
aggregate of the image's own 4,096-pixel code pool), not the raw continuous images: every
generative arm is capped at 6-bit z, so the controls must be too (the same audit fix that
controls.py's CTRL-interp and baseline_dcgan.py's training data carry).

Python 3.9 compatible (no `X | None`).
"""
import argparse
import json
import os
import time

import numpy as np

import data_lens


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CTRL-copy / CTRL-mean controls for the lens pilot.")
    p.add_argument("--arm", choices=["copy", "mean"], required=True)
    p.add_argument("--data", choices=["lens"], default="lens",
                   help="present for symmetry with the other arms; only 'lens' is meaningful")
    p.add_argument("--lens-class", type=str, required=True,
                   help="which DeepLense Model II class (axion/cdm/no_sub)")
    p.add_argument("--n-train", type=int, default=11, help="N training images (11 or 100)")
    p.add_argument("--lens-sqrt", action="store_true",
                   help="use the sqrt-intensity quantization (secondary arm); the written fields "
                        "are squared back to intensity space so they stay comparable")
    p.add_argument("--n-gen", type=int, default=200, help="number of images to write")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="default: outputs/ctrl_<arm>")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    out_dir = args.out if args.out else os.path.join("outputs", "ctrl_%s" % args.arm)
    os.makedirs(out_dir, exist_ok=True)

    names = data_lens.get_class_names()
    if args.lens_class not in names:
        raise SystemExit("--lens-class must be one of %s, got %r" % (names, args.lens_class))
    a_z, b_z = data_lens.get_z_range()

    train = data_lens.raw_training_images(args.lens_class, args.n_train, sqrt=args.lens_sqrt)
    n_train = int(train.shape[0])

    if args.arm == "copy":
        # cycle the N training images to n_gen outputs, in anchor order
        images = np.stack([train[i % n_train] for i in range(int(args.n_gen))], axis=0)
        arm_label = "CTRL-copy"
    else:
        mean_img = train.mean(axis=0)
        images = np.repeat(mean_img[None, :, :], int(args.n_gen), axis=0)
        arm_label = "CTRL-mean"

    if args.lens_sqrt:
        images = np.stack([data_lens.invert_intensity(f, sqrt=True) for f in images], axis=0)
        arm_label = arm_label + "-sqrt"

    for i in range(images.shape[0]):
        np.save(os.path.join(out_dir, "generated_{:03d}.npy".format(i)),
                images[i].astype(np.float32))

    meta = {
        "arm": arm_label,
        "param_count": 0,
        "seed": int(args.seed),
        "data": "lens",
        "class_names": names,
        "lens_class": args.lens_class,
        "lens_sqrt": bool(args.lens_sqrt),
        "n_train": n_train,
        "train_snapshots": n_train,
        "anchors": list(range(1, n_train + 1)),
        "n_gen": int(images.shape[0]),
        "a_z": a_z,
        "b_z": b_z,
        "synth_mode": args.arm,
        "aggregation": "n/a",
        "wallclock_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("[%s] wrote %d images to %s (n_train=%d, class=%s)"
          % (arm_label, images.shape[0], out_dir, n_train, args.lens_class))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
