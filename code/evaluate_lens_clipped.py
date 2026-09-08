#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clipped sub-score for the DeepLense lens pilot / round 2 (docs/IQP_LENS_PILOT_PREREGISTRATION.md).

The round-2 pmf-encoding arm writes generated_{i:03d}.npy = q * S, where q is the
12-qubit-derived distribution over the 4,096 pixels and S is the training image's total
intensity (codec12.pmf_to_image; see lens_synth.write_pmf_pair). Because q need not
concentrate its mass exactly the way a real field does, q * S can push individual pixels
above the real images' range of [0, 1]. The pre-registration commits to also reporting a
second score where every GENERATED image is clipped to [0, 1] before scoring, so the
headline PDF-JS / MMD2 numbers are not confounded by "how far outside [0,1] did it go" as
well as "how well does the in-range shape match".

This script does NOT reimplement any scoring. It imports evaluate_lens and monkeypatches
its private `_generated` loader -- the only place raw generated_*.npy arrays are read --
to clip its return value, then delegates the entire scan/score/warn/write/print flow to
evaluate_lens.main() unchanged. Real images (held-out, training) are never touched: they
are already in [0, 1] by construction (data_lens), so clipping them would be a no-op
anyway, and the pre-registration only asks to clip the GENERATED side.

Effect: any arm whose generated images are already within [0, 1] -- every round-1 arm,
and round-2's non-pmf arms -- scores IDENTICALLY to plain evaluate_lens.py, because
np.clip(x, 0, 1) is a no-op when x is already in [0, 1]. Only pmf arms (IQP-pmf-*) are
expected to move.

CLI is identical to evaluate_lens.py (--lens-class, --outputs-dir, --runs, --n-holdout,
--n-train-max, --results), with one different default: --results defaults to
results_lens_round2_clipped_<class>.jsonl instead of results_lens_<class>.jsonl, so this
script's output never collides with the unclipped table.

Does not modify evaluate_lens.py (or any other existing file) -- the patch below rebinds
attributes on the imported module object in THIS process only.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import sys

import numpy as np

import evaluate_lens as _el

# --------------------------------------------------------------------------------------
# The one preprocessing step: clip every GENERATED array to [0, 1] before it reaches any
# metric. `evaluate_lens.evaluate_run` looks up the bare name `_generated` in
# evaluate_lens's own module globals at call time, so rebinding the attribute here
# changes what `evaluate_run` sees without touching evaluate_lens.py or copying its body.
# --------------------------------------------------------------------------------------
_orig_generated = _el._generated


def _generated_clipped(run_dir):
    gen = _orig_generated(run_dir)
    if gen is None:
        return None
    return np.clip(gen, 0.0, 1.0).astype(np.float32)


_el._generated = _generated_clipped

# Cosmetic-only second patch: label the printed table's header as clipped. render_table's
# `lens_class` argument is used ONLY to build the "### <label> -- ..." header line (row
# selection/order/metrics are untouched), so wrapping it cannot change any number.
_orig_render_table = _el.render_table


def _render_table_clipped(rows, lens_class):
    label = "%s [CLIPPED: generated images clipped to [0,1] before scoring]" % lens_class
    return _orig_render_table(rows, label)


_el.render_table = _render_table_clipped


def main(argv=None):
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # Parse once, only to read --lens-class and whether --results was already given.
    # evaluate_lens.main() parses argv again itself -- parsing twice is cheap and keeps
    # this script from having to know about (or duplicate) any other flag's semantics.
    args = _el.parse_args(argv)
    if not args.results:
        argv = argv + ["--results", "results_lens_round2_clipped_%s.jsonl" % args.lens_class]
    print("[evaluate_lens_clipped] preprocessing: every GENERATED image is "
          "np.clip(gen, 0.0, 1.0) before scoring; held-out/training images are untouched "
          "(already in [0,1]). Non-pmf arms therefore score IDENTICALLY to plain "
          "evaluate_lens.py -- clip is a no-op for them. Only pmf-encoded arms "
          "(q*scale can exceed 1 per pixel) are expected to change.")
    return _el.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
