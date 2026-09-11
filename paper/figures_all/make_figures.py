#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build every ML4PS-2026 figure.  Usage:  python3 make_figures.py [stem ...]

With no arguments it rebuilds all of them; otherwise only the named stems.

The three `*_seeds` stems are the 5-seed versions of fig2_compact, figA_fig3_c and
figA_fig4_c.  They read the seed aggregates (mech_test2/results_seeds.json,
outputs_controlled/results_fig3_v2_ls*.jsonl, outputs_controlled/results_fig4_seeds34.jsonl)
and will exit with a clear message if those have not been produced yet, so a plain
`python3 make_figures.py` on a fresh checkout still rebuilds the 1-seed figures.
Each figure is written as <stem>.pdf (vector) and <stem>.png (300 dpi) into this directory.
CPU only, seconds per figure.  Python 3.9 compatible.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fig_pipeline                # noqa: E402
import fig_pipeline_v2             # noqa: E402
import fig_pipeline_v2_c           # noqa: E402
import fig1_map_samples            # noqa: E402
import fig1_compact                # noqa: E402
import fig2_mechanism_scaling      # noqa: E402
import fig2_compact                # noqa: E402
import fig2_compact_seeds          # noqa: E402
import figA_scaling                # noqa: E402
import figA_scaling_seeds          # noqa: E402
import figA_convergence            # noqa: E402
import figA_fig4                   # noqa: E402
import figA_fig4_c               # noqa: E402
import figA_fig4_c_seeds         # noqa: E402
import figA_fig3                   # noqa: E402
import figA_fig3_c               # noqa: E402
import figA_fig3_c_seeds         # noqa: E402
import figB_classes                # noqa: E402
import figB_classes_c            # noqa: E402

BUILDERS = [
    ("fig_pipeline",            fig_pipeline.make),
    ("fig_pipeline_v2",         fig_pipeline_v2.make),
    ("fig_pipeline_v2_c",       fig_pipeline_v2_c.make),
    ("fig1_map_samples",        fig1_map_samples.make),
    ("fig1_compact",            fig1_compact.make),
    ("fig2_mechanism_scaling",  fig2_mechanism_scaling.make),
    ("fig2_compact",            fig2_compact.make),
    ("fig2_compact_seeds",      fig2_compact_seeds.make),
    ("figA_scaling",            figA_scaling.make),
    ("figA_scaling_seeds",      figA_scaling_seeds.make),
    ("figA_convergence",        figA_convergence.make),
    ("figA_fig3",               figA_fig3.make),
    ("figA_fig3_c",           figA_fig3_c.make),
    ("figA_fig3_c_seeds",     figA_fig3_c_seeds.make),
    ("figA_fig4",               figA_fig4.make),
    ("figA_fig4_c",           figA_fig4_c.make),
    ("figA_fig4_c_seeds",     figA_fig4_c_seeds.make),
    ("figB_classes",            figB_classes.make),
    ("figB_classes_c",        figB_classes_c.make),
]


def main(argv):
    want = set(argv[1:])
    for stem, fn in BUILDERS:
        if want and stem not in want:
            continue
        fn(HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
