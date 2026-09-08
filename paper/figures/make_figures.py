#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build every ML4PS-2026 figure.  Usage:  python3 make_figures.py [stem ...]

With no arguments it rebuilds all five; otherwise only the named stems.
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
import fig1_map_samples            # noqa: E402
import fig2_mechanism_scaling      # noqa: E402
import figA_fig4                   # noqa: E402
import figA_fig3                   # noqa: E402

BUILDERS = [
    ("fig_pipeline",            fig_pipeline.make),
    ("fig_pipeline_v2",         fig_pipeline_v2.make),
    ("fig1_map_samples",        fig1_map_samples.make),
    ("fig2_mechanism_scaling",  fig2_mechanism_scaling.make),
    ("figA_fig3",               figA_fig3.make),
    ("figA_fig4",               figA_fig4.make),
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
