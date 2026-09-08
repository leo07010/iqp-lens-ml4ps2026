"""Render the two controlled-study figures of arXiv 2603.06440 from the jsonl produced by
`fig3_support_mismatch.py` and `fig4_qcli_max.py`.

    docs/figures/iqp_fig3_support_mismatch.png
        one panel per learner gate count: achieved MMD^2 vs the target's I_QCLI, with the
        binning-and-spline upper/lower envelopes of Appendix D (`paper.txt:2487-2515`,
        bin width Delta = 0.09, upper = per-bin argmax, lower = per-bin argmin, both anchored
        at the selected point's own I_QCLI, then a UnivariateSpline of degree 2), plus a
        fourth panel overlaying the three upper envelopes -- that overlay is where the
        paper's second claim lives ("increasing G_n shifts the envelopes downward").

    docs/figures/iqp_fig4_qcli_cci.png
        I_CCI vs I_QCLI for every circuit sampled along the QCLI-only optimisation, coloured
        by system size, with per-size binned medians.  The left panel is clipped to the
        paper's I_QCLI range; the right panel shows the full sweep together with the
        exact-i.i.d.-uniform calibration point (I_QCLI ~ 1, I_CCI = 0), which is the ceiling
        of the indicator and the place where a monotone coupling has to break.

Design notes: categorical hues are the Okabe-Ito blue/vermillion/green triple in fixed order
(validated: all six checks pass on a light surface, worst adjacent CVD Delta-E 11.0), each
series additionally carries its own marker shape and a direct label on its trend curve, so
identity never rests on colour alone.  Single y-axis per panel; recessive grid.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
_FIGDIR = os.path.join(_REPO, "docs", "figures")

# fixed categorical order -- never cycled, never reassigned by rank
PALETTE = ["#0072B2", "#D55E00", "#009E73"]


def _legend(ax, **kw):
    """matplotlib warns (and draws an empty box) when no artist carries a label."""
    h, _l = ax.get_legend_handles_labels()
    if h:
        ax.legend(**kw)
MARKERS = ["o", "s", "^"]
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#dcdcdc"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Plot the Fig. 3 / Fig. 4 controlled studies.")
    p.add_argument("--fig3", default=None, help="results_fig3.jsonl (skip the figure if absent)")
    p.add_argument("--fig4", default=None, help="results_fig4.jsonl (skip the figure if absent)")
    p.add_argument("--out-fig3", default=os.path.join(_FIGDIR, "iqp_fig3_support_mismatch.png"))
    p.add_argument("--out-fig4", default=os.path.join(_FIGDIR, "iqp_fig4_qcli_cci.png"))
    p.add_argument("--x", choices=("exact", "sample"), default="exact",
                   help="I_QCLI of the exact target circuit pmf (default) or of the "
                        "10^4-sample dataset (the paper's literal protocol).  At n=16 with "
                        "M=10^4 over 2^16 outcomes the sample estimator sits below its own "
                        "i.i.d.-uniform noise floor for every target we can build, so the "
                        "exact value is the only axis on which the targets are separated.")
    p.add_argument("--yscale", choices=("log", "linear"), default="log")
    p.add_argument("--bin-width", type=float, default=0.09, help="Appendix D Delta")
    p.add_argument("--upper-smooth", type=float, default=0.05,
                   help="UnivariateSpline smoothing of the upper envelope (Appendix D: small)")
    p.add_argument("--lower-smooth", type=float, default=0.15,
                   help="smoothing of the lower envelope (Appendix D: stronger than the upper)")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args(argv)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# =======================================================================================
# Appendix D: binning-and-spline envelopes
# =======================================================================================
def envelope_points(x, y, delta, upper=True):
    """Per-bin argmax (upper) / argmin (lower) of y, anchored at the selected point's own x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return np.array([]), np.array([])
    lo = float(np.floor(x.min() / delta) * delta)
    hi = float(x.max()) + 1e-12
    ex, ey = [], []
    a = lo
    while a < hi:
        m = (x >= a) & (x < a + delta)
        if m.any():
            idx = np.argmax(y[m]) if upper else np.argmin(y[m])
            ex.append(x[m][idx])
            ey.append(y[m][idx])
        a += delta
    order = np.argsort(ex)
    return np.asarray(ex)[order], np.asarray(ey)[order]


def spline_curve(ex, ey, smooth_factor, k=2, n_out=200):
    """UnivariateSpline of degree k through the envelope points (Appendix D).  Returns
    (xs, ys) or (None, None) when there are too few bins for a spline of that degree."""
    if ex.size < k + 1:
        return None, None
    try:
        from scipy.interpolate import UnivariateSpline
    except Exception:                                     # scipy unavailable -> linear join
        return ex, ey
    # UnivariateSpline needs strictly increasing x; collapse duplicates by averaging.
    ux, inv = np.unique(ex, return_inverse=True)
    uy = np.zeros_like(ux)
    for i in range(ux.size):
        uy[i] = ey[inv == i].mean()
    if ux.size < k + 1:
        return ux, uy
    s = float(smooth_factor) * float(np.var(uy)) * ux.size
    try:
        sp = UnivariateSpline(ux, uy, k=k, s=s)
    except Exception:
        return ux, uy
    xs = np.linspace(ux.min(), ux.max(), n_out)
    # A smoothing spline is free to bow outside the envelope points it was fitted to, and a
    # strongly-smoothed LOWER envelope routinely arcs above the upper one, which is not just
    # ugly but false.  Clamp the curve to the band its own anchors define.
    return xs, np.clip(sp(xs), uy.min(), uy.max())


def draw_envelope(ax, x, y, delta, color, upper, logy, smooth):
    ex, ey = envelope_points(x, y, delta, upper=upper)
    if ex.size == 0:
        return
    fy = np.log10(np.clip(ey, 1e-300, None)) if logy else ey
    cx, cy = spline_curve(ex, fy, smooth)
    if cx is None:
        return
    yy = np.power(10.0, cy) if logy else cy
    ax.plot(cx, yy, ls="--", lw=2.0, color=color, zorder=4,
            label="upper envelope" if upper else "lower envelope")
    ax.plot(ex, ey, ls="none", marker="x", ms=5, color=color, alpha=0.8, zorder=4)


# =======================================================================================
# Fig. 3
# =======================================================================================
def plot_fig3(rows, args):
    fits = [r for r in rows if r.get("record") == "fit"]
    if not fits:
        print("[plot] no fit rows in the Fig. 3 jsonl -- skipping", flush=True)
        return False
    nulls = [r for r in rows if r.get("record") == "null"]
    q_null = float(nulls[0]["I_QCLI_uniform_mean"]) if nulls else None
    xkey = "I_QCLI" if args.x == "sample" else "I_QCLI_exact"
    xlab = ("$I_{\\mathrm{QCLI}}$ of the target dataset ($10^4$ samples)" if args.x == "sample"
            else "$I_{\\mathrm{QCLI}}$ of the target circuit (exact pmf)")
    learners = sorted({int(r["learner_gates"]) for r in fits})
    logy = (args.yscale == "log")
    xall = np.array([float(r[xkey]) for r in fits])

    fig, axes = plt.subplots(1, len(learners) + 1,
                             figsize=(4.1 * (len(learners) + 1), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    uppers = []
    for i, g in enumerate(learners):
        ax = axes[i]
        col = PALETTE[i % len(PALETTE)]
        sub = [r for r in fits if int(r["learner_gates"]) == g]
        x = np.array([float(r[xkey]) for r in sub])
        y = np.array([max(float(r["final_mmd2"]), 1e-16) for r in sub])
        ax.scatter(x, y, s=26, marker=MARKERS[i % len(MARKERS)], facecolor=col, alpha=0.55,
                   edgecolor="white", linewidth=0.6, zorder=3, label="targets")
        draw_envelope(ax, x, y, args.bin_width, col, True, logy,
                      smooth=float(args.upper_smooth))
        draw_envelope(ax, x, y, args.bin_width, INK_MUTED, False, logy,
                      smooth=float(args.lower_smooth))
        uppers.append((g, x, y, col, MARKERS[i % len(MARKERS)]))
        eff = int(sub[0].get("learner_gates_effective", g))
        ttl = "learner $G_n$ = %d" % g
        if eff != g:
            ttl += "  (%d exist)" % eff
        ax.set_title(ttl, fontsize=11, color=INK)
        ax.set_xlabel(xlab, fontsize=10, color=INK)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        if logy:
            ax.set_yscale("log")
        if q_null is not None and args.x == "sample":
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            ax.axvline(q_null, color=INK_MUTED, ls=":", lw=1.4, zorder=2)
            ax.text(q_null, 0.98, " i.i.d.-uniform null", fontsize=7.5, color=INK_MUTED,
                    va="top", ha="left", rotation=90, transform=tr, clip_on=True)
        _legend(ax, fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("achieved $\\mathrm{MMD}^2$ after training", fontsize=10, color=INK)

    ax = axes[-1]
    for i, (g, x, y, col, mk) in enumerate(uppers):
        ex, ey = envelope_points(x, y, args.bin_width, upper=True)
        fy = np.log10(np.clip(ey, 1e-300, None)) if logy else ey
        cx, cy = spline_curve(ex, fy, float(args.upper_smooth))
        if cx is None:
            continue
        yy = np.power(10.0, cy) if logy else cy
        ax.plot(cx, yy, lw=2.2, color=col, zorder=4, label="$G_n$ = %d" % g)
        ax.plot(ex, ey, ls="none", marker=mk, ms=6, color=col, alpha=0.85, zorder=4)
        # small deterministic vertical stagger, in the same order as the curves (learners are
        # plotted G_n-ascending, so the top curve gets the upward nudge): the three envelope ends
        # often sit within a few percent of each other and overlapping labels are worse than none.
        ax.annotate("$G_n$=%d" % g, xy=(cx[-1], yy[-1]), xytext=(5, (1 - i) * 10),
                    textcoords="offset points", fontsize=8.5, color=col, va="center")
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.22 * (x1 - x0))
    ax.set_title("upper envelopes overlaid", fontsize=11, color=INK)
    ax.set_xlabel(xlab, fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    if logy:
        ax.set_yscale("log")
    _legend(ax, fontsize=8, frameon=False, loc="lower left")

    # headroom for the direct labels on the overlay panel (the axes are shared)
    ylo, yhi = axes[0].get_ylim()
    axes[0].set_ylim(ylo, yhi * (2.5 if logy else 1.25))

    sub = ("targets span $I_{\\mathrm{QCLI}}$ = %.3f-%.3f on this axis" % (xall.min(), xall.max()))
    if args.x == "sample" and q_null is not None:
        sub += ";  i.i.d.-uniform null = %.3f" % q_null
    fig.suptitle("Fig. 3 replication: does the target's QCLI bound the achievable MMD?\n"
                 + sub, fontsize=12.5, color=INK, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_fig3)), exist_ok=True)
    fig.savefig(args.out_fig3, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print("[plot] wrote %s (%d targets x %d learners)"
          % (args.out_fig3, len(fits) // max(1, len(learners)), len(learners)), flush=True)
    return True


# =======================================================================================
# Fig. 4
# =======================================================================================
def _binned_median(x, y, nbins=18):
    if x.size == 0:
        return np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max() + 1e-12, nbins + 1)
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 3:
            cx.append(float(np.median(x[m])))
            cy.append(float(np.median(y[m])))
    return np.asarray(cx), np.asarray(cy)


def plot_fig4(rows, args):
    traj = [r for r in rows if r.get("record") == "traj"]
    if not traj:
        print("[plot] no traj rows in the Fig. 4 jsonl -- skipping", flush=True)
        return False
    calib = {int(r["n"]): r for r in rows if r.get("record") == "calibration"}
    sizes = sorted({int(r["n"]) for r in traj})

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for panel, (ax, clip) in enumerate(zip(axes, (0.7, None))):
        for i, n in enumerate(sizes):
            col = PALETTE[i % len(PALETTE)]
            mk = MARKERS[i % len(MARKERS)]
            sub = [r for r in traj if int(r["n"]) == n]
            x = np.array([float(r["I_QCLI"]) for r in sub])
            y = np.array([float(r["I_CCI"]) for r in sub])
            if clip is not None:
                keep = x <= clip
                x, y = x[keep], y[keep]
            if x.size == 0:
                continue
            ax.scatter(x, y, s=5, marker=mk, color=col, alpha=0.18, linewidths=0, zorder=2)
            cx, cy = _binned_median(x, y)
            if cx.size:
                ax.plot(cx, cy, lw=2.2, color=col, zorder=4, label="%d qubits (n = %d circuits)"
                        % (n, len(sub)))
                ax.annotate("%dQ" % n, xy=(cx[-1], cy[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=9, color=col, va="center")
            if clip is None and n in calib:
                ax.plot([float(calib[n]["I_QCLI"])], [float(calib[n]["I_CCI"])], marker="*",
                        ms=13, color=col, mec="white", mew=0.8, zorder=6)
        ax.set_xlabel("$I_{\\mathrm{QCLI}}$ (the only quantity optimised)", fontsize=10, color=INK)
        ax.set_ylabel("$I_{\\mathrm{CCI}}$ (never optimised)", fontsize=10, color=INK)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.03, 1.03)
        _legend(ax, fontsize=8, frameon=False, loc="lower left")
        if clip is not None:
            ax.set_title("paper's range ($I_{\\mathrm{QCLI}} \\leq 0.7$)", fontsize=11, color=INK)
        else:
            ax.set_title("full sweep; $\\star$ = exact i.i.d.-uniform (the QCLI ceiling)",
                         fontsize=11, color=INK)
    fig.suptitle("Fig. 4 replication: maximising QCLI alone -- does CCI follow?",
                 fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_fig4)), exist_ok=True)
    fig.savefig(args.out_fig4, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print("[plot] wrote %s (%d circuits)" % (args.out_fig4, len(traj)), flush=True)
    return True


def main(argv=None):
    args = parse_args(argv)
    did = False
    if args.fig3 and os.path.exists(args.fig3):
        did = plot_fig3(read_jsonl(args.fig3), args) or did
    elif args.fig3:
        print("[plot] %s not found -- skipping Fig. 3" % args.fig3, flush=True)
    if args.fig4 and os.path.exists(args.fig4):
        did = plot_fig4(read_jsonl(args.fig4), args) or did
    elif args.fig4:
        print("[plot] %s not found -- skipping Fig. 4" % args.fig4, flush=True)
    if not did:
        print("[plot] nothing plotted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
