"""Figures for docs/LENS_IMAGE_CHARACTERISTICS.md.

Palette: validated categorical slots (blue/orange/aqua/yellow) from the
dataviz reference palette; diverging = blue<->gray<->orange; sequential = blue ramp.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lens_common import fwht, popcount_table, layout_index, image_to_pmf12

SCRATCH = os.environ.get("LENS_SCRATCH", "/tmp")
FIG = "<REPO_ROOT>/docs/figures"
CLASSES = ["axion", "cdm", "no_sub"]

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#2a78d6", "#e8e7e0",
                                                "#eb6834", "#a33c14"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec",
                                                "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "lines.linewidth": 2.0, "axes.titlelocation": "left",
})

C = np.load(os.path.join(HERE, "intermediates.npz"))
R = json.load(open(os.path.join(HERE, "results.json")))
RB = json.load(open(os.path.join(HERE, "results_bits.json")))
val_sub = np.load(os.path.join(SCRATCH, "val_sub.npy"))


def finish(fig, path):
    fig.tight_layout(pad=1.6)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, os.path.getsize(path) // 1024, "KB")


# =====================================================================
# Figure 1 - intensity
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2))

a = ax[0, 0]
h, e = C["n1_hist"], C["n1_edges"]
ctr = 0.5 * (e[1:] + e[:-1])
a.step(ctr, np.maximum(h, 0.5), where="mid", color=S[0])
a.fill_between(ctr, 0.5, np.maximum(h, 0.5), step="mid", color=S[0], alpha=0.18)
a.set_yscale("log")
for thr, lab, yf in [(1 / 64, "1/64", 0.30), (0.1, "0.1", 0.030)]:
    a.axvline(thr, color=MUTED, ls="--", lw=1.2)
    a.text(thr, h.max() * yf, "  z=" + lab, color=INK2, fontsize=8)
ps = R["n1_intensity"]["pixel_stats"]
a.set_title("(a) Pixel intensity, all 8910 val images (36.5M px)")
a.set_xlabel("intensity z"); a.set_ylabel("pixel count (log)")
a.text(0.97, 0.93, "median %.1e\nmean %.3f\n%.1f%% of px > 1/64" % (
    ps["median"], ps["mean"], 100 * ps["frac_gt_1_64"]),
    transform=a.transAxes, ha="right", va="top", fontsize=8, color=INK2)

a = ax[0, 1]
names = ["raw", "sqrt", "log1p100", "quantile"]
lab = {"raw": "raw  $64z$", "sqrt": "sqrt  $64\\sqrt{z}$",
       "log1p100": "log1p  $64\\,\\log(1{+}100z)/\\log 101$", "quantile": "quantile (equal-freq)"}
for i, n in enumerate(names):
    a.plot(np.arange(64), np.maximum(C["n1_occ"][i], 1e-8), color=S[i],
           label="%s   H=%.2f b" % (lab[n], R["n1_intensity"]["quantizers"][n]["entropy_bits"]))
a.set_yscale("log"); a.set_ylim(3e-5, 3)
a.set_title("(b) Occupancy of the 64 levels under four 6-bit quantizers")
a.set_xlabel("6-bit level $k_z$"); a.set_ylabel("fraction of pixels (log)")
a.legend(fontsize=7.5, loc="upper right")

a = ax[1, 0]
H = [R["n1_intensity"]["quantizers"][n]["entropy_bits"] for n in names]
bars = a.bar(range(4), H, color=S[:4], width=0.62)
a.axhline(6, color=MUTED, ls="--", lw=1.2)
a.text(-0.42, 6.12, "6 bits available", color=INK2, fontsize=8, ha="left")
for i, v in enumerate(H):
    a.text(i, v + 0.12, "%.2f b\n($2^H$=%.1f lv)" % (v, 2 ** v), ha="center",
           fontsize=8, color=INK)
a.set_xticks(range(4)); a.set_xticklabels(["raw", "sqrt", "log1p", "quantile"])
a.set_ylim(0, 7.2); a.set_ylabel("entropy of $k_z$ (bits)")
a.set_title("(c) Bits actually used by a 6-qubit intensity register")

a = ax[1, 1]
for i, cn in enumerate(CLASSES):
    fr = (val_sub[i].reshape(200, -1) > 1 / 64).mean(1)
    a.hist(fr, bins=28, histtype="step", lw=2, color=S[i], label="%s  %.3f$\\pm$%.3f" %
           (cn, fr.mean(), fr.std()))
pr = R["n2_walsh12"]["participation_ratio"]["cdm"]["frac_of_4096"]
a.axvline(pr, color=MUTED, ls="--", lw=1.2)
a.text(pr, a.get_ylim()[1] * 0.40, "  participation\n  ratio $1/\\|p\\|^2$\n  = 8.1%",
       color=INK2, fontsize=8, va="top")
a.set_title("(d) Per-image fraction of pixels above 1/64  (200 val img / class)")
a.set_xlabel("active fraction"); a.set_ylabel("images")
a.legend(fontsize=8, loc="upper center")
finish(fig, os.path.join(FIG, "lens_characteristics_intensity.png"))

# =====================================================================
# Figure 2 - Walsh
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2))
k12 = np.arange(13)

a = ax[0, 0]
for i, cn in enumerate(CLASSES):
    m = C["n2_os_per"][i].mean(0); sd = C["n2_os_per"][i].std(0)
    a.plot(k12, m, color=S[i], marker="o", ms=4, label=cn)
    a.fill_between(k12, m - sd, m + sd, color=S[i], alpha=0.15)
a.plot(k12, R["n2_walsh12"]["class_mean_image"]["cdm"], color=INK2, ls=":", lw=1.8,
       marker="s", ms=3.5, label="class-mean image (cdm)")
a.set_title("(a) 12-bit image-as-pmf Walsh order spectrum ($\\pm$1 sd)")
a.set_xlabel("Hamming weight $|s|$"); a.set_ylabel("fraction of Walsh energy")
a.legend(fontsize=8)
a.text(0.60, 0.30, "odd orders vanish for the\nclass-mean image\n(centro-symmetry)",
       transform=a.transAxes, ha="left", fontsize=7.5, color=INK2)

a = ax[0, 1]
lays = ["standard", "morton", "gray"]
sty = [dict(color=S[0], marker="o", ms=4),
       dict(color=S[1], marker="x", ms=7, ls="--", lw=1.6),
       dict(color=S[2], marker="D", ms=4)]
for i, L in enumerate(lays):
    a.plot(k12, C["n7_os"][i], label="%s   mean$|s|$=%.2f" %
           (L, R["n7_layout"][L]["mean_|s|"]), **sty[i])
a.set_title("(b) Bit layout: Morton $\\equiv$ standard exactly; Gray lowers $|s|$")
a.set_xlabel("Hamming weight $|s|$"); a.set_ylabel("fraction of Walsh energy")
a.legend(fontsize=8)

a = ax[1, 0]
cum = C["n2_cum_pooled"]
a.plot(np.arange(1, 4097), cum, color=S[0], label="data-adaptive (energy-ranked)")
from math import comb
for j, (L, col) in enumerate([("standard", S[1]), ("gray", S[2])]):
    sp = np.array(R["n7_layout"][L]["order_spectrum_mean"])
    xs = np.cumsum([comb(12, k) for k in range(13)])
    a.step(xs, np.cumsum(sp), where="post", color=col, ls="--",
           label="Hamming-ordered, %s layout" % L)
for M in [50, 500, 1000, 2510]:
    a.plot([M], [cum[M - 1]], "o", color=INK, ms=5, zorder=5)
    dx, dy = (6, -16) if M != 2510 else (4, -22)
    ha = "left"
    a.annotate("M=%d\n%.3f" % (M, cum[M - 1]), (M, cum[M - 1]), ha=ha,
               textcoords="offset points", xytext=(dx, dy), fontsize=7.5, color=INK)
a.set_xscale("log"); a.set_xlim(20, 4300); a.set_ylim(0.55, 1.02)
a.set_title("(c) Cumulative Walsh energy vs number of generators (pooled)")
a.set_xlabel("number of subsets $M$ retained (log)"); a.set_ylabel("cumulative energy fraction")
a.legend(fontsize=8, loc="lower right")

a = ax[1, 1]
k18 = np.arange(19)
keys = list(C["n8_os18_keys"])
raw = np.mean([C["n8_os18"][keys.index("raw_%s" % c)] for c in CLASSES], 0)
sq = np.mean([C["n8_os18"][keys.index("sqrt_%s" % c)] for c in CLASSES], 0)
a.plot(k18, raw, color=S[1], marker="o", ms=4, label="18-bit tuple, raw   mean$|s|$=%.2f" %
       np.mean([R["n8_tuple18"]["raw_%s" % c]["mean_|s|"] for c in CLASSES]))
a.plot(k18, sq, color=S[3], marker="^", ms=4, label="18-bit tuple, sqrt   mean$|s|$=%.2f" %
       np.mean([R["n8_tuple18"]["sqrt_%s" % c]["mean_|s|"] for c in CLASSES]))
a.plot(k12, C["n7_os"][0], color=S[0], marker="o", ms=4,
       label="12-bit image-as-pmf   mean$|s|$=%.2f" % R["n7_layout"]["standard"]["mean_|s|"])
a.axvline(6, color=MUTED, ls="--", lw=1.2)
a.text(6.15, 0.245, "paper's cut $|s|\\leq 6$\n31,180 gen. (18b) vs\n2,510 (12b)",
       fontsize=7.5, color=INK2)
a.set_title("(d) Tuple encoding pushes energy to higher order")
a.set_xlabel("Hamming weight $|s|$"); a.set_ylabel("fraction of Walsh energy")
a.legend(fontsize=7.5)
finish(fig, os.path.join(FIG, "lens_characteristics_walsh.png"))

# =====================================================================
# Figure 3 - PCA
# =====================================================================
RS = json.load(open(os.path.join(HERE, "results_pc_semantics.json")))
fig = plt.figure(figsize=(13.2, 6.6))
gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.22)
mu = C["n4_mu_pooled"]; pcs = C["n4_pc_pooled"]; ev = C["n4_ev_pooled"]

a = fig.add_subplot(gs[0, 0])
im = a.imshow(mu, cmap=SEQ)
a.set_title("mean image\n6000 train images", fontsize=9)
a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)

# each PC named by the geometry feature it correlates with (see results_pc_semantics.json)
desc = [("x-position of the lens", "centroid_x", "PC1"),
        ("total flux / ring size", "total_flux", "PC2"),
        ("y-position of the lens", "centroid_y", "PC3"),
        ("ellipticity $\\sin 2\\theta$", "ellip_sin2t", "PC4"),
        ("ellipticity $\\cos 2\\theta$", "ellip_cos2t", "PC5"),
        ("substructure residual", None, "PC6")]
cells = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
for i, (r, c) in enumerate(cells):
    a = fig.add_subplot(gs[r, c])
    v = np.abs(pcs[i]).max()
    a.imshow(pcs[i], cmap=DIV, vmin=-v, vmax=v)
    name, feat, key = desc[i]
    if feat is None:
        sub = "no geometry match ($R^2$=%.2f)" % RS["pc_R2_from_6_geometry_features"][key]
    else:
        sub = "$r$=%+.2f with %s" % (RS["pc_vs_geometry_corr"][key][feat],
                                     feat.replace("_", " "))
    a.set_title("PC%d  %.1f%% - %s\n%s" % (i + 1, 100 * ev[i] / ev.sum(), name, sub),
                fontsize=8.2)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)

a = fig.add_subplot(gs[1, 3])
cs = np.cumsum(ev) / ev.sum()
a.plot(np.arange(1, len(cs) + 1), cs, color=S[0])
for f, col, dy in [(0.90, S[1], -15), (0.95, S[2], -15), (0.99, S[3], 7)]:
    n = int(np.searchsorted(cs, f) + 1)
    a.axhline(f, color=col, ls="--", lw=1.0, alpha=0.7)
    a.plot([n], [cs[n - 1]], "o", color=col, ms=6, zorder=5)
    a.annotate("%d%%: %d PCs" % (int(f * 100), n), (n, cs[n - 1]),
               textcoords="offset points", xytext=(9, dy), fontsize=8, color=col)
a.set_xscale("log"); a.set_xlim(1, 300); a.set_ylim(0.25, 1.04)
a.set_title("explained variance (pooled)\ntop 6 PCs = 89.4%", fontsize=9)
a.set_xlabel("number of PCs (log)"); a.set_ylabel("cumulative fraction")
finish(fig, os.path.join(FIG, "lens_characteristics_pca.png"))

# =====================================================================
# Figure 4 - geometry
# =====================================================================
fig, ax = plt.subplots(2, 3, figsize=(13, 7.0))
a = ax[0, 0]
for i, cn in enumerate(CLASSES):
    a.plot(np.arange(45), C["n6_radial"][i], color=S[i], label=cn)
a.axvline(8, color=MUTED, ls="--", lw=1.2)
a.text(9.0, C["n6_radial"].max() * 0.62, "peak r$\\approx$8 px\n(Einstein radius)",
       fontsize=8, color=INK2)
a.set_title("(a) Radial profile about the image centre")
a.set_xlabel("radius r (px)"); a.set_ylabel("mean intensity")
a.legend(fontsize=8); a.set_xlim(0, 44)

a = ax[0, 1]
ang = np.arange(36) * 10 - 180 + 5
rng = np.random.RandomState(0)
yy, xx = np.mgrid[0:64, 0:64]
xr, yr = xx.ravel(), yy.ravel()
rr = np.sqrt((xr - 31.5) ** 2 + (yr - 31.5) ** 2)
th = np.arctan2(yr - 31.5, xr - 31.5)
tb = ((th + np.pi) / (2 * np.pi) * 36).astype(int) % 36
ann = (rr >= 5) & (rr <= 25)
for i, cn in enumerate(CLASSES):
    a.plot(ang, C["n6_azim"][i], color=S[i], label="%s  (split-half $r$=%.2f)" %
           (cn, RS["azimuthal_split_half"][cn]["split_half_corr"]))
    im200 = val_sub[i].reshape(200, -1).astype(np.float64)
    perm = rng.permutation(200)
    for half in (perm[:100], perm[100:]):
        m = im200[half].mean(0)
        pr_ = np.bincount(tb[ann], weights=m[ann], minlength=36) / \
            np.maximum(np.bincount(tb[ann], minlength=36), 1)
        a.plot(ang, pr_, color=S[i], lw=0.9, alpha=0.42)
a.set_title("(b) Azimuthal profile, $5\\leq r\\leq 25$ (thin = 100-image halves)")
a.set_xlabel("angle (deg)"); a.set_ylabel("mean intensity")
a.legend(fontsize=7.5, loc="lower center"); a.set_xticks([-180, -90, 0, 90, 180])
a.text(0.02, 0.97, "the two halves disagree:\nthe ensemble is ~isotropic",
       transform=a.transAxes, fontsize=7.5, color=INK2, va="top")

a = ax[0, 2]
for i, cn in enumerate(CLASSES):
    w_ = val_sub[i].reshape(200, -1).astype(np.float64)
    t_ = w_.sum(1)
    cx_ = (w_ * xr).sum(1) / t_; cy_ = (w_ * yr).sum(1) / t_
    off = np.sqrt((cx_ - 31.5) ** 2 + (cy_ - 31.5) ** 2)
    a.hist(off, bins=24, histtype="step", lw=2, color=S[i],
           label="%s  %.2f$\\pm$%.2f px" % (cn, off.mean(), off.std()))
a.set_title("(c) Light-centroid offset from the image centre")
a.set_xlabel("offset (px)"); a.set_ylabel("images")
a.set_ylim(0, 23)
a.legend(fontsize=8, loc="upper right")
a.text(0.03, 0.97, "the lens is NOT centred:\nPC1+PC3 (50% of variance)\nare exactly these two shifts",
       transform=a.transAxes, fontsize=7.5, color=INK2, va="top")

vmax = C["n6_meanimg"].max()
for i, cn in enumerate(CLASSES):
    a = ax[1, i]
    im = a.imshow(C["n6_meanimg"][i], cmap=SEQ, vmin=0, vmax=vmax)
    a.set_title("(d%d) mean image, %s (200 val)" % (i + 1, cn), fontsize=9)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
ax[1, 1].set_xlabel("class means are visually identical - nearest-centroid gets only 38%",
                    fontsize=8, color=INK2)
finish(fig, os.path.join(FIG, "lens_characteristics_geometry.png"))
print("all figures done")
