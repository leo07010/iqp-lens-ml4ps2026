"""Decisive round. mech_test2 showed the ancilla latent only interpolates smoothly once a
short-path penalty mu*sum(1-cos(dlambda)) is added. So the penalty must be applied to the
CONTROLS as well, otherwise the success is attributable to the penalty rather than to the
ancilla. Adds the pure cross-fade null (1-t)p_A + t p_B as the yardstick, and draws the
final figure."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "<REPO_ROOT>/code"
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np
import torch

torch.set_num_threads(4)

import codec10
import mech_test as M
import mech_test2 as M2

RES = {}
MU = 0.03


def summarise(rows):
    cA = [r["corr_to"][0] for r in rows]
    cB = [r["corr_to"][1] for r in rows]
    rad = [r["radius"] for r in rows]
    e1 = [r["e1"] for r in rows]
    mono = lambda v, s: all(s * (v[i + 1] - v[i]) >= -1e-9 for i in range(len(v) - 1))
    return {"endpoint_corrA": cA[0], "endpoint_corrB": cB[-1],
            "min_max_corr": min(max(a, b) for a, b in zip(cA, cB)),
            "corrA_monotone_down": mono(cA, -1), "corrB_monotone_up": mono(cB, +1),
            "radius_monotone_down": mono(rad, -1), "e1_monotone_up": mono(e1, +1),
            "min_ring": min(r["ringiness"] for r in rows),
            "mean_corr_blend": float(np.mean([r["corr_blend"] for r in rows])),
            "min_corr_blend": float(np.min([r["corr_blend"] for r in rows]))}


def main():
    d, e1a, e2a, rada, ia, ib = M.load_pair()
    S = float(0.5 * (d.train_totals[ia] + d.train_totals[ib]))
    pA = np.asarray(d.train_pmfs[ia], dtype=np.float64)
    pB = np.asarray(d.train_pmfs[ib], dtype=np.float64)
    print("pair A=%d B=%d  corr(pA,pB)=%.4f" % (ia, ib, M.corr(pA, pB)), flush=True)
    RES["corr_pA_pB"] = M.corr(pA, pB)

    # ---- yardstick: the pure pixel cross-fade
    xf_rows, xf_imgs = [], []
    for i in range(11):
        t = i / 10.0
        q = (1 - t) * pA + t * pB
        img, qq = M.quants(q, S)
        xf_imgs.append(img)
        xf_rows.append({"label": "t=%.2f" % t, "corr_to": [M.corr(q, pA), M.corr(q, pB)],
                        "corr_blend": 1.0, "radius": qq["radius"], "e1": qq["e1"],
                        "e2": qq["e2"], "ringiness": M.ringiness(img)})
    RES["crossfade_null"] = {"rows": xf_rows, "summary": summarise(xf_rows)}
    print("\n=== cross-fade null (1-t)pA + t pB ===", flush=True)
    print("   t      corrA   corrB   radius   e1       ring", flush=True)
    for r in xf_rows:
        print("   %-6s %+.4f %+.4f  %6.3f  %+.4f %+.4f"
              % (r["label"], r["corr_to"][0], r["corr_to"][1], r["radius"], r["e1"],
                 r["ringiness"]), flush=True)

    jobs = [
        ("b_ancilla_mu0", 10, 1, M.latent_masks_ancilla(10, 1), 4000, 0.0),
        ("b_ancilla_mu.03", 10, 1, M.latent_masks_ancilla(10, 1), 4000, MU),
        ("b2_imageonly_mu.03", 10, 1, M.latent_masks_lowvar(10), 4000, MU),
        ("bp_noancilla_mu.03", 10, 0, M.latent_masks_lowvar(10), 4000, MU),
        ("c_k2_ancilla_mu.03", 10, 2, M.latent_masks_ancilla(10, 2), 5000, MU),
    ]
    fig_rows = [("cross-fade null (data)", xf_rows, xf_imgs)]
    for name, ni, na, lm, steps, mu in jobs:
        print("\n=== %s (n=%d latent=%d mu=%g) ===" % (name, ni + na, len(lm), mu), flush=True)
        o = M2.run(name, ni, na, lm, [pA, pB], S, steps=steps, mu=mu)
        rows = o["wrapped"]["rows"]
        s = summarise(rows)
        RES[name] = {"n": o["n"], "n_latent": o["n_latent"], "n_core": o["n_core"], "mu": mu,
                     "kl_endpoints": o["kl_endpoints"],
                     "max_abs_dlambda": o["max_abs_dlambda_wrapped"],
                     "l2_dlambda": o["l2_dlambda_wrapped"], "rows": rows, "summary": s}
        print("   t      corrA   corrB   corr_blend  radius   e1       e2      ring", flush=True)
        for r in rows:
            print("   %-6s %+.4f %+.4f  %+.4f     %6.3f  %+.4f %+.4f %+.4f"
                  % (r["label"], r["corr_to"][0], r["corr_to"][1], r["corr_blend"],
                     r["radius"], r["e1"], r["e2"], r["ringiness"]), flush=True)
        print("   summary: " + json.dumps(s, default=float), flush=True)
        fig_rows.append(("%s (mu=%g)" % (name, mu), rows, o["wrapped_imgs"]))

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tgtA = codec10.pmf_to_image(pA, S)
    tgtB = codec10.pmf_to_image(pB, S)
    vmax = max([float(np.max(im)) for _, _, ims in fig_rows for im in ims]
               + [float(tgtA.max()), float(tgtB.max())])
    nrow = len(fig_rows)
    fig, axes = plt.subplots(nrow, 11, figsize=(1.45 * 11, 1.95 * nrow))
    for ri, (title, rows, ims) in enumerate(fig_rows):
        for ci in range(11):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(ims[ci], cmap="inferno", vmin=0.0, vmax=vmax, origin="lower")
            r = rows[ci]
            ax.set_title("%s  A%.2f B%.2f" % (r["label"], r["corr_to"][0], r["corr_to"][1]),
                         fontsize=6.0)
            if ci == 0:
                ax.set_ylabel(title, fontsize=6.2)
    fig.suptitle("A(e1=%+.2f) -> B(e1=%+.2f) latent scan, t = 0 ... 1, identical colour scale. "
                 "Row 1 = pure pixel cross-fade of the two DATA images (the null)."
                 % (e1a[ia], e1a[ib]), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(HERE, "interp_final.png")
    fig.savefig(p, dpi=125)
    plt.close(fig)
    RES["figure"] = p
    with open(os.path.join(HERE, "results3.json"), "w") as fh:
        json.dump(RES, fh, indent=2, default=float)
    print("\nfigure: %s\nwrote results3.json" % p, flush=True)


if __name__ == "__main__":
    main()
