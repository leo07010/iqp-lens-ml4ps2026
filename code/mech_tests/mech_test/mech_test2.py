"""Follow-up to mech_test.py: (b) failed, so (per the spec) try the k=2 rescue, and first
rule out two confounds that could have caused the failure for reasons unrelated to the
hypothesis:

  W1  phase wrap. Every theta_s is EXACTLY 2*pi-periodic (only exp(i phi) matters and
      chi_s(y) = +-1), so lambda_B and lambda_B + 2*pi*k give the identical distribution but
      a completely different straight-line path. Wrapping (lambda_B - lambda_A) into
      [-pi, pi] keeps both endpoints and takes the SHORTEST path -- strictly a fairer test.
  W2  path length. The unconstrained joint fit has no reason to place lambda_A and lambda_B
      near each other. Add mu * sum_j (1 - cos(dLambda_j)) to the joint loss to ask for a
      short path, then re-scan.
"""
from __future__ import annotations

import json
import math
import os
import sys

REPO = "<REPO_ROOT>/code"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

torch.set_num_threads(4)

import codec10
import mech_test as M

OUT = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.float64
RES = {}


def wrap(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def run(name, n_img, n_anc, lat_masks, pmfs, S, steps=4000, mu=0.0, lrs=(0.03, 0.012), n_t=11):
    n = n_img + n_anc
    sim = M.IQPFull(n)
    P = sim.size
    lat_idx = torch.as_tensor(np.asarray(lat_masks, dtype=np.int64))
    cm = np.ones(P, dtype=bool)
    cm[np.asarray(lat_masks, dtype=np.int64)] = False
    core_idx = torch.as_tensor(np.nonzero(cm)[0])
    B = len(pmfs)
    p_t = torch.as_tensor(np.stack(pmfs), dtype=DTYPE)

    def assemble(core, lam, nb):
        base = torch.zeros(nb, P, dtype=DTYPE)
        base = base.index_copy(1, core_idx, core.unsqueeze(0).expand(nb, -1))
        base = base.index_copy(1, lat_idx, lam)
        return base

    def fwd(core, lam, nb):
        q = sim.probs(assemble(core, lam, nb))
        return M.marginal(q, n_img) if n_anc > 0 else q

    def loss_fn(ps):
        core, lam = ps
        L = M.kl_from_ce(p_t, fwd(core, lam, B)).sum()
        if mu > 0:
            d = lam[1] - lam[0]
            L = L + mu * (1.0 - torch.cos(d)).sum()
        return L

    best_v, best_state = float("inf"), None
    for r, lr in enumerate(lrs):
        g = torch.Generator().manual_seed(200 + r)
        core0 = torch.randn(core_idx.numel(), generator=g, dtype=DTYPE) * 0.05
        lam0 = torch.randn(B, lat_idx.numel(), generator=g, dtype=DTYPE) * 0.5
        v, st = M.fit(loss_fn, [core0, lam0], steps, lr, seed=r)
        if v < best_v:
            best_v, best_state = v, st
    core, lam = best_state

    with torch.no_grad():
        kl_end = M.kl_from_ce(p_t, fwd(core, lam, B)).tolist()
    lam_np = lam.numpy()
    d_raw = lam_np[1] - lam_np[0]
    d_wrap = wrap(d_raw)

    out = {"name": name, "n": n, "n_latent": int(lat_idx.numel()),
           "n_core": int(core_idx.numel()), "mu": mu, "loss": best_v,
           "kl_endpoints": kl_end,
           "max_abs_dlambda_raw": float(np.abs(d_raw).max()),
           "max_abs_dlambda_wrapped": float(np.abs(d_wrap).max()),
           "l2_dlambda_raw": float(np.linalg.norm(d_raw)),
           "l2_dlambda_wrapped": float(np.linalg.norm(d_wrap))}

    for mode, delta in (("raw", d_raw), ("wrapped", d_wrap)):
        rows, imgs = [], []
        for i in range(n_t):
            t = float(i) / (n_t - 1)
            l_vec = torch.as_tensor(lam_np[0] + t * delta, dtype=DTYPE)
            with torch.no_grad():
                q = fwd(core, l_vec.unsqueeze(0), 1)[0].numpy()
            img, qq = M.quants(q, S)
            imgs.append(img)
            blend = (1 - t) * pmfs[0] + t * pmfs[1]
            rows.append({"label": "t=%.2f" % t, "corr_to": [M.corr(q, pmfs[0]), M.corr(q, pmfs[1])],
                         "corr_blend": M.corr(q, blend), "radius": qq["radius"], "e1": qq["e1"],
                         "e2": qq["e2"], "ringiness": M.ringiness(img)})
        out[mode] = {"rows": rows}
        out[mode + "_imgs"] = imgs
    return out


def show(o, mode):
    rows = o[mode]["rows"]
    cA = [r["corr_to"][0] for r in rows]
    cB = [r["corr_to"][1] for r in rows]
    floor = min(max(a, b) for a, b in zip(cA, cB))
    print("\n  [%s / %s]  latent=%d core=%d mu=%g  |dlam|inf raw %.2f wrapped %.2f"
          % (o["name"], mode, o["n_latent"], o["n_core"], o["mu"],
             o["max_abs_dlambda_raw"], o["max_abs_dlambda_wrapped"]), flush=True)
    print("   t      corrA   corrB   corr_blend  radius   e1       e2      ring", flush=True)
    for r in rows:
        print("   %-6s %+.4f %+.4f  %+.4f     %6.3f  %+.4f %+.4f %+.4f"
              % (r["label"], r["corr_to"][0], r["corr_to"][1], r["corr_blend"],
                 r["radius"], r["e1"], r["e2"], r["ringiness"]), flush=True)
    print("   -> min over t of max(corrA,corrB) = %.4f   min ring = %.4f"
          % (floor, min(r["ringiness"] for r in rows)), flush=True)
    return {"min_max_corr": floor, "min_ring": float(min(r["ringiness"] for r in rows)),
            "endpoint_corrA": cA[0], "endpoint_corrB": cB[-1]}


def main():
    d, e1, e2, rad, ia, ib = M.load_pair()
    S = float(0.5 * (d.train_totals[ia] + d.train_totals[ib]))
    pA = np.asarray(d.train_pmfs[ia], dtype=np.float64)
    pB = np.asarray(d.train_pmfs[ib], dtype=np.float64)
    print("pair A=%d B=%d" % (ia, ib), flush=True)

    jobs = [
        ("b_k1", 10, 1, M.latent_masks_ancilla(10, 1), 4000, 0.0),
        ("b_k1_shortpath", 10, 1, M.latent_masks_ancilla(10, 1), 4000, 0.03),
        ("b2_imageonly", 10, 1, M.latent_masks_lowvar(10), 4000, 0.0),
        ("bp_noancilla", 10, 0, M.latent_masks_lowvar(10), 4000, 0.0),
        ("c_k2", 10, 2, M.latent_masks_ancilla(10, 2), 5000, 0.0),
        ("c_k2_shortpath", 10, 2, M.latent_masks_ancilla(10, 2), 5000, 0.03),
    ]
    fig_rows = []
    for name, ni, na, lm, steps, mu in jobs:
        print("\n=== %s (n=%d, latent=%d, mu=%g) ===" % (name, ni + na, len(lm), mu), flush=True)
        o = run(name, ni, na, lm, [pA, pB], S, steps=steps, mu=mu)
        summ = {}
        for mode in ("raw", "wrapped"):
            summ[mode] = show(o, mode)
        RES[name] = {k: v for k, v in o.items() if not k.endswith("_imgs")}
        RES[name]["summary"] = summ
        fig_rows.append(("%s / wrapped" % name, o["wrapped"]["rows"], o["wrapped_imgs"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tgtA = codec10.pmf_to_image(pA, S)
    tgtB = codec10.pmf_to_image(pB, S)
    vmax = max([float(np.max(im)) for _, _, ims in fig_rows for im in ims]
               + [float(tgtA.max()), float(tgtB.max())])
    nrow = len(fig_rows) + 1
    fig, axes = plt.subplots(nrow, 11, figsize=(1.5 * 11, 1.9 * nrow))
    for ci in range(11):
        ax = axes[0, ci]
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0:
            ax.imshow(tgtA, cmap="inferno", vmin=0, vmax=vmax, origin="lower")
            ax.set_title("target A (e1 %+.2f)" % e1[ia], fontsize=6.5)
            ax.set_ylabel("data", fontsize=7)
        elif ci == 10:
            ax.imshow(tgtB, cmap="inferno", vmin=0, vmax=vmax, origin="lower")
            ax.set_title("target B (e1 %+.2f)" % e1[ib], fontsize=6.5)
        else:
            ax.axis("off")
    for ri, (title, rows, ims) in enumerate(fig_rows, start=1):
        for ci in range(11):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(ims[ci], cmap="inferno", vmin=0, vmax=vmax, origin="lower")
            r = rows[ci]
            ax.set_title("%s\nA %.2f B %.2f" % (r["label"], r["corr_to"][0], r["corr_to"][1]),
                         fontsize=6.2)
            if ci == 0:
                ax.set_ylabel(title, fontsize=6.5)
    fig.suptitle("A->B latent interpolation, shortest-path (2pi-wrapped) scan, same colour scale",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, "interp_wrapped.png")
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print("\nfigure: %s" % p, flush=True)
    RES["figure"] = p
    with open(os.path.join(OUT, "results2.json"), "w") as fh:
        json.dump(RES, fh, indent=2, default=float)
    print("wrote results2.json", flush=True)


if __name__ == "__main__":
    main()
