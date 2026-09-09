#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixel-space Gaussian MMD^2 for the v12 IQP arms (the coordinator's arm E).

WHY.  Both objectives the paper's IQP arms train on are functions of the HAMMING distance
between bitstrings: the flat kernel weights every Walsh subset equally, the paper kernel
weights it by |s|.  Neither knows that code 15 and code 16 are adjacent pixels.  The metric
the arms are judged by (evaluate_v11.conv_mmd2) is a Conv2D encoder followed by an RBF kernel,
i.e. a SPATIAL statistic, and the DCGAN's convolutional generator is spatially smooth by
construction.  This module supplies the missing spatial objective.

WHAT.  With the codec's pmf q and the image's empirical pmf p on the same 2^n codes,

    MMD^2_spatial(q, p) = (q - p)^T K (q - p),
    K = sum_{sigma in sigmas} K_sigma ,
    K_sigma[c, c'] = exp( -||pos(c) - pos(c')||^2 / (2 sigma^2) )

with pos(c) the pixel coordinate of code c, in units of the 32x32 crop for EVERY encoding, so
one sigma means one physical distance whatever the grid is.  This is the exact
population MMD^2 of the Gaussian kernel k(x, x') = exp(-||pos(x) - pos(x')||^2 / 2 sigma^2)
between the two distributions over pixels; on hardware the same quantity is the ordinary
two-sample kernel estimate on sampled bitstrings, so the objective is not simulation-only.

HOW (why it is cheap).  ||pos(c) - pos(c')||^2 separates into the x and y parts, so each
K_sigma = Kx_sigma (x) Ky_sigma with Kx, Ky of size grid x grid (32 or 16).  The quadratic
form is then three small matrix products per bandwidth instead of one 1024x1024 product:
O(B * grid^3) rather than O(B * grid^4).  The code index is a * grid + b throughout (a on the
x axis, b on the y axis) -- the one ordering rule codec_v12 also follows -- which is what
makes the separation legal for the Gray-coded encoding too (a Gray code permutes each AXIS,
so the code grid is still a product grid, only its axis labels move).

FLAT TERM.  The flat-Hamming MMD is exactly ||q - p||^2:  it is 2^-n sum_s (W(q-p))_s^2 and
W^T W = 2^n I, so the two are equal.  `lam_flat` therefore adds an identity component to K,
which is also the sigma -> 0 limit of the Gaussian one.

Python 3.9 compatible.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["SpatialMMD", "parse_sigmas"]


def parse_sigmas(text):
    """'1,2,4' -> [1.0, 2.0, 4.0]; rejects non-positive values."""
    out = [float(s) for s in str(text).split(",") if s.strip()]
    if not out:
        raise ValueError("--spatial-sigmas must be a non-empty comma-separated list")
    if any(s <= 0 for s in out):
        raise ValueError("--spatial-sigmas must all be > 0, got %r" % (out,))
    return out


class SpatialMMD(object):
    """The separable multi-bandwidth spatial kernel of one encoding.

    Parameters
    ----------
    axis_x, axis_y : (grid,) physical coordinates of each axis index, in crop pixels.
    sigmas         : bandwidths, in the same pixel units.
    lam_flat       : weight of the added identity (flat-Hamming) component; 0 disables it.
    """

    def __init__(self, axis_x, axis_y, sigmas, device, dtype, lam_flat=0.0):
        self.sigmas = [float(s) for s in sigmas]
        self.grid = int(np.asarray(axis_x).size)
        if int(np.asarray(axis_y).size) != self.grid:
            raise ValueError("SpatialMMD: axis_x and axis_y must have the same length")
        self.ncodes = self.grid * self.grid
        self.lam_flat = float(lam_flat)
        ax = np.asarray(axis_x, dtype=np.float64)
        ay = np.asarray(axis_y, dtype=np.float64)
        self.Kx, self.Ky = [], []
        for s in self.sigmas:
            gx = np.exp(-((ax[:, None] - ax[None, :]) ** 2) / (2.0 * s * s))
            gy = np.exp(-((ay[:, None] - ay[None, :]) ** 2) / (2.0 * s * s))
            self.Kx.append(torch.as_tensor(gx, dtype=dtype, device=device))
            self.Ky.append(torch.as_tensor(gy, dtype=dtype, device=device))

    def set_lam_flat(self, lam):
        self.lam_flat = float(lam)

    def rows(self, d):
        """(B,) quadratic form d^T K d for every row of d (B, ncodes).

        d is the pmf-space residual q_i - p_i.  Autograd-safe; no 2^n x 2^n object is ever
        formed.
        """
        if d.dim() != 2 or d.shape[1] != self.ncodes:
            raise ValueError("SpatialMMD.rows: expected (B, %d), got %s"
                             % (self.ncodes, tuple(d.shape)))
        Dm = d.reshape(d.shape[0], self.grid, self.grid)          # [b, a(x), b(y)]
        out = None
        for Kx, Ky in zip(self.Kx, self.Ky):
            t = torch.einsum("ap,npq->naq", Kx, Dm)
            t = torch.einsum("naq,qr->nar", t, Ky)
            v = (Dm * t).sum(dim=(1, 2))
            out = v if out is None else out + v
        if self.lam_flat != 0.0:
            out = out + self.lam_flat * (d * d).sum(dim=1)
        return out

    def dense(self):
        """The explicit (ncodes, ncodes) K -- for tests only, never used in training."""
        K = None
        for Kx, Ky in zip(self.Kx, self.Ky):
            m = torch.kron(Kx, Ky)
            K = m if K is None else K + m
        if self.lam_flat != 0.0:
            K = K + self.lam_flat * torch.eye(self.ncodes, dtype=K.dtype, device=K.device)
        return K


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import codec_v12
    torch.manual_seed(0)
    for nm in ("crop10", "gray10", "lowres8"):
        e = codec_v12.get(nm)
        sp = SpatialMMD(e.axis_x, e.axis_y, [1.0, 2.0, 4.0], "cpu", torch.float64, lam_flat=0.3)
        d = torch.randn(5, e.ncodes, dtype=torch.float64)
        fast = sp.rows(d)
        K = sp.dense()
        slow = ((d @ K) * d).sum(dim=1)
        print("[objective_spatial] %-8s ncodes=%4d  separable vs dense max|d| = %.3e"
              % (nm, e.ncodes, float((fast - slow).abs().max())))
    # the flat-Hamming identity: flat MMD == ||q - p||^2
    import iqp_exact
    d = torch.randn(4, 1024, dtype=torch.float64)
    flat = (iqp_exact.fwht(d) ** 2).sum(dim=1) / 1024.0
    print("[objective_spatial] flat-Hamming MMD == ||d||^2 : max|d| = %.3e"
          % float((flat - (d * d).sum(dim=1)).abs().max()))
