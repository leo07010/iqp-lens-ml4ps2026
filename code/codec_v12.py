#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three interchangeable code<->crop encodings for the v12 IQP arms.

NEW file.  codec10.py is NOT modified: `crop10` below delegates to it, and test_v12.py checks
that the delegation is exact, so the default v12 arm keeps reproducing the published arm
bit-for-bit.  The other two encodings are the coordinator's arms F and G.

    crop10   the published one.  32x32 aligned crop, 10 qubits, code = (kx << 5) | ky.
    gray10   arm G.  The SAME 32x32 crop and the same 1,024 Walsh generators, but each axis
             index is written in reflected Gray code before the two are packed:
             code = (G(kx) << 5) | G(ky),  G(v) = v ^ (v >> 1).
             The circuit is unchanged (all 1,024 subsets are trainable, and a permutation of
             the code axis cannot change what the model can express); what changes is the
             KERNEL.  Both trained kernels are functions of Hamming distance between codes,
             and under the plain binary code a 1-pixel step can flip up to 5 bits
             (15 -> 16 = 01111 -> 10000).  Under a Gray code every 1-pixel step flips exactly
             one bit per axis, so Hamming distance is much closer to a pixel distance.  This
             arm tests whether that alignment is what the Hamming kernels were missing.
    lowres8  arm F.  The 32x32 crop is block-summed 2x2 down to 16x16 -> 8 qubits, 256 codes,
             all 256 Walsh generators trainable (256 + M x 256 parameters, 768 at M = 2, a
             seventh of the 10-qubit arm).  Generation upsamples the 16x16 pmf back to 32x32
             with a FIXED bilinear map (no learned parameters) and rescales to the flux S.
             The bilinear map is `U q U^T` with U the 32x16 align_corners=False interpolation
             matrix -- separable, exact, and the same matrix for every image, so nothing about
             it is fitted.  Its cost is real and is REPORTED: `encode_decode_corr` in
             meta.json is the correlation between a real training crop and its own
             block-sum -> bilinear round trip, i.e. the ceiling this encoding can reach.

Every encoding exposes:
    name, n_qubits, ncodes, grid (side of the code grid), axis_x/axis_y (physical pixel
    coordinate of each axis index, in units of the 32x32 crop, so one sigma means the same
    distance in every encoding), pmf_of_crop(crop), crop_of_pmf(q, S).

Python 3.9 compatible.
"""
from __future__ import annotations

import math

import numpy as np

import codec10

CROP = 32          # side of the aligned crop every encoding starts from
__all__ = ["get", "ENCODINGS", "Encoding", "gray_code", "bilinear_matrix"]


# --------------------------------------------------------------------------------------
def gray_code(v):
    """Reflected binary Gray code, G(v) = v ^ (v >> 1). Bijective on 0 .. 2^k - 1."""
    v = np.asarray(v, dtype=np.int64)
    return v ^ (v >> 1)


def bilinear_matrix(n_out, n_in):
    """(n_out, n_in) separable bilinear interpolation matrix, align_corners=False semantics.

    Output index j samples the input at x = (j + 0.5) * n_in / n_out - 0.5, linearly between
    its two neighbours, with the edges replicated (the clamp).  `U @ v` is then the 1-D
    upsample of v and `U @ A @ U.T` the 2-D one.  Matches
    torch.nn.functional.interpolate(mode='bilinear', align_corners=False); test_v12.py checks
    that against torch directly.
    """
    n_out, n_in = int(n_out), int(n_in)
    U = np.zeros((n_out, n_in), dtype=np.float64)
    scale = float(n_in) / float(n_out)
    for j in range(n_out):
        x = (j + 0.5) * scale - 0.5
        x0 = int(math.floor(x))
        w = x - x0
        a = min(max(x0, 0), n_in - 1)
        b = min(max(x0 + 1, 0), n_in - 1)
        U[j, a] += (1.0 - w)
        U[j, b] += w
    return U


# --------------------------------------------------------------------------------------
class Encoding(object):
    """One code<->crop convention.  `axis_x[a]` is the physical x (in 32x32 crop pixels) of
    axis index a, and the flat code index is always `a * grid + b` with a on the x axis and
    b on the y axis, so every encoding's code axis is a product grid and the spatial kernel
    stays separable."""

    def __init__(self, name, n_qubits, grid, axis_x, axis_y):
        self.name = name
        self.n_qubits = int(n_qubits)
        self.ncodes = 1 << int(n_qubits)
        self.grid = int(grid)
        self.axis_x = np.asarray(axis_x, dtype=np.float64)
        self.axis_y = np.asarray(axis_y, dtype=np.float64)
        self.phi_scale = math.sqrt(float(self.ncodes))
        if self.grid * self.grid != self.ncodes:
            raise ValueError("%s: grid^2 != ncodes" % name)

    def pmf_of_crop(self, crop):
        raise NotImplementedError

    def crop_of_pmf(self, q, S):
        raise NotImplementedError

    def pmfs_of_crops(self, crops):
        crops = np.asarray(crops, dtype=np.float64)
        return np.stack([self.pmf_of_crop(crops[i]) for i in range(crops.shape[0])], axis=0)

    def roundtrip_corr(self, crops):
        """Mean Pearson correlation between a crop and its own encode -> decode image.

        1.0 for the lossless encodings; for `lowres8` it is the ceiling the arm can reach.
        """
        crops = np.asarray(crops, dtype=np.float64)
        out = []
        for i in range(crops.shape[0]):
            a = crops[i].ravel()
            b = self.crop_of_pmf(self.pmf_of_crop(crops[i]), float(a.sum())).astype(np.float64).ravel()
            a = a - a.mean()
            b = b - b.mean()
            den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
            out.append(float((a * b).sum() / den) if den > 0 else 0.0)
        return float(np.mean(out)) if out else float("nan")


class _Crop10(Encoding):
    """The published encoding.  Delegates to codec10 so it cannot drift from it."""

    def __init__(self):
        Encoding.__init__(self, "crop10", 10, CROP,
                          np.arange(CROP, dtype=np.float64), np.arange(CROP, dtype=np.float64))

    def pmf_of_crop(self, crop):
        return codec10.image_to_pmf(crop)

    def crop_of_pmf(self, q, S):
        return codec10.pmf_to_image(q, S)


class _Gray10(Encoding):
    """32x32 crop, but each axis index is Gray-coded before packing (arm G).

    `_to_gray[s] = g` maps the plain code s = (kx << 5) | ky to g = (G(kx) << 5) | G(ky); the
    map is a permutation of 0..1023, so this is exactly a relabelling of the 1,024 basis
    states.  axis_x[a] = G^{-1}(a) keeps the PHYSICAL pixel coordinate attached to the axis
    index, which is what the spatial kernel and the figures need.
    """

    def __init__(self):
        g = gray_code(np.arange(CROP, dtype=np.int64))          # axis index of pixel v
        inv = np.zeros(CROP, dtype=np.int64)
        inv[g] = np.arange(CROP, dtype=np.int64)                # pixel of axis index a
        Encoding.__init__(self, "gray10", 10, CROP,
                          inv.astype(np.float64), inv.astype(np.float64))
        self._g = g
        self._inv = inv
        kx = np.arange(CROP, dtype=np.int64)[:, None] * np.ones((1, CROP), dtype=np.int64)
        ky = np.ones((CROP, 1), dtype=np.int64) * np.arange(CROP, dtype=np.int64)[None, :]
        # plain index kx*32+ky  ->  gray index G(kx)*32+G(ky)
        self._perm = (g[kx] * CROP + g[ky]).ravel()

    def pmf_of_crop(self, crop):
        p = codec10.image_to_pmf(crop)                          # plain code order
        out = np.zeros(self.ncodes, dtype=np.float64)
        out[self._perm] = p
        return out

    def crop_of_pmf(self, q, S):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        return codec10.pmf_to_image(q[self._perm], S)


class _LowRes8(Encoding):
    """16x16 block-summed pmf + a fixed bilinear upsample back to 32x32 (arm F)."""

    G16 = 16

    def __init__(self):
        c = (np.arange(self.G16, dtype=np.float64) * 2.0 + 0.5)   # centre of each 2x2 block
        Encoding.__init__(self, "lowres8", 8, self.G16, c, c)
        self._U = bilinear_matrix(CROP, self.G16)

    def pmf_of_crop(self, crop):
        crop = np.asarray(crop, dtype=np.float64)
        if crop.shape != (CROP, CROP):
            raise ValueError("lowres8: expected a (32, 32) crop, got %s" % (crop.shape,))
        if np.any(crop < 0):
            raise ValueError("lowres8: negative intensities are not a pmf")
        # (ky, kx) -> 2x2 block sums -> (kx, ky) flat, index = kx*16 + ky (same rule as codec10)
        small = crop.reshape(self.G16, 2, self.G16, 2).sum(axis=(1, 3))     # (ky16, kx16)
        flat = np.ascontiguousarray(small.T).ravel()
        total = float(flat.sum())
        if not (total > 0):
            raise ValueError("lowres8: total intensity must be > 0, got %r" % total)
        return flat / total

    def crop_of_pmf(self, q, S):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != self.ncodes:
            raise ValueError("lowres8: expected %d entries, got %d" % (self.ncodes, q.size))
        small = q.reshape(self.G16, self.G16).T                  # (ky16, kx16)
        up = self._U @ small @ self._U.T                         # (32, 32), (ky, kx)
        up = np.clip(up, 0.0, None)
        s = float(up.sum())
        if not (s > 0):
            up = np.full((CROP, CROP), 1.0 / (CROP * CROP), dtype=np.float64)
            s = 1.0
        return np.ascontiguousarray(up * (float(S) / s)).astype(np.float32)


ENCODINGS = {"crop10": _Crop10, "gray10": _Gray10, "lowres8": _LowRes8}


def get(name):
    name = str(name)
    if name not in ENCODINGS:
        raise ValueError("unknown encoding %r, expected one of %s" % (name, sorted(ENCODINGS)))
    return ENCODINGS[name]()


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    crop = (rng.random((CROP, CROP)) ** 3) * 5.0
    for nm in ("crop10", "gray10", "lowres8"):
        e = get(nm)
        p = e.pmf_of_crop(crop)
        back = e.crop_of_pmf(p, crop.sum())
        print("[codec_v12] %-8s n=%2d ncodes=%4d pmf sum=%.10f  round-trip max|err|=%.3e "
              "corr=%.6f  total=%.6f"
              % (nm, e.n_qubits, e.ncodes, p.sum(),
                 float(np.abs(back.astype(np.float64) - crop).max()),
                 e.roundtrip_corr(crop[None]), float(back.sum())))
