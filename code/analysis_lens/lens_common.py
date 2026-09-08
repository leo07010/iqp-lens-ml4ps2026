"""Shared helpers for the DeepLense image-characteristics analysis.

Self-contained on purpose: nothing is imported from the round-1/round-2
IQP pipeline so that this analysis cannot perturb it.
Python 3.9, numpy only.
"""
import numpy as np

# ----------------------------------------------------------------- FWHT
def fwht(a):
    """Unnormalized fast Walsh-Hadamard transform along the last axis.

    Returns W with W[s] = sum_x a[x] * (-1)^{popcount(s & x)}.
    Works batched: a of shape (..., 2**n).
    """
    a = np.array(a, dtype=np.float64, copy=True)
    shape = a.shape
    N = shape[-1]
    assert N & (N - 1) == 0, "length must be a power of two"
    a = a.reshape(-1, N)
    B = a.shape[0]
    h = 1
    while h < N:
        a = a.reshape(B, N // (2 * h), 2, h)
        x = a[:, :, 0, :].copy()
        y = a[:, :, 1, :].copy()
        a[:, :, 0, :] = x + y
        a[:, :, 1, :] = x - y
        a = a.reshape(B, N)
        h *= 2
    return a.reshape(shape)


def popcount_table(n_bits):
    """popcount for every index in [0, 2**n_bits)."""
    N = 1 << n_bits
    pc = np.zeros(N, dtype=np.int8)
    for b in range(n_bits):
        pc |= 0  # keep dtype
    pc = np.zeros(N, dtype=np.int16)
    idx = np.arange(N, dtype=np.int64)
    for b in range(n_bits):
        pc += ((idx >> b) & 1).astype(np.int16)
    return pc


def order_spectrum(W, pc, n_bits):
    """Fraction of Walsh energy W[s]**2 carried by each Hamming weight |s|.

    W: (..., 2**n). Returns (..., n_bits+1) rows summing to 1 (s=0 included).
    """
    E = W.astype(np.float64) ** 2
    flat = E.reshape(-1, E.shape[-1])
    out = np.zeros((flat.shape[0], n_bits + 1))
    for k in range(n_bits + 1):
        m = (pc == k)
        if m.any():
            out[:, k] = flat[:, m].sum(axis=1)
    out /= out.sum(axis=1, keepdims=True)
    return out.reshape(E.shape[:-1] + (n_bits + 1,))


# ------------------------------------------------------- 12-bit layouts
def _gray(v):
    return v ^ (v >> 1)


def layout_index(kind):
    """Return perm[4096] mapping raster position (ky*64+kx) -> 12-bit code.

    kx = column (x), ky = row (y), matching the pipeline convention
    code = (kx << 6) | ky for the standard layout.
    """
    ky, kx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    kx = kx.ravel().astype(np.int64)
    ky = ky.ravel().astype(np.int64)
    if kind == "standard":
        code = (kx << 6) | ky
    elif kind == "morton":
        code = np.zeros_like(kx)
        for b in range(6):
            code |= ((kx >> b) & 1) << (2 * b + 1)
            code |= ((ky >> b) & 1) << (2 * b)
    elif kind == "gray":
        code = (_gray(kx) << 6) | _gray(ky)
    else:
        raise ValueError(kind)
    return code


def image_to_pmf12(imgs, code):
    """(B,64,64) images -> (B,4096) pmf laid out on the given 12-bit code."""
    flat = imgs.reshape(imgs.shape[0], -1).astype(np.float64)
    s = flat.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    flat = flat / s
    out = np.zeros_like(flat)
    out[:, code] = flat
    return out


# ---------------------------------------------------- 18-bit tuple pmf
def quantize_z(z, mode):
    if mode == "raw":
        v = z
    elif mode == "sqrt":
        v = np.sqrt(np.clip(z, 0, None))
    else:
        raise ValueError(mode)
    return np.clip(np.floor(64.0 * v), 0, 63).astype(np.int64)


def image_to_pmf18(imgs, mode):
    """(B,64,64) -> (B, 2**18) pmf, uniform over the 4096 (kx,ky,kz) tuples."""
    B = imgs.shape[0]
    ky, kx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    kx = kx.ravel().astype(np.int64)
    ky = ky.ravel().astype(np.int64)
    base = (kx << 12) | (ky << 6)
    out = np.zeros((B, 1 << 18), dtype=np.float64)
    for i in range(B):
        kz = quantize_z(imgs[i].ravel().astype(np.float64), mode)
        np.add.at(out[i], base | kz, 1.0 / 4096.0)
    return out
