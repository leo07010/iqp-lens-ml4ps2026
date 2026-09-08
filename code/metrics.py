"""Evaluation metrics of arXiv 2603.06440 Appendix I (paper.txt:2699-2751).

(i)  pdf_js  : Jensen-Shannon divergence between the marginal pixel-intensity histograms
               (B = 50 bins over the JOINT min/max of both sets, eps-smoothed, log base 2).
(ii) conv_mmd2: biased MMD^2 in the feature space of a FIXED, RANDOMLY INITIALIZED Conv2D encoder
               (three Conv2D-ReLU stride-2 blocks + global average pooling -> 64-dim features),
               Gaussian kernel with sigma from the median heuristic on the pooled features,
               N = M = 200 snapshots.

Both functions take (N, 64, 64) numpy arrays.  Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = ["pdf_js", "conv_mmd2", "ConvEncoder"]

N_EVAL = 200


# --------------------------------------------------------------------------------------
# (i) PDF Jensen-Shannon divergence
# --------------------------------------------------------------------------------------
def pdf_js(real: np.ndarray, gen: np.ndarray, bins: int = 50, eps: float = 1e-12) -> float:
    """JS divergence (log base 2, in bits) between pixel-intensity histograms of two snapshot sets."""
    x = np.asarray(real, dtype=np.float64).ravel()
    y = np.asarray(gen, dtype=np.float64).ravel()
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(bins) + 1)
    p, _ = np.histogram(x, bins=edges)
    q, _ = np.histogram(y, bins=edges)
    p = p.astype(np.float64) / max(p.sum(), 1.0) + eps
    q = q.astype(np.float64) / max(q.sum(), 1.0) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log2(p) - np.log2(m))))
    kl_qm = float(np.sum(q * (np.log2(q) - np.log2(m))))
    return 0.5 * kl_pm + 0.5 * kl_qm


# --------------------------------------------------------------------------------------
# (ii) Conv2D feature-space MMD^2
# --------------------------------------------------------------------------------------
class ConvEncoder(nn.Module):
    """Three Conv2D(k=3, stride=2, padding=1) - ReLU blocks (1->16->32->64) + global average pool."""

    def __init__(self, channels=(16, 32, 64)):
        super(ConvEncoder, self).__init__()
        layers = []
        c_in = 1
        for c_out in channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.feat_dim = c_in

    def forward(self, x):
        h = self.body(x)
        return h.mean(dim=(2, 3))


def _features(arr, encoder, lo, hi):
    x = np.asarray(arr, dtype=np.float32)
    x = (x - lo) / max(hi - lo, 1e-12)
    t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return encoder(t).double()


def _subsample(arr, n, rng):
    arr = np.asarray(arr)
    if n is None or arr.shape[0] <= n:
        return arr
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[np.sort(idx)]


def conv_mmd2(real: np.ndarray, gen: np.ndarray, seed: int = 0,
              n_eval: Optional[int] = N_EVAL) -> float:
    """Biased empirical MMD^2 in the fixed random Conv2D feature space (Appendix I, Eq. 62-63)."""
    rng = np.random.default_rng(seed)
    real = _subsample(real, n_eval, rng)
    gen = _subsample(gen, n_eval, rng)

    lo = float(min(np.min(real), np.min(gen)))
    hi = float(max(np.max(real), np.max(gen)))

    gen_t = torch.Generator().manual_seed(int(seed))
    encoder = ConvEncoder()
    with torch.no_grad():
        for p in encoder.parameters():
            if p.dim() > 1:
                fan_in = int(np.prod(p.shape[1:]))
                bound = 1.0 / np.sqrt(fan_in)
                p.uniform_(-bound, bound, generator=gen_t)
            else:
                p.zero_()
    encoder.eval()

    f = _features(real, encoder, lo, hi)
    g = _features(gen, encoder, lo, hi)

    pooled = torch.cat((f, g), dim=0)
    d = torch.cdist(pooled, pooled, p=2)
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
    med = torch.median(d[iu[0], iu[1]])
    sigma = float(med) if float(med) > 0 else 1.0

    def _k(a, b):
        dd = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dd / (2.0 * sigma * sigma))

    mmd2 = float(_k(f, f).mean() + _k(g, g).mean() - 2.0 * _k(f, g).mean())
    return mmd2
