"""12-qubit "image as pmf" codec for the round-2 lens arms (docs/IQP_LENS_PILOT_PREREGISTRATION.md
Addendum 2026-09-06).

Round 1 used the paper's literal tuple encoding: one code = (kx, ky, kz) on 18 qubits, so the
model has to learn p(x, y, z) over a pool in which 84% of the pixels sit in intensity bin 0.
Round 2's pmf encoding drops the z channel entirely and treats **the image itself as the
distribution over pixels**:

    code = (kx << 6) | ky          (qubit j == bit j; ky occupies bits 0-5, kx bits 6-11)
    p(code) = image[ky, kx] / sum(image)
    reconstruction: image_hat[ky, kx] = q(code) * S,  S = mean total intensity of the
                    training images (carried outside the circuit, one scalar per arm)

so the 12-qubit circuit models exactly the 4,096-cell spatial pmf and nothing else. |s| <= 6
gives 2,510 generators, |s| <= 12 gives all 4,096 (a complete basis for the 12-bit Walsh
spectrum).

Flat-index convention (the ONE ordering rule; every function here obeys it and
test_lens_round2.py checks it):

    index = code = kx * 64 + ky     # kx outer, ky inner

so `pmf` is NOT `image.ravel()` (which is ky-outer) but `image.T.ravel()`. The (kx, ky)
meaning of the low 12 bits is identical to codec.py's: codec.decode(c18) -> (kx, ky, kz)
with kx = c18 >> 12 and ky = (c18 >> 6) & 63, and codec12.decode((kx << 6) | ky) gives the
same (kx, ky).  test_lens_round2.py asserts that agreement.

Python 3.9 compatible (no `X | None`).
"""
import numpy as np

N_QUBITS = 12
GRID = 64
NCODES = 1 << N_QUBITS          # 4096 == GRID * GRID


def encode(kx, ky):
    """(kx, ky) -> 12-bit code = (kx << 6) | ky."""
    kx = np.asarray(kx, dtype=np.int64)
    ky = np.asarray(ky, dtype=np.int64)
    return (kx << 6) | ky


def decode(codes):
    """12-bit codes -> (kx, ky). Inverse of encode()."""
    codes = np.asarray(codes, dtype=np.int64)
    return (codes >> 6) & 63, codes & 63


def all_codes():
    """The 4,096 codes in flat-index order (code i is at position i, trivially)."""
    return np.arange(NCODES, dtype=np.int64)


def image_to_pmf(img):
    """(64, 64) image (row = ky, col = kx) -> (4096,) float64 pmf in code order.

    index = kx * 64 + ky, i.e. `img.T.ravel()` normalised. Raises on a non-positive
    total (a black image has no pmf).
    """
    img = np.asarray(img, dtype=np.float64)
    if img.shape != (GRID, GRID):
        raise ValueError("image_to_pmf: expected (%d, %d), got %s" % (GRID, GRID, (img.shape,)))
    if np.any(img < 0):
        raise ValueError("image_to_pmf: negative intensities are not a pmf")
    flat = np.ascontiguousarray(img.T).ravel()      # [kx, ky] -> index kx*64 + ky
    total = float(flat.sum())
    if not (total > 0):
        raise ValueError("image_to_pmf: total intensity must be > 0, got %r" % total)
    return flat / total


def total_intensity(img):
    """sum of the image -- the scalar S that pmf_to_image multiplies back in."""
    return float(np.asarray(img, dtype=np.float64).sum())


def pmf_to_image(q, S):
    """(4096,) pmf in code order -> (64, 64) float32 image, row = ky, col = kx, scaled by S.

    Exact inverse of image_to_pmf up to the scale: pmf_to_image(image_to_pmf(f), f.sum()) == f.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != NCODES:
        raise ValueError("pmf_to_image: expected %d entries, got %d" % (NCODES, q.size))
    img = q.reshape(GRID, GRID).T * float(S)        # (kx, ky) -> (ky, kx)
    return np.ascontiguousarray(img).astype(np.float32)


def sample_codes_from_pmf(pmf, n, rng):
    """Draw n int64 codes with replacement from `pmf` (length 4096, code order)."""
    p = np.asarray(pmf, dtype=np.float64).reshape(-1)
    if p.size != NCODES:
        raise ValueError("sample_codes_from_pmf: expected %d entries, got %d" % (NCODES, p.size))
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if not (s > 0):
        raise ValueError("sample_codes_from_pmf: pmf sums to %r" % s)
    p = p / s
    return np.asarray(rng.choice(NCODES, size=int(n), replace=True, p=p), dtype=np.int64)


def codes_to_pmf(codes):
    """Empirical pmf (length 4096) of a set of 12-bit codes -- codec.codes_to_pmf's analogue."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.size == 0:
        raise ValueError("codes_to_pmf: empty code set")
    return np.bincount(codes, minlength=NCODES).astype(np.float64) / float(codes.size)


def codes_to_image(codes, S):
    """Counting-density reconstruction of an image from sampled codes (the `_sampled/`
    control that every pmf arm writes next to its exact reconstruction)."""
    return pmf_to_image(codes_to_pmf(codes), S)


def bits_of_codes(codes):
    """(m, 12) float64 0/1 bit matrix of `codes` (bit j == qubit j), for data-initialisation."""
    codes = np.asarray(codes, dtype=np.int64)
    return ((codes[:, None] >> np.arange(N_QUBITS)[None, :]) & 1).astype(np.float64)


def pmf_bit_moments(pmf):
    """Exact (means, cov_pm1) of the bits under `pmf` -- the full-grid analogue of taking
    sample moments of a code draw.

    means[j]      = E[x_j]                      (x in {0, 1})
    cov_pm1[j, k] = E[(2x_j-1)(2x_k-1)] - E[2x_j-1] E[2x_k-1]   (diagonal zeroed, as in
                    Ref [38]'s initialize_from_data)
    """
    p = np.asarray(pmf, dtype=np.float64).reshape(-1)
    if p.size != NCODES:
        raise ValueError("pmf_bit_moments: expected %d entries, got %d" % (NCODES, p.size))
    bits = bits_of_codes(all_codes())                       # (4096, 12)
    means = p @ bits                                        # (12,)
    pm1 = 2.0 * bits - 1.0                                  # (4096, 12)
    mu = 2.0 * means - 1.0                                  # E[2x-1]
    second = (pm1 * p[:, None]).T @ pm1                     # E[(2x_j-1)(2x_k-1)]
    cov = second - np.outer(mu, mu)
    np.fill_diagonal(cov, 0.0)
    return means, cov


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.random((GRID, GRID)) ** 3
    q = image_to_pmf(img)
    back = pmf_to_image(q, total_intensity(img))
    print("[codec12] round-trip max|err| =", float(np.abs(back - img.astype(np.float32)).max()))
    kx, ky = decode(np.arange(NCODES, dtype=np.int64))
    print("[codec12] encode(decode(c)) == c:", bool(np.array_equal(encode(kx, ky), np.arange(NCODES))))
    print("[codec12] pmf sum =", float(q.sum()), " n_codes =", NCODES)
