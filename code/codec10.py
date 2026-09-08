"""10-qubit "centred crop as pmf" codec for round 3 (docs/IQP_LENS_ROUND3_PREREGISTRATION.md).

Round 2 modelled the whole 64x64 frame as a 12-qubit spatial pmf. Round 3 first ALIGNS each
image on its intensity centroid and crops the central 32x32 window (data_lens_r3.align_crop);
the model then only has to describe the 1,024-cell pmf of that crop, and the two things the
crop threw away -- WHERE the crop sat in the frame (dy, dx) and HOW MUCH light it holds (S) --
are carried outside the circuit and put back by paste() and the scale S.

    code = (kx << 5) | ky           (qubit j == bit j; ky occupies bits 0-4, kx bits 5-9)
    p(code) = crop[ky, kx] / sum(crop)
    reconstruction: crop_hat[ky, kx] = q(code) * S

10 qubits, so |s| <= 10 is the COMPLETE Walsh basis: 1,024 generators, one per subset. The bit
layout is therefore irrelevant to the model's expressiveness (every subset is trainable) --
it only fixes which 50 of the 1,024 are called "the latent block" (latent_select_r3.py).

Flat-index convention (the ONE ordering rule, identical in spirit to codec12.py):

    index = code = kx * 32 + ky     # kx outer, ky inner

so `pmf` is NOT `crop.ravel()` (which is ky-outer) but `crop.T.ravel()`. test_r3.py checks the
round trip and checks that decode(encode(kx, ky)) == (kx, ky) for all 1,024 codes.

Python 3.9 compatible (no `X | None`).
"""
import numpy as np

N_QUBITS = 10
GRID = 32                       # crop side
FRAME = 64                      # full image side
NCODES = 1 << N_QUBITS          # 1024 == GRID * GRID


def encode(kx, ky):
    """(kx, ky) -> 10-bit code = (kx << 5) | ky."""
    kx = np.asarray(kx, dtype=np.int64)
    ky = np.asarray(ky, dtype=np.int64)
    if np.any(kx < 0) or np.any(kx >= GRID) or np.any(ky < 0) or np.any(ky >= GRID):
        raise ValueError("encode: kx, ky must be in [0, %d)" % GRID)
    return (kx << 5) | ky


def decode(codes):
    """10-bit codes -> (kx, ky). Inverse of encode()."""
    codes = np.asarray(codes, dtype=np.int64)
    return (codes >> 5) & 31, codes & 31


def all_codes():
    """The 1,024 codes in flat-index order (code i is at position i, trivially)."""
    return np.arange(NCODES, dtype=np.int64)


def image_to_pmf(crop):
    """(32, 32) crop (row = ky, col = kx) -> (1024,) float64 pmf in code order.

    index = kx * 32 + ky, i.e. `crop.T.ravel()` normalised.
    """
    crop = np.asarray(crop, dtype=np.float64)
    if crop.shape != (GRID, GRID):
        raise ValueError("image_to_pmf: expected (%d, %d), got %s" % (GRID, GRID, (crop.shape,)))
    if np.any(crop < 0):
        raise ValueError("image_to_pmf: negative intensities are not a pmf")
    flat = np.ascontiguousarray(crop.T).ravel()      # [kx, ky] -> index kx*32 + ky
    total = float(flat.sum())
    if not (total > 0):
        raise ValueError("image_to_pmf: total intensity must be > 0, got %r" % total)
    return flat / total


def total_intensity(crop):
    """sum of the crop -- the scalar S that pmf_to_image multiplies back in."""
    return float(np.asarray(crop, dtype=np.float64).sum())


def pmf_to_image(q, S):
    """(1024,) pmf in code order -> (32, 32) float32 crop, row = ky, col = kx, scaled by S.

    Exact inverse of image_to_pmf up to the scale:
        pmf_to_image(image_to_pmf(c), c.sum()) == c
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != NCODES:
        raise ValueError("pmf_to_image: expected %d entries, got %d" % (NCODES, q.size))
    crop = q.reshape(GRID, GRID).T * float(S)        # (kx, ky) -> (ky, kx)
    return np.ascontiguousarray(crop).astype(np.float32)


def sample_codes_from_pmf(pmf, n, rng):
    """Draw n int64 codes with replacement from `pmf` (length 1024, code order)."""
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
    """Empirical pmf (length 1024) of a set of 10-bit codes."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.size == 0:
        raise ValueError("codes_to_pmf: empty code set")
    return np.bincount(codes, minlength=NCODES).astype(np.float64) / float(codes.size)


def codes_to_image(codes, S):
    """Counting-density reconstruction of a 32x32 crop from sampled codes (the RBM arm's
    only reconstruction route -- an RBM has no closed-form marginal over all 1,024 states
    that is cheaper than enumerating them, and the pre-registration asks for Gibbs samples)."""
    return pmf_to_image(codes_to_pmf(codes), S)


def bits_of_codes(codes):
    """(m, 10) float64 0/1 bit matrix of `codes` (bit j == qubit j)."""
    codes = np.asarray(codes, dtype=np.int64)
    return ((codes[:, None] >> np.arange(N_QUBITS)[None, :]) & 1).astype(np.float64)


def paste(crop, dy, dx, frame=FRAME):
    """Put a (32, 32) crop back into a black (64, 64) frame with its top-left corner at
    (dy, dx). The exact inverse of data_lens_r3.align_crop's windowing:

        crop, dy, dx = align_crop(img)
        paste(crop, dy, dx) == img   everywhere inside the window, 0 outside

    so `paste(align_crop(img)) != img` only by the light that fell outside the 32x32 window
    (measured 0.6% of the total on average, worst 1.6% -- test_r3.py re-measures it).

    (dy, dx) is the CORNER, not the centroid offset: it is what the generative arms draw
    with replacement from the training images and hand straight to this function.
    """
    crop = np.asarray(crop, dtype=np.float32)
    if crop.shape != (GRID, GRID):
        raise ValueError("paste: expected a (%d, %d) crop, got %s" % (GRID, GRID, (crop.shape,)))
    dy = int(dy)
    dx = int(dx)
    if not (0 <= dy <= frame - GRID and 0 <= dx <= frame - GRID):
        raise ValueError("paste: corner (%d, %d) does not fit a %dx%d crop in a %dx%d frame"
                         % (dy, dx, GRID, GRID, frame, frame))
    out = np.zeros((frame, frame), dtype=np.float32)
    out[dy:dy + GRID, dx:dx + GRID] = crop
    return out


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    crop = rng.random((GRID, GRID)) ** 3
    q = image_to_pmf(crop)
    back = pmf_to_image(q, total_intensity(crop))
    print("[codec10] round-trip max|err| =", float(np.abs(back - crop.astype(np.float32)).max()))
    kx, ky = decode(all_codes())
    print("[codec10] encode(decode(c)) == c:", bool(np.array_equal(encode(kx, ky), all_codes())))
    print("[codec10] pmf sum =", float(q.sum()), " n_codes =", NCODES)
