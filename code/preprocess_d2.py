#!/usr/bin/env python3
"""Stream-parse the D2 turbulence zip into a compact per-t cache for data_d2.py.

See INTERFACES.md and the 2026-09-05 entry of
docs/IQP_TURBULENCE_REPLICATION_PREREGISTRATION.md for the confirmed file layout:
`<WORK>/data/turbulence_khojasteh2022/2D_velocity_snapshot_sub_domain_2.zip`
(4.1 GB) holds 1000 members `2D_DNS_3900/2D_sub_domain_2_z=[-1,0,1]_t={t}.txt`
(t = 1..1000, complete). Each member has 3 blocks (one per z-plane), each preceded
by its own 2-line `#` header; every block holds 308 (x) x 328 (y) = 101,024 rows of
`x y z u v w` (whitespace-separated, z here is the SPATIAL z-coordinate label, not
the "z_val" used elsewhere for the velocity-magnitude channel). z-plane labels are
-1/0/+1 with actual z-coordinate values approx. 1.96875 / 3.0 / 3.984375.

This script does NOT assume the on-disk block order matches ascending z (it groups
rows by their own z column and sorts by value), and writes only what data_d2.py
needs: the two coordinate axes (shared by every t) plus per-t velocity magnitude
z_val = sqrt(u^2 + v^2 + w^2) on the native (plane, y, x) grid.

Output (in --out, default <WORK>/data/turbulence_khojasteh2022/d2_cache/):
  grid_xy.npz   -- x (308,), y (328,), plane_z (3,) [sorted ascending: -1.97, 3.0, 3.98]
  zval_all.npy  -- float32 memmap-able array, shape (n_t, 3, 328, 308):
                   index 0 = t-1 (1-based t), axis 1 = plane (z ascending),
                   axis 2 = y (0..327, ascending), axis 3 = x (0..307, ascending).
                   Full run (n_t=1000) is ~1.2 GB.

Login-node only: this is I/O-bound preprocessing, not training, but it is still run
niced and single-threaded so it does not starve other users on the login node.

Python 3.9 compatible (no `X | None`).
"""
import argparse
import os
import time
import zipfile

import numpy as np

ZIP_PATH_DEFAULT = "<WORK>/data/turbulence_khojasteh2022/2D_velocity_snapshot_sub_domain_2.zip"
CACHE_DIR_DEFAULT = "<WORK>/data/turbulence_khojasteh2022/d2_cache"
MEMBER_TMPL = "2D_DNS_3900/2D_sub_domain_2_z=[-1,0,1]_t={t}.txt"
NX, NY, NPLANE = 308, 328, 3
N_T_TOTAL = 1000


def parse_member(raw_text):
    """Parse one decoded member's text.

    Returns (x_unique (308,) float64 ascending, y_unique (328,) float64 ascending,
    plane_z (3,) float64 ascending, zmag (3, 328, 308) float32 [plane, y, x]).
    Robust to on-disk block/row order: every row is placed by its OWN (x, y, z)
    values via searchsorted against the row's unique axis values, not by position.
    """
    lines = raw_text.split("\n")
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not data_lines:
        raise ValueError("no data rows found (only headers/blank lines?)")
    body = "\n".join(data_lines)
    flat = np.fromstring(body, sep=" ", dtype=np.float64)
    if flat.size % 6 != 0:
        raise ValueError("token count %d is not a multiple of 6 (x y z u v w)" % flat.size)
    arr = flat.reshape(-1, 6)
    x_col, y_col, z_col, u, v, w = arr.T

    x_unique = np.unique(x_col)
    y_unique = np.unique(y_col)
    plane_z = np.unique(z_col)
    if x_unique.shape[0] != NX:
        raise ValueError("expected %d unique x values, got %d" % (NX, x_unique.shape[0]))
    if y_unique.shape[0] != NY:
        raise ValueError("expected %d unique y values, got %d" % (NY, y_unique.shape[0]))
    if plane_z.shape[0] != NPLANE:
        raise ValueError("expected %d unique z-planes, got %d" % (NPLANE, plane_z.shape[0]))

    x_idx = np.searchsorted(x_unique, x_col)
    y_idx = np.searchsorted(y_unique, y_col)
    p_idx = np.searchsorted(plane_z, z_col)
    zmag = np.sqrt(u * u + v * v + w * w).astype(np.float32)

    out = np.full((NPLANE, NY, NX), np.nan, dtype=np.float32)
    out[p_idx, y_idx, x_idx] = zmag
    n_expected = NPLANE * NY * NX
    if arr.shape[0] != n_expected:
        raise ValueError("expected %d rows, got %d" % (n_expected, arr.shape[0]))
    if np.isnan(out).any():
        raise ValueError("grid has unfilled cells after placement (duplicate/missing points?)")
    return x_unique, y_unique, plane_z, out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Preprocess D2 turbulence zip into a numpy cache.")
    p.add_argument("--zip", default=ZIP_PATH_DEFAULT)
    p.add_argument("--out", default=CACHE_DIR_DEFAULT)
    p.add_argument("--limit", type=int, default=0, help="only process the first N t (smoke test); 0 = all 1000")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Login-node courtesy: this is single-threaded I/O + light numpy, but nice it and
    # cap BLAS threads so it never contends with other users' interactive work.
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    os.makedirs(args.out, exist_ok=True)
    n_t = args.limit if args.limit and args.limit > 0 else N_T_TOTAL
    if n_t > N_T_TOTAL:
        raise SystemExit("--limit %d exceeds available t range (1..%d)" % (n_t, N_T_TOTAL))

    zval_path = os.path.join(args.out, "zval_all.npy")
    zval_all = np.lib.format.open_memmap(
        zval_path, mode="w+", dtype=np.float32, shape=(n_t, NPLANE, NY, NX)
    )

    x_unique = y_unique = plane_z = None
    t0 = time.time()
    with zipfile.ZipFile(args.zip) as zf:
        for i, t in enumerate(range(1, n_t + 1)):
            member = MEMBER_TMPL.format(t=t)
            with zf.open(member) as f:
                raw = f.read().decode("ascii")
            xu, yu, pz, grid = parse_member(raw)
            if x_unique is None:
                x_unique, y_unique, plane_z = xu, yu, pz
            elif not (np.array_equal(xu, x_unique) and np.array_equal(yu, y_unique)
                      and np.array_equal(pz, plane_z)):
                raise ValueError("grid_xy mismatch at t=%d (grid should be identical across t)" % t)
            zval_all[i] = grid
            if (i + 1) % 100 == 0 or (i + 1) == n_t:
                dt = time.time() - t0
                print(
                    "[preprocess_d2] %d/%d t done (%.1fs elapsed, %.3f s/t)"
                    % (i + 1, n_t, dt, dt / (i + 1)),
                    flush=True,
                )

    zval_all.flush()
    del zval_all

    np.savez(os.path.join(args.out, "grid_xy.npz"), x=x_unique, y=y_unique, plane_z=plane_z)

    dt = time.time() - t0
    size_mb = os.path.getsize(zval_path) / 1e6
    print(
        "[preprocess_d2] done: %d t -> %s (%.1f MB, %.1fs total, %.3f s/t)"
        % (n_t, zval_path, size_mb, dt, dt / max(n_t, 1))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
