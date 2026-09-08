"""Supplement: which BITS carry the cross-image information, and why the
bit layout does / does not change the Walsh order spectrum.

Two claims tested numerically:
  (A) Morton (Z-order) is a pure *permutation of bits*, which cannot change
      |s|, so the order spectrum must be identical to the standard layout.
  (B) Gray-coded coordinates are an invertible GF(2)-LINEAR map on the
      position label, so the multiset of Walsh coefficients is preserved and
      only their |s| labels change -> the data-adaptive top-M energy is
      identical, but the Hamming-ordered spectrum improves.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lens_common import fwht, popcount_table, layout_index, image_to_pmf12

SCRATCH = os.environ.get("LENS_SCRATCH", "/tmp")
CLASSES = ["axion", "cdm", "no_sub"]
val_sub = np.load(os.path.join(SCRATCH, "val_sub.npy"))
allimg = val_sub.reshape(-1, 64, 64)
PC12 = popcount_table(12)
out = {}

# ---- (A)/(B) layout invariance -------------------------------------
Ws = {k: fwht(image_to_pmf12(allimg, layout_index(k)))
      for k in ("standard", "morton", "gray")}
ref = np.sort((Ws["standard"] ** 2).mean(0))[::-1]
for k in ("morton", "gray"):
    cur = np.sort((Ws[k] ** 2).mean(0))[::-1]
    out["multiset_maxreldiff_vs_standard_%s" % k] = float(
        np.abs(cur - ref).max() / ref.max())
# per-image check too (not just the mean)
out["per_image_sorted_energy_maxreldiff_gray"] = float(
    np.abs(np.sort(Ws["gray"] ** 2, axis=1) - np.sort(Ws["standard"] ** 2, axis=1)).max()
    / (Ws["standard"] ** 2).max())
out["note"] = ("multiset of Walsh energies is invariant under both layouts; "
               "only the |s| labelling changes")

# ---- bit-level variance mass ---------------------------------------
W = Ws["standard"]
var = W.var(axis=0)
tot = var.sum()
idx = np.arange(4096)
per_bit = {}
for b in range(12):
    m = ((idx >> b) & 1).astype(bool)
    name = ("y%d" % b) if b < 6 else ("x%d" % (b - 6))
    per_bit[name] = float(var[m].sum() / tot)
out["variance_mass_touching_bit"] = per_bit

# subsets restricted to the 6 coarse bits {y3,y4,y5,x3,x4,x5}
coarse_bits = [3, 4, 5, 9, 10, 11]
coarse_mask = 0
for b in coarse_bits:
    coarse_mask |= (1 << b)
only_coarse = (idx & ~coarse_mask) == 0            # 64 subsets
out["coarse6"] = {
    "n_subsets": int(only_coarse.sum()),
    "variance_frac": float(var[only_coarse].sum() / tot),
    "energy_frac": float((W ** 2).mean(0)[only_coarse].sum() / (W ** 2).mean(0).sum()),
}
# same for the 6 fine bits {y0..y2,x0..x2}
fine_mask = 0
for b in [0, 1, 2, 6, 7, 8]:
    fine_mask |= (1 << b)
only_fine = (idx & ~fine_mask) == 0
out["fine6"] = {
    "n_subsets": int(only_fine.sum()),
    "variance_frac": float(var[only_fine].sum() / tot),
    "energy_frac": float((W ** 2).mean(0)[only_fine].sum() / (W ** 2).mean(0).sum()),
}
# top-50 data-adaptive: how many touch a fine bit?
top50 = np.argsort(var)[::-1][:50]
out["top50_touching_a_fine_bit"] = int(((top50 & fine_mask) != 0).sum())
out["top200_touching_a_fine_bit"] = int(
    ((np.argsort(var)[::-1][:200] & fine_mask) != 0).sum())
out["top50_within_coarse6"] = int(((top50 & ~coarse_mask) == 0).sum())

# ---- if we only ever needed a 32x32 image (drop the 2 finest bits) ---
img32 = allimg.reshape(-1, 32, 2, 32, 2).sum(axis=(2, 4))
p32 = img32.reshape(-1, 1024)
p32 = p32 / p32.sum(1, keepdims=True)
W32 = fwht(p32)
PC10 = popcount_table(10)
E32 = (W32 ** 2).mean(0)
sp = np.zeros(11)
for k in range(11):
    sp[k] = E32[PC10 == k].sum()
sp /= sp.sum()
out["downsample32"] = {
    "order_spectrum": sp.tolist(),
    "mean_|s|": float((sp * np.arange(11)).sum()),
    "energy_k<=4": float(sp[:5].sum()),
    "energy_k<=6": float(sp[:7].sum()),
    "n_subsets_le4_10bit": int(1 + 10 + 45 + 120 + 210),
}

with open(os.path.join(HERE, "results_bits.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
