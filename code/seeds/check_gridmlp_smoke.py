"""Read-back checks on the smoke outputs: contract, loss decrease, projection, non-linearity."""
import json, os, sys
import numpy as np

root = sys.argv[1]
bad = []


def check(name, ok, detail=""):
    print("  %-46s %s %s" % (name, "OK " if ok else "FAIL", detail))
    if not ok:
        bad.append(name)


for tag, frame, ncodes in (("tiny_n12", 64, 4096), ("tiny_mlp", 64, 1024),
                           ("tiny_mlpzs", 64, 1024), ("tiny_mlplin", 64, 1024)):
    d = os.path.join(root, tag)
    print("[check] %s" % d)
    m = json.load(open(os.path.join(d, "meta.json")))
    gen = sorted(f for f in os.listdir(d) if f.startswith("generated_"))
    a = np.load(os.path.join(d, gen[0]))
    crops = sorted(f for f in os.listdir(os.path.join(d, "crops")))
    c = np.load(os.path.join(d, "crops", crops[0]))
    curve = np.load(os.path.join(d, "loss_curve.npy"))
    th0 = np.load(os.path.join(d, "theta0.npy"))
    phi = np.load(os.path.join(d, "Phi.npy"))
    check("%s: 8 frames written" % tag, len(gen) == 8, str(len(gen)))
    check("%s: frame shape %dx%d" % (tag, frame, frame), a.shape == (frame, frame), str(a.shape))
    check("%s: crop shape 32x32" % tag, c.shape == (32, 32), str(c.shape))
    check("%s: frames finite and >= 0" % tag,
          bool(np.isfinite(a).all() and a.min() >= 0), "min=%.3g max=%.3g" % (a.min(), a.max()))
    check("%s: theta0 has %d entries" % (tag, ncodes), th0.size == ncodes, str(th0.shape))
    check("%s: Phi is (M, %d)" % (tag, ncodes), phi.shape == (4, ncodes), str(phi.shape))
    check("%s: loss fell" % tag, float(curve[-1, 1]) < float(curve[0, 1]),
          "%.4e -> %.4e" % (curve[0, 1], curve[-1, 1]))
    check("%s: train corr > 0.5" % tag, float(m["train_corr_mean"]) > 0.5,
          "%.4f" % m["train_corr_mean"])
    check("%s: projection ran" % tag, m["projection_corr_mean"] is not None,
          "corr=%.4f" % (m["projection_corr_mean"] or -1))
    check("%s: total flux ~ S" % tag, abs(a.sum() / m["S"] - 1.0) < 0.05,
          "gen/S=%.4f" % (a.sum() / m["S"]))
    if "nonlinearity_ratio" in m:
        print("      nonlinearity_ratio=%.4f  pre|.|mean=%.3f  frac>1=%.3f  params=%d"
              % (m["nonlinearity_ratio"], m["g_stats"]["pre_abs_mean"],
                 m["g_stats"]["pre_frac_gt1"], m["param_count"]))
    else:
        print("      param_count=%d (shared %d)" % (m["param_count"], m["param_count_shared"]))

# the intended contrast: a wide input scale must bend tanh, a unit one must not
m_nl = json.load(open(os.path.join(root, "tiny_mlp", "meta.json")))
m_li = json.load(open(os.path.join(root, "tiny_mlplin", "meta.json")))
check("MLP non-linearity is actually exercised",
      m_nl["nonlinearity_ratio"] > 5.0 * max(m_li["nonlinearity_ratio"], 1e-9),
      "input-scale 20 -> %.4f vs input-scale 1 -> %.5f"
      % (m_nl["nonlinearity_ratio"], m_li["nonlinearity_ratio"]))
m_zs = json.load(open(os.path.join(root, "tiny_mlpzs", "meta.json")))
check("zscale locks ||g(z)|| = PHI*||z||",
      abs(m_zs["g_init"]["g_norm_over_phi_norm_init"] - 1.0) < 1e-6,
      "%.8f" % m_zs["g_init"]["g_norm_over_phi_norm_init"])
check("rownorm calibrated to the same magnitude at init",
      abs(m_nl["g_init"]["g_norm_over_phi_norm_init"] - 1.0) < 1e-6,
      "%.8f" % m_nl["g_init"]["g_norm_over_phi_norm_init"])

print("\n[check] %s" % ("ALL OK" if not bad else "FAILURES: %s" % bad))
sys.exit(1 if bad else 0)
