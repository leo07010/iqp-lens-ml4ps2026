"""Seconds-level smoke + correctness tests for the Fig. 3 / Fig. 4 controlled studies.

Runs on the login node (CPU only, n <= 8).  Two kinds of check:

  (a) CORRECTNESS -- the fast batched paths added in `common.py` must agree, to numerical
      precision, with the already-audited references in the parent directory:
          common.BatchIQP.probs   vs  iqp_exact.IQPExact.probs        (same masks)
          common.qcli_torch       vs  diagnostics_qcli_cci.qcli_exact
          common.cci_batch        vs  diagnostics_qcli_cci.cci_exact
          common.mmd2_from_expvals vs iqp_exact.IQPExact.mmd2
      If any of these drift, both figures are wrong, so they are asserted, not printed.

  (b) SMOKE -- `fig3_support_mismatch.main` and `fig4_qcli_max.main` run end to end at n = 8
      with tiny budgets and produce well-formed jsonl.

Usage:  /usr/bin/python3 test_controlled.py      (or: pytest test_controlled.py)
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (BatchIQP, cci_batch, empirical_pmf, kernel_c_by_weight,  # noqa: E402
                    mmd2_from_expvals, n_available_masks, popcount_index, qcli_torch,
                    qcli_numpy_batch, random_gate_masks, segment_by_weight, spsa_maximize)

sys.path.insert(0, common.REPL_DIR)
import diagnostics_qcli_cci as diag  # noqa: E402
from iqp_exact import IQPExact, subset_masks  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 1))


# =======================================================================================
# (a) correctness against the audited references
# =======================================================================================
def test_batchiqp_matches_iqpexact():
    """With masks = "all |s| <= K", BatchIQP must reproduce IQPExact exactly."""
    n, K = 6, 2
    ref = IQPExact(n, K, dtype=torch.complex128)
    masks = subset_masks(n, K)                       # includes the empty subset at index 0
    bat = BatchIQP(n, masks, dtype=torch.float64)
    rng = np.random.default_rng(0)
    theta = torch.as_tensor(rng.normal(0.0, 0.7, size=ref.n_params), dtype=torch.float64)
    p_ref = ref.probs(theta).to(torch.float64).numpy()
    p_bat = bat.probs(theta[None, :])[0].numpy()
    err = float(np.abs(p_ref - p_bat).max())
    print("  max |probs_BatchIQP - probs_IQPExact| = %.3e" % err)
    assert err < 1e-12, err
    e_ref = ref.expvals(theta).to(torch.float64).numpy()
    e_bat = bat.expvals(theta[None, :])[0].numpy()
    assert float(np.abs(e_ref - e_bat).max()) < 1e-12


def test_qcli_torch_matches_qcli_exact():
    n = 8
    rng = np.random.default_rng(1)
    masks = np.stack([random_gate_masks(n, 3, 40, rng)[0] for _ in range(4)])
    sim = BatchIQP(n, masks, dtype=torch.float64)
    theta = sim.init_theta(4, rng, std=0.8)
    p = sim.probs(theta)
    got = qcli_torch(p, n, popcount_index(n, "cpu")).numpy()
    ref = qcli_numpy_batch(p.numpy(), n)
    err = float(np.abs(got - ref).max())
    print("  QCLI values %s ; max abs diff vs qcli_exact = %.3e" % (np.round(got, 5).tolist(), err))
    assert err < 1e-12, err
    # the two analytically known corner cases (see common.py docstring)
    pm = torch.zeros(1, 1 << n, dtype=torch.float64)
    pm[0, 37] = 1.0
    q_point = float(qcli_torch(pm, n, popcount_index(n, "cpu"))[0])
    pu = torch.full((1, 1 << n), 1.0 / float(1 << n), dtype=torch.float64)
    q_unif = float(qcli_torch(pu, n, popcount_index(n, "cpu"))[0])
    print("  corner cases: I_QCLI(point mass) = %.3e (exactly 0), "
          "I_QCLI(exact uniform) = %.4f (the indicator's max, NOT 0)" % (q_point, q_unif))
    assert q_point < 1e-12
    assert q_unif > 0.9


def test_cci_batch_matches_cci_exact():
    """The Walsh-coefficient shortcut for the marginals must reproduce cci_exact bit for bit."""
    n = 8
    rng = np.random.default_rng(2)
    masks = np.stack([random_gate_masks(n, 2, 30, rng)[0] for _ in range(3)])
    sim = BatchIQP(n, masks, dtype=torch.float64)
    theta = sim.init_theta(3, rng, std=1.0)
    p = sim.probs(theta)
    fast = cci_batch(p, n)
    ref = [diag.cci_exact(p[b].numpy(), n) for b in range(3)]
    for key in ("I_CCI", "I_TC", "I_TC_tree"):
        err = max(abs(float(fast[key][b]) - float(ref[b][key])) for b in range(3))
        print("  max |%s_fast - %s_exact| = %.3e" % (key, key, err))
        assert err < 1e-9, (key, err)
    # empirical (sparse) pmf path too -- this is what Fig. 3 feeds it
    codes = sim.sample(theta, 4000, generator=torch.Generator().manual_seed(5)).numpy()
    pe = np.stack([empirical_pmf(codes[b], n) for b in range(3)])
    fast_e = cci_batch(pe, n)
    ref_e = [diag.cci_exact(pe[b], n) for b in range(3)]
    err = max(abs(float(fast_e["I_CCI"][b]) - float(ref_e[b]["I_CCI"])) for b in range(3))
    print("  empirical-pmf path: max |I_CCI_fast - I_CCI_exact| = %.3e" % err)
    assert err < 1e-9, err


def test_mmd2_matches_iqp_exact_mmd2():
    """The weight-grouped MMD must equal the per-subset sum of iqp_exact.IQPExact.mmd2."""
    n, K = 6, 3
    ref = IQPExact(n, K, dtype=torch.complex128)
    masks = subset_masks(n, K)
    bat = BatchIQP(n, masks, dtype=torch.float64)
    rng = np.random.default_rng(3)
    theta = torch.as_tensor(rng.normal(0.0, 0.5, size=ref.n_params), dtype=torch.float64)
    data_codes = rng.integers(0, 1 << n, size=3000, dtype=np.int64)
    pmf = empirical_pmf(data_codes, n)
    sigmas = [1.1, 1.9, 2.7]

    ez_data_ref = ref.data_expvals(pmf)
    w_ref = ref.kernel_weights(sigmas)
    m_ref = float(ref.mmd2(theta, ez_data_ref, w_ref))

    pc = popcount_index(n, "cpu")
    ez_data = torch.as_tensor(pmf, dtype=torch.float64)[None, :]
    ez_data = common.fwht(ez_data)
    cw = torch.as_tensor(kernel_c_by_weight(n, sigmas), dtype=torch.float64)[None, :]
    m_bat = float(mmd2_from_expvals(bat.expvals(theta[None, :]), ez_data, cw, n, pc)[0])
    print("  MMD^2 reference = %.12e ; batched = %.12e ; rel diff = %.3e"
          % (m_ref, m_bat, abs(m_ref - m_bat) / max(m_ref, 1e-30)))
    assert abs(m_ref - m_bat) / max(m_ref, 1e-30) < 1e-10


def test_random_gate_masks_distinct_and_capped():
    rng = np.random.default_rng(4)
    for n in (8, 12, 16):
        m, eff = random_gate_masks(n, 2, 150, rng)
        avail = n_available_masks(n, 2)
        print("  n=%2d: requested 150 <=2-local gates, available %3d, used %3d" % (n, avail, eff))
        assert eff == min(150, avail)
        assert np.unique(m).size == eff          # distinct
        assert m.min() > 0                       # empty subset excluded
        pc = np.array([bin(int(x)).count("1") for x in m])
        assert pc.max() <= 2 and pc.min() >= 1
    m4, eff4 = random_gate_masks(16, 4, 1050, rng)
    assert eff4 == 1050 and np.unique(m4).size == 1050


def test_spsa_increases_qcli():
    """The Fig. 4 mechanism in miniature: SPSA on I_QCLI alone must raise I_QCLI, and I_CCI
    (never in the objective) must come along."""
    n = 8
    rng = np.random.default_rng(5)
    masks = np.stack([random_gate_masks(n, 2, 150, rng)[0] for _ in range(3)])
    sim = BatchIQP(n, masks, dtype=torch.float64)
    pc = popcount_index(n, "cpu")
    theta = sim.init_theta(3, rng, std=0.05)

    def obj(t):
        return qcli_torch(sim.probs(t), n, pc)

    q0 = obj(theta).numpy()
    c0 = cci_batch(sim.probs(theta), n)["I_CCI"]
    gen = torch.Generator()
    gen.manual_seed(11)
    t1 = spsa_maximize(obj, theta, 300, a=2.0, c=0.2, generator=gen)
    q1 = obj(t1).numpy()
    c1 = cci_batch(sim.probs(t1), n)["I_CCI"]
    print("  I_QCLI %s -> %s" % (np.round(q0, 4).tolist(), np.round(q1, 4).tolist()))
    print("  I_CCI  %s -> %s  (never optimised)" % (np.round(c0, 4).tolist(), np.round(c1, 4).tolist()))
    assert float(q1.mean()) > float(q0.mean()) + 0.05
    # I_CCI must follow for the majority of the trajectories.  Not for ALL of them: pushed far
    # enough, the QCLI-only objective walks toward the exact uniform distribution, whose I_CCI
    # is 0 -- the coupling the paper reports is real but not monotone all the way to the
    # indicator's ceiling, which is exactly what Fig. 4's second panel is there to show.
    risen = int((c1 > c0).sum())
    print("  I_CCI rose in %d of %d trajectories" % (risen, c1.size))
    assert risen >= (c1.size + 1) // 2


def test_uniform_null_floor():
    """The QCLI x-axis has a sample-size-dependent floor; the scripts must report it."""
    n = 8
    for M in (500, 5000):
        mean, std = common.uniform_null_qcli(n, M, 3, 7)
        print("  n=8, M=%5d i.i.d.-uniform null: I_QCLI = %.4f +/- %.4f" % (M, mean, std))
        assert mean > 0.0


# =======================================================================================
# (b) end-to-end smoke of the two scripts
# =======================================================================================
def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_fig3_smoke():
    import fig3_support_mismatch as f3
    d = tempfile.mkdtemp(prefix="fig3_smoke_")
    out = os.path.join(d, "fig3.jsonl")
    t0 = time.time()
    rc = f3.main(["--n", "8", "--gates-target", "20,40", "--learner-gates", "8,16",
                  "--n-targets", "4", "--n-samples", "500", "--target-steps", "60",
                  "--steps", "20", "--train-batch", "8",
                  "--null-reps", "2", "--progress-every", "0", "--out", out])
    assert rc == 0
    rows = _read_jsonl(out)
    kinds = {}
    for r in rows:
        kinds[r["record"]] = kinds.get(r["record"], 0) + 1
    print("  fig3 smoke %.1fs, rows by kind: %s" % (time.time() - t0, kinds))
    assert kinds.get("target") == 8                 # 2 gate counts x 4 targets
    assert kinds.get("fit") == 16                   # x 2 learners
    assert kinds.get("null") == 1 and kinds.get("meta") == 1
    for r in rows:
        if r["record"] != "fit":
            continue
        for k in ("target_id", "gates_target", "I_QCLI", "I_CCI", "I_QCLI_exact",
                  "learner_gates", "final_mmd", "final_mmd2", "init_mmd2", "mmd2_floor_proxy",
                  "final_mmd2_vs_exact", "final_mmd2_exacttrained"):
            assert k in r, k
        assert r["final_mmd2"] >= 0.0 and math.isfinite(r["final_mmd2"])
        assert r["final_mmd2"] <= r["init_mmd2"] * 1.5   # Adam must not blow the loss up
        assert r["final_mmd2_exacttrained"] >= 0.0 and math.isfinite(r["final_mmd2_exacttrained"])
    # v2: the target levels are laid out on the EXACT I_QCLI, so the targets must actually be
    # spread on that axis (v1 laid them out on a metric where they all collapsed to one point).
    tg = [r for r in rows if r["record"] == "target"]
    qx = sorted(float(r["I_QCLI_exact"]) for r in tg)
    print("  fig3 target exact I_QCLI span %.4f..%.4f over %d targets" % (qx[0], qx[-1], len(qx)))
    assert qx[-1] - qx[0] > 0.2, qx


def test_fig4_smoke():
    import fig4_qcli_max as f4
    d = tempfile.mkdtemp(prefix="fig4_smoke_")
    out = os.path.join(d, "fig4.jsonl")
    t0 = time.time()
    rc = f4.main(["--sizes", "8", "--seeds", "2", "--steps", "40", "--record-every", "4",
                  "--spsa-a", "0.5,2.0", "--progress-every", "0", "--out", out])
    assert rc == 0
    rows = _read_jsonl(out)
    traj = [r for r in rows if r["record"] == "traj"]
    calib = [r for r in rows if r["record"] == "calibration"]
    print("  fig4 smoke %.1fs, %d traj rows, %d calibration rows" % (time.time() - t0,
                                                                    len(traj), len(calib)))
    assert len(calib) == 1 and calib[0]["I_CCI"] == 0.0     # exact uniform is tree-explainable
    assert len(traj) == 2 * 2 * (1 + 40 // 4)               # arms x seeds x (step 0 + records)
    for r in traj:
        for k in ("n", "arm", "seed_idx", "step", "I_QCLI", "I_CCI", "gates_effective"):
            assert k in r, k
        assert 0.0 <= r["I_QCLI"] <= 1.0 and -1e-9 <= r["I_CCI"] <= 1.0 + 1e-9
    # trend inside the strong arm
    strong = [r for r in traj if r["arm"] == 2.0]
    first = [r["I_QCLI"] for r in strong if r["step"] == 0]
    last = max(r["step"] for r in strong)
    final = [r["I_QCLI"] for r in strong if r["step"] == last]
    print("  fig4 strong arm I_QCLI %s -> %s" % (np.round(first, 4).tolist(),
                                                 np.round(final, 4).tolist()))
    assert float(np.mean(final)) > float(np.mean(first))


def test_plot_smoke():
    """plot_fig34.py must produce both PNGs from the smoke jsonl (it runs at the end of the job)."""
    import fig3_support_mismatch as f3
    import fig4_qcli_max as f4
    import plot_fig34 as pl
    d = tempfile.mkdtemp(prefix="plot_smoke_")
    o3 = os.path.join(d, "fig3.jsonl")
    o4 = os.path.join(d, "fig4.jsonl")
    f3.main(["--n", "8", "--gates-target", "20,40", "--learner-gates", "8,16",
             "--n-targets", "8", "--n-samples", "500", "--target-steps", "80",
             "--steps", "20", "--train-batch", "16",
             "--null-reps", "2", "--progress-every", "0", "--out", o3])
    f4.main(["--sizes", "8", "--seeds", "2", "--steps", "40", "--record-every", "4",
             "--spsa-a", "0.5,2.0", "--progress-every", "0", "--out", o4])
    p3 = os.path.join(d, "fig3.png")
    p4 = os.path.join(d, "fig4.png")
    rc = pl.main(["--fig3", o3, "--fig4", o4, "--out-fig3", p3, "--out-fig4", p4])
    assert rc == 0
    for p in (p3, p4):
        assert os.path.exists(p) and os.path.getsize(p) > 5000, p
        print("  wrote %s (%d bytes)" % (os.path.basename(p), os.path.getsize(p)))


def main():
    tests = [
        test_batchiqp_matches_iqpexact,
        test_qcli_torch_matches_qcli_exact,
        test_cci_batch_matches_cci_exact,
        test_mmd2_matches_iqp_exact_mmd2,
        test_random_gate_masks_distinct_and_capped,
        test_spsa_increases_qcli,
        test_uniform_null_floor,
        test_fig3_smoke,
        test_fig4_smoke,
        test_plot_smoke,
    ]
    t0 = time.time()
    for t in tests:
        print("--- %s ---" % t.__name__)
        t()
        print("PASS\n")
    print("ALL TESTS PASSED (%.1f s)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
