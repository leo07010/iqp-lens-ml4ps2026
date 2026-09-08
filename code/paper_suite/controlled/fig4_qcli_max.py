"""Fig. 4 of arXiv 2603.06440 -- does I_CCI rise when only I_QCLI is optimised?

Paper setting (Table 4, `paper.txt:2809-2831`; Sec. II.7, `paper.txt:930-948`):

    System sizes       n in {8, 12, 16}
    Architecture       IQP circuits with 150 randomly selected 2-local parameterised gates
    Init / optimiser   random initialisation; SPSA
    Objective          maximise I_QCLI ONLY; I_CCI never enters the objective
    Sampling           20000 IQP circuits per system size, taken along the optimisation
    Evaluation         I_QCLI and I_CCI of every sampled circuit

Prediction under test: I_CCI increases together with I_QCLI even though it is never optimised.

Implementation notes / deviations, all recorded in the output:
  * "150 2-local gates" does not exist at these sizes: there are only C(n,1)+C(n,2) = 36 / 78 /
    136 distinct <=2-local Pauli-Z strings at n = 8 / 12 / 16, and repeated gates in an IQP
    circuit merely add their angles.  We use every distinct string up to the requested budget
    and record `gates_effective` (see `common.random_gate_masks`).
  * "sampled circuits" are parameter snapshots along the SPSA trajectory.  With
    `--seeds 3 --steps 6667 --record-every 1` this yields ~20000 circuits per system size,
    matching Table 4's count.
  * I_QCLI and I_CCI are computed from each circuit's EXACT output distribution (n <= 16, so
    the 2^n pmf is affordable).  The paper does not say whether it used the exact pmf or a
    finite sample; the exact pmf is the only choice that makes I_CCI meaningful at n = 16,
    where a finite sample cannot populate 65536 outcomes.  For calibration the script also
    reports the exact-pmf I_QCLI/I_CCI of the i.i.d.-uniform and of a random unoptimised
    circuit at step 0.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (BatchIQP, cci_batch, popcount_index, qcli_torch,  # noqa: E402
                    random_gate_masks, spsa_maximize)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fig. 4: maximise QCLI only, watch CCI follow.")
    p.add_argument("--sizes", default="8,12,16", help="qubit counts (paper Table 4)")
    p.add_argument("--gates", type=int, default=150, help="requested 2-local gates (paper: 150)")
    p.add_argument("--locality", type=int, default=2)
    p.add_argument("--seeds", type=int, default=3, help="random initialisations per size")
    p.add_argument("--steps", type=int, default=3333,
                   help="SPSA steps per arm; seeds*arms*steps ~ the paper's 20000 circuits/size")
    p.add_argument("--record-every", type=int, default=1)
    p.add_argument("--spsa-a", default="0.5,2.0",
                   help="comma-separated SPSA gains, one arm each.  The small gain reproduces "
                        "the paper's I_QCLI range (up to ~0.7); the large gain runs the "
                        "optimisation to the indicator's degenerate ceiling (the exact uniform "
                        "distribution, I_QCLI ~ 0.98 at n=8), which is where the QCLI-CCI "
                        "coupling can be checked for non-monotonicity.")
    p.add_argument("--spsa-c", type=float, default=0.2)
    p.add_argument("--init-std", type=float, default=0.05,
                   help="std of the N(0,.) random initialisation")
    p.add_argument("--dtype", choices=tuple(_DTYPES), default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--out", default="results_fig4.jsonl")
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args(argv)


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


class JsonlWriter(object):
    def __init__(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self.f = open(path, "w")
        self.n_rows = 0

    def write(self, row):
        self.f.write(json.dumps({k: _jsonable(v) for k, v in row.items()}) + "\n")
        self.n_rows += 1
        if self.n_rows % 200 == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()


def run_size(args, n, w, device, dtype, log):
    """One system size.  Each SPSA gain ("arm") runs `--seeds` trajectories as one batch and is
    recorded along the way; seeds x arms x steps circuits are written for this size."""
    rng = np.random.default_rng(int(args.seed) + 1000 * n)
    B = int(args.seeds)
    pc_idx = popcount_index(n, device)
    arms = [float(x) for x in str(args.spsa_a).split(",") if x.strip()]
    avail = common.n_available_masks(n, args.locality)

    # calibration point: the exact i.i.d.-uniform distribution, which is the argmax of Eq. 6-7
    # (m -> delta_0) and has I_CCI = 0 -- the ceiling against which the trend must be read.
    p_unif = torch.full((1, 1 << n), 1.0 / float(1 << n), dtype=dtype, device=device)
    with torch.no_grad():
        q_unif = float(qcli_torch(p_unif, n, pc_idx)[0])
    c_unif = float(cci_batch(p_unif, n)["I_CCI"][0])
    w.write({"record": "calibration", "n": n, "which": "exact_uniform",
             "I_QCLI": q_unif, "I_CCI": c_unif})
    log("[fig4 n=%d] exact-uniform calibration: I_QCLI=%.4f I_CCI=%.4f" % (n, q_unif, c_unif))

    out = []
    for arm_i, a in enumerate(arms):
        masks = np.stack([random_gate_masks(n, args.locality, args.gates, rng)[0] for _ in range(B)])
        g_eff = int(masks.shape[1])
        log("[fig4 n=%d arm a=%.3g] %d seeds; gates requested %d -> %d distinct <=%d-local "
            "strings (only %d exist)" % (n, a, B, args.gates, g_eff, args.locality, avail))
        sim = BatchIQP(n, masks, device=device, dtype=dtype)
        theta = sim.init_theta(B, rng, std=args.init_std)
        gen = torch.Generator(device=torch.device(device))
        gen.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))

        def objective(th, _sim=sim):
            return qcli_torch(_sim.probs(th), n, pc_idx)

        state = {"t0": time.time(), "n_rec": 0}

        def record(step, th, _sim=sim, _a=a, _g=g_eff, _st=state):
            with torch.no_grad():
                p = _sim.probs(th)
                q = qcli_torch(p, n, pc_idx).cpu().numpy()
            c = cci_batch(p, n)
            for b in range(B):
                w.write({"record": "traj", "n": n, "arm": _a, "seed_idx": b, "step": int(step),
                         "I_QCLI": float(q[b]), "I_CCI": float(c["I_CCI"][b]),
                         "I_TC": float(c["I_TC"][b]), "I_TC_tree": float(c["I_TC_tree"][b]),
                         "gates_requested": int(args.gates), "gates_effective": _g})
            _st["n_rec"] += B
            return q, c["I_CCI"]

        q0, c0 = record(0, theta)
        log("[fig4 n=%d arm a=%.3g] step 0 (random circuit): I_QCLI %s  I_CCI %s"
            % (n, a, np.round(q0, 4).tolist(), np.round(c0, 4).tolist()))

        every = max(1, int(args.record_every))

        def cb(step, th, val, _a=a, _st=state):
            if (step + 1) % every == 0 or step == args.steps - 1:
                record(step + 1, th)
            if args.progress_every and ((step + 1) % args.progress_every == 0
                                        or step == args.steps - 1):
                dt = time.time() - _st["t0"]
                log("  [fig4 n=%d a=%.3g] step %5d/%d  QCLI(mid) %s  %.2f ms/step  %d rows"
                    % (n, _a, step + 1, args.steps,
                       np.round(val.detach().cpu().numpy(), 4).tolist(),
                       1e3 * dt / float(step + 1), _st["n_rec"]))

        theta = spsa_maximize(objective, theta, args.steps, a=a, c=args.spsa_c,
                              generator=gen, callback=cb, maximize=True)
        dt = time.time() - state["t0"]
        with torch.no_grad():
            pf = sim.probs(theta)
            qf = qcli_torch(pf, n, pc_idx).cpu().numpy()
        cf = cci_batch(pf, n)["I_CCI"]
        log("[fig4 n=%d a=%.3g] done: %d steps in %.1fs (%.2f ms/step), %d circuits; final "
            "I_QCLI %s  I_CCI %s" % (n, a, args.steps, dt, 1e3 * dt / float(args.steps),
                                     state["n_rec"], np.round(qf, 4).tolist(),
                                     np.round(cf, 4).tolist()))
        out.append({"n": n, "arm": a, "gates_effective": g_eff, "n_records": state["n_rec"],
                    "ms_per_step": 1e3 * dt / float(args.steps), "wall_seconds": dt,
                    "final_I_QCLI": qf.tolist(), "final_I_CCI": cf.tolist()})
        del sim
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device(args.device)
    dtype = _DTYPES[args.dtype]
    torch.manual_seed(int(args.seed))
    sizes = [int(x) for x in str(args.sizes).split(",") if x.strip()]

    def log(msg):
        print(msg, flush=True)

    w = JsonlWriter(args.out)
    t_all = time.time()
    log("[fig4] sizes=%s seeds=%d steps=%d record_every=%d dtype=%s device=%s"
        % (sizes, args.seeds, args.steps, args.record_every, args.dtype, args.device))
    summ = []
    for n in sizes:
        summ.extend(run_size(args, n, w, device, dtype, log))
    w.write({"record": "meta", "sizes": sizes, "seeds": int(args.seeds), "steps": int(args.steps),
             "record_every": int(args.record_every), "gates_requested": int(args.gates),
             "locality": int(args.locality), "seed": int(args.seed), "dtype": args.dtype,
             "device": str(args.device), "per_size": summ,
             "wall_seconds": time.time() - t_all})
    w.close()
    log("[fig4] wrote %s in %.1f s" % (args.out, time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
