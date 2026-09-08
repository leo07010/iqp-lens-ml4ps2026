#!/usr/bin/env python3
"""Turn the convergence-sweep evaluation into results_convergence.jsonl + a best-of-steps table.

Input : the jsonl written by evaluate_r3b.py over outputs_convergence/ (one row per snapshot),
        plus each snapshot's meta.json (for `steps` and the training-loss column) and the two
        round-3 result files (for the Table-2 IQP reference values).
Output: results_convergence.jsonl -- one row per (arm, n_train, steps) with exactly the columns
        the addendum quotes -- and a markdown table on stdout.

The reference row is the BEST Table-2 IQP entry per column, recomputed here from
results_r3.jsonl / results_r3b.jsonl rather than typed in, and checked against the published
values so a change in either file is caught instead of silently propagating.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import json
import os

# The four IQP rows of Table 2 (main.tex, \label{tab:main}) and the file each comes from.
TABLE2_IQP = {
    "IQP-lowvar-paper-MMD": "results_r3.jsonl",
    "IQP-lowvar-joint-MMD": "results_r3.jsonl",
    "IQP-affine-M4-MMDflat-resample": "results_r3b.jsonl",
    "IQP-affine-M4-auto3-resample": "results_r3b.jsonl",
}
# Published best-IQP values per column, from Table 2. (n_train -> metric -> value)
TABLE2_IQP_BEST = {
    11:  {"pdf_js": 0.00095, "conv_mmd2": 0.1134, "conv_mmd2_aligned": 0.0279},
    100: {"pdf_js": 0.00092, "conv_mmd2": 0.0861, "conv_mmd2_aligned": 0.0275},
}
# Published baseline values at the fixed 20k budget, from Table 2.
TABLE2_BASE_20K = {
    ("RBM-10bit", 11):   {"pdf_js": 0.00632, "conv_mmd2": 0.0741, "conv_mmd2_aligned": 0.1240},
    ("RBM-10bit", 100):  {"pdf_js": 0.00195, "conv_mmd2": 0.0752, "conv_mmd2_aligned": 0.1144},
    ("DCGAN-32", 11):    {"pdf_js": 0.00183, "conv_mmd2": 0.2165, "conv_mmd2_aligned": 0.1241},
    ("DCGAN-32", 100):   {"pdf_js": 0.00763, "conv_mmd2": 0.0371, "conv_mmd2_aligned": 0.0694},
}
METRICS = ("pdf_js", "conv_mmd2", "conv_mmd2_aligned", "nn_holdout")
OUT_KEYS = ("arm", "arm_base", "n_train", "steps", "pdf_js", "conv_mmd2", "conv_mmd2_aligned",
            "nn_holdout", "pdf_js_aligned", "nn_train", "nn_holdout_aligned", "nn_within",
            "recon_mse", "lossD", "lossG", "gen_mean_total", "holdout_mean_total",
            "param_count", "seed", "run_dir")


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iqp_reference(repo):
    """Best Table-2 IQP value per (N, metric), recomputed from the round-3 result files."""
    pools = {}
    for arm, fname in TABLE2_IQP.items():
        path = os.path.join(repo, fname)
        for r in read_jsonl(path):
            if r.get("arm") != arm:
                continue
            n = int(r["n_train"])
            for m in ("pdf_js", "conv_mmd2", "conv_mmd2_aligned"):
                v = r.get(m)
                if v is None:
                    continue
                key = (n, m)
                if key not in pools or float(v) < pools[key][0]:
                    pools[key] = (float(v), arm)
    out = {}
    for (n, m), (v, arm) in pools.items():
        out.setdefault(n, {})[m] = {"value": v, "arm": arm}
    for n, want in TABLE2_IQP_BEST.items():
        for m, published in want.items():
            got = out.get(n, {}).get(m)
            if got is None:
                raise SystemExit("aggregate_convergence.py: no IQP value for N=%d %s" % (n, m))
            if abs(got["value"] - published) > 0.6 * 10 ** -_digits(published):
                raise SystemExit("aggregate_convergence.py: IQP N=%d %s is %.6g but Table 2 "
                                 "says %.6g -- the round-3 files changed" % (n, m, got["value"],
                                                                             published))
    return out


def _digits(x):
    """Number of decimals in the published (already rounded) value, for the tolerance check."""
    s = ("%g" % x)
    return len(s.split(".")[1]) if "." in s else 0


def build_rows(eval_rows, repo):
    rows = []
    for r in eval_rows:
        rd = r.get("run_dir")
        if not rd:
            continue                      # reference rows (HOLDOUT split-half) have no run dir
        meta_path = os.path.join(rd, "meta.json") if os.path.isabs(rd) else os.path.join(repo, rd, "meta.json")
        if not os.path.isfile(meta_path):
            raise SystemExit("aggregate_convergence.py: missing %s" % meta_path)
        with open(meta_path) as fh:
            meta = json.load(fh)
        if meta.get("sweep") != "convergence":
            continue                      # anything that is not a sweep snapshot
        out = {k: None for k in OUT_KEYS}
        out.update({
            "arm": meta.get("arm"), "arm_base": meta.get("arm_base"),
            "n_train": int(r["n_train"]), "steps": int(meta["steps"]),
            "run_dir": rd, "param_count": meta.get("param_count"), "seed": meta.get("seed"),
        })
        for k in ("pdf_js", "conv_mmd2", "conv_mmd2_aligned", "nn_holdout", "pdf_js_aligned",
                  "nn_train", "nn_holdout_aligned", "nn_within", "gen_mean_total",
                  "holdout_mean_total"):
            out[k] = r.get(k)
        if meta.get("script") == "baseline_rbm_sweep.py":
            out["recon_mse"] = meta.get("recon_mse_last")
        else:
            out["lossD"] = meta.get("lossD_last")
            out["lossG"] = meta.get("lossG_last")
        rows.append(out)
    rows.sort(key=lambda z: (str(z["arm_base"]), z["n_train"], z["steps"]))
    return rows


def best_table(rows, ref):
    """markdown: per (arm, N), the 20k value next to the best-over-steps value and its step."""
    lines = ["| Arm | N | metric | 20k | best over steps | best @ steps | IQP best (Table 2) | "
             "does the longer run beat IQP? |", "|" + "---|" * 8]
    groups = {}
    for r in rows:
        groups.setdefault((r["arm_base"], r["n_train"]), []).append(r)
    label = {"pdf_js": "PDF-JS", "conv_mmd2": "MMD^2", "conv_mmd2_aligned": "MMD^2_al",
             "nn_holdout": "NN-holdout"}
    for (arm, n) in sorted(groups):
        pts = sorted(groups[(arm, n)], key=lambda z: z["steps"])
        at20 = [p for p in pts if p["steps"] == 20000]
        for m in METRICS:
            vals = [(p[m], p["steps"]) for p in pts if p.get(m) is not None]
            if not vals:
                continue
            bv, bs = min(vals)
            v20 = at20[0][m] if at20 else None
            iq = ref.get(n, {}).get(m)
            iqv = iq["value"] if iq else None
            verdict = "n/a (not a Table 2 column)" if iqv is None else (
                "YES, baseline now lower" if bv < iqv else "no, IQP still lower")
            lines.append("| %s | %d | %s | %s | %.5g | %d | %s | %s |"
                         % (arm, n, label[m], ("%.5g" % v20) if v20 is not None else "—",
                            bv, bs, ("%.5g" % iqv) if iqv is not None else "—", verdict))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aggregate the baseline convergence sweep.")
    p.add_argument("--eval-results", default="results_convergence_eval.jsonl")
    p.add_argument("--out", default="results_convergence.jsonl")
    p.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--expect", type=int, default=32, help="expected number of sweep rows")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ref = iqp_reference(args.repo)
    print("[aggregate] Table-2 IQP reference (recomputed, matches the published values):")
    for n in sorted(ref):
        for m in sorted(ref[n]):
            print("    N=%-3d %-18s %.6g  (%s)" % (n, m, ref[n][m]["value"], ref[n][m]["arm"]))
    eval_rows = read_jsonl(os.path.join(args.repo, args.eval_results)
                           if not os.path.isabs(args.eval_results) else args.eval_results)
    rows = build_rows(eval_rows, args.repo)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(args.repo, args.out)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in OUT_KEYS}) + "\n")
    print("[aggregate] wrote %d rows to %s" % (len(rows), out_path))
    n_rbm = sum(1 for r in rows if r["arm_base"] == "RBM-10bit")
    n_gan = sum(1 for r in rows if r["arm_base"] == "DCGAN-32")
    print("[aggregate] RBM rows=%d  DCGAN rows=%d" % (n_rbm, n_gan))
    if len(rows) != args.expect:
        print("[aggregate] WARNING: expected %d rows, got %d" % (args.expect, len(rows)))
    print()
    print(best_table(rows, ref))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
