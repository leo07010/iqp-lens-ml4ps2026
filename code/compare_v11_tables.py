#!/usr/bin/env python3
"""v10-vs-v11 comparison tables with rank changes, from results_v11_seed0.jsonl.

Prints, for Table 2's six arms (both N), the per-class runs, the data-scaling arms and the
convergence sweep: each metric's v10 (legacy) value, its v11 value, and the row's rank under
each protocol (a `**` marks a rank that moved), plus every run whose generated pixels exceed
the data range v11 clips to.  Read-only; it does not recompute anything.

    cd experiments/13_iqp_turbulence_replication && python3 compare_v11_tables.py
"""
import json, os, sys

REPO = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
rows = [json.loads(l) for l in open(os.path.join(REPO, "results_v11_seed0.jsonl"))
        if l.strip()]
runs = [r for r in rows if r.get("run_dir")]

TABLE2 = [("iqp_paper", "IQP paper-protocol (Alg.1)"),
          ("iqp_resample", "IQP fixed basis, joint"),
          ("iqp_m4_mmdflat_resample", "IQP learned M=4 (flat MMD)"),
          ("iqp_m4_auto3_resample", "IQP learned M=4 (paper kernel)"),
          ("rbm", "RBM-10bit"),
          ("dcgan", "DCGAN-32")]

def base(r):
    b = os.path.basename(r["run_dir"])
    for suf in ("_n100", "_n11", "_n10", "_n30", "_n50", "_n70", "_n90"):
        if b.endswith(suf):
            return b[:-len(suf)]
    return b

def rank(rs, key):
    ok = [r for r in rs if r.get(key) is not None]
    order = sorted(ok, key=lambda r: r[key])
    return {id(r): i + 1 for i, r in enumerate(order)}

def block(title, groups, cols):
    print("\n### %s" % title)
    hdr = ["arm", "N"]
    for _newk, _oldk, lab in cols:
        hdr += ["%s v10" % lab, "%s v11" % lab, "rank v10→v11"]
    hdr += ["gen_max", "clip frac"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for gname, rs in groups:
        ranks = {}
        for newk, oldk, lab in cols:
            ranks[(lab, "new")] = rank(rs, newk)
            ranks[(lab, "old")] = rank(rs, oldk)
        for r in rs:
            cells = [r["_label"], str(r["n_train"])]
            for newk, oldk, lab in cols:
                ro = ranks[(lab, "old")].get(id(r), "-")
                rn = ranks[(lab, "new")].get(id(r), "-")
                mark = "" if ro == rn else "  **"
                cells += ["%.5g" % r[oldk], "%.5g" % r[newk],
                          "%s→%s%s" % (ro, rn, mark)]
            cells += ["%.3f" % r["gen_max"], "%.2g" % r["gen_clip_frac"]]
            print("| " + " | ".join(cells) + " |")
    return

COLS = [("pdf_js", "pdf_js_legacy", "PDF–JS"),
        ("conv_mmd2", "conv_mmd2_legacy", "MMD²"),
        ("conv_mmd2_aligned", "conv_mmd2_aligned_legacy", "MMD²al")]

# ---- Table 2 ----
for N in (11, 100):
    rs = []
    for key, lab in TABLE2:
        m = [r for r in runs if base(r) == key and r["n_train"] == N and r["class_name"] is None
             and r["tree"] in ("outputs_lens_r3", "outputs_lens_r3b")]
        if not m:
            print("MISSING", key, N)
            continue
        m[0]["_label"] = lab
        rs.append(m[0])
    block("Table 2 arms, N=%d (ranked within the 6 arms)" % N, [("t2", rs)], COLS)

# ---- Table 3 (per class) ----
for N in (11, 100):
    rs = [r for r in runs if r["class_name"] and r["n_train"] == N]
    for r in rs:
        r["_label"] = r["class_name"]
    if rs:
        block("Table 3 per-class, N=%d (ranked among the 3 classes)" % N, [("t3", rs)], COLS)

# ---- scaling ----
for N in (10, 30, 50, 70, 90, 100):
    rs = [r for r in runs if r["tree"] == "outputs_lens_scaling" and r["n_train"] == N]
    for r in rs:
        r["_label"] = base(r)
    if rs:
        block("Scaling N=%d (ranked among the 3 arms)" % N, [("sc", rs)], COLS)

# ---- convergence: winner step per arm ----
print("\n### Convergence sweep: best step by each metric (v10 vs v11)")
conv = [r for r in runs if r["tree"] == "outputs_convergence"]
gr = {}
for r in conv:
    gr.setdefault((r.get("arm_base") or r["arm"], r["n_train"]), []).append(r)
print("| arm | N | metric | best step v10 | best step v11 | v10 value | v11 value |")
print("|---|---|---|---|---|---|---|")
for (arm, N), rs in sorted(gr.items()):
    for newk, oldk, lab in COLS:
        bo = min(rs, key=lambda r: r[oldk])
        bn = min(rs, key=lambda r: r[newk])
        print("| %s | %d | %s | %s | %s | %.5g | %.5g |"
              % (arm, N, lab, bo["selected_steps"], bn["selected_steps"],
                 bo[oldk], bn[newk]))

# ---- clip summary ----
print("\n### Clip fractions (v11 range [0,1]) — every run with gen_max > 1")
print("| run | N | gen_max | clip frac | above | below |")
print("|---|---|---|---|---|---|")
for r in sorted(runs, key=lambda z: -z["gen_max"]):
    if r["gen_max"] > 1.0 + 1e-6:
        print("| %s | %s | %.3f | %.3g | %.3g | %.3g |"
              % (os.path.basename(r["run_dir"]), r["n_train"], r["gen_max"],
                 r["gen_clip_frac"], r["gen_clip_frac_above"], r["gen_clip_frac_below"]))
n_clip = sum(1 for r in runs if r["gen_max"] > 1.0 + 1e-6)
print("\n%d of %d runs have gen_max > 1" % (n_clip, len(runs)))
