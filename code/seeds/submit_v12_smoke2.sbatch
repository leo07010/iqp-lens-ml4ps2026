#!/bin/bash
#SBATCH -J v12_smoke2
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-v12smoke2-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-v12smoke2-%j.err
#
# SMOKE 2: the coordinator's arms E (pixel-space kernel), F (8-qubit 16x16 + bilinear),
# G (Gray code), H (small MLP latent map) and I (lowres8 + MLP).  One seed each, N = 11.
#
#   1  py_compile + unit checks (codec round trips, separable == dense kernel, crop10 pmfs
#      identical to data_lens_r3's own)
#   2  REGRESSION: the default path must STILL be bit-identical to the published seed-0 arm
#      after the encoding / latent-map / spatial refactor
#   3  one full run of each new arm, timed, on 4 GPUs
#   4  evaluate_v11 on all of them, printed beside the published Table 2 (v11) numbers
# ---------------------------------------------------------------------------------------
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
SM="$REPO/outputs_v12_smoke2"
mkdir -p logs_lens
rm -rf "$SM"; mkdir -p "$SM"
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"

echo "[$(ts)] 1/4 py_compile + unit checks"
"$PY" -m py_compile train_iqp_affine_v12.py codec_v12.py objective_spatial.py v12_aggregate.py \
    v12_figs.py || { echo "[FATAL] py_compile"; exit 1; }
"$PY" - <<'PYEOF' || { echo "[FATAL] unit checks failed"; exit 1; }
import sys
sys.path.insert(0, "<REPO_ROOT>/code")
import numpy as np, torch
import codec_v12, objective_spatial, data_lens_r3, codec10
d = data_lens_r3.load_r3(11, n_holdout=20)
e = codec_v12.get("crop10")
p = e.pmfs_of_crops(d.train_crops)
assert np.array_equal(p, np.asarray(d.train_pmfs)), "crop10 pmfs differ from data_lens_r3's"
print("[unit] crop10 pmfs == data_lens_r3.train_pmfs : bit-identical")
for nm in ("crop10", "gray10", "lowres8"):
    e = codec_v12.get(nm)
    c = e.roundtrip_corr(d.train_crops)
    sp = objective_spatial.SpatialMMD(e.axis_x, e.axis_y, [1., 2., 4.], "cpu",
                                      torch.float64, lam_flat=0.3)
    dd = torch.randn(4, e.ncodes, dtype=torch.float64)
    err = float((sp.rows(dd) - ((dd @ sp.dense()) * dd).sum(dim=1)).abs().max())
    print("[unit] %-8s ncodes=%4d  encode->decode corr on real crops = %.6f | "
          "separable vs dense kernel max|d| = %.2e" % (nm, e.ncodes, c, err))
    assert err < 1e-9
U = codec_v12.bilinear_matrix(32, 16)
a = np.random.default_rng(0).random((16, 16))
ref = torch.nn.functional.interpolate(torch.tensor(a)[None, None], size=(32, 32),
                                      mode="bilinear", align_corners=False)[0, 0].numpy()
print("[unit] bilinear matrix vs torch.interpolate : max|d| = %.2e" % np.abs(ref - U @ a @ U.T).max())
assert np.abs(ref - U @ a @ U.T).max() < 1e-12
PYEOF

echo "[$(ts)] 2/4 regression: the default path after the refactor"
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_iqp_affine_v12.py \
    --n-train 11 --n-holdout 200 --n-gen 200 --seed 0 --M 4 \
    --encoding crop10 --latent-map affine --objective mmdflat --mu 0 --flux mean \
    --prior resample --steps 4000 --lr 0.02 --theta0-steps 1000 --infer-restarts 8 \
    --infer-steps 1000 --infer-lr 0.005 --infer-chunk 50 --log-every 2000 --device cuda \
    --out "$SM/regress_flat_n11" > logs_lens/v12_smoke2_regress.log 2>&1 \
    || { echo "[FATAL] regression run failed"; tail -40 logs_lens/v12_smoke2_regress.log; exit 1; }
"$PY" - <<'PYEOF' || { echo "[FATAL] regression comparison failed"; exit 1; }
import glob, json, os
import numpy as np
R = "<REPO_ROOT>/code"
ref = os.path.join(R, "outputs_lens_r3b", "iqp_m4_mmdflat_resample_n11")
new = os.path.join(R, "outputs_v12_smoke2", "regress_flat_n11")
n = len(glob.glob(os.path.join(ref, "generated_*.npy")))
w = max(float(np.abs(np.load(os.path.join(ref, "generated_%03d.npy" % i)).astype(np.float64)
                     - np.load(os.path.join(new, "generated_%03d.npy" % i)).astype(np.float64)).max())
        for i in range(n))
mb = json.load(open(os.path.join(new, "meta.json")))
print("[regress] %d images, max |ref - v12| = %.3e | proj corr %.10f | params %d"
      % (n, w, mb["projection_corr_mean"], mb["param_count"]))
print("[regress] VERDICT:", "IDENTICAL" if w == 0.0 else "DIFFERENT -- investigate")
raise SystemExit(0 if w == 0.0 else 1)
PYEOF

echo "[$(ts)] 3/4 one full run of each new arm (4 GPUs)"
COMMON="--n-train 11 --n-holdout 200 --n-gen 200 --seed 0 --prior resample --steps 4000 \
 --lr 0.02 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 \
 --infer-chunk 50 --log-every 2000 --device cuda --sigmas auto3"
ARMS=(
 "E_spatial_M4|--objective spatial --M 4 --flux resample"
 "E_spatialflat_M4|--objective spatial --spatial-flat auto --M 4 --flux resample"
 "F_lowres8_M2_flat|--encoding lowres8 --objective mmdflat --M 2 --flux resample"
 "F_lowres8_M2_spatial|--encoding lowres8 --objective spatial --M 2 --flux resample"
 "G_gray10_M4_flat|--encoding gray10 --objective mmdflat --M 4 --flux resample"
 "H_mlpH8_M4_flat|--latent-map mlp --hidden 8 --objective mmdflat --M 4 --flux resample"
 "H_mlpH4_M4_flat|--latent-map mlp --hidden 4 --objective mmdflat --M 4 --flux resample"
 "I_lowres8_mlpH8_flat|--encoding lowres8 --latent-map mlp --hidden 8 --objective mmdflat --M 4 --flux resample"
 "EH_mlpH8_spatial|--latent-map mlp --hidden 8 --objective spatial --M 4 --flux resample"
)
i=0; pids=()
for entry in "${ARMS[@]}"; do
    tag="${entry%%|*}"; opt="${entry#*|}"
    slot=$(( i % 4 ))
    (
      t0=$(date +%s)
      if CUDA_VISIBLE_DEVICES="$slot" $PY -u train_iqp_affine_v12.py $COMMON $opt \
             --out "$SM/$tag" > "logs_lens/v12_smoke2_$tag.log" 2>&1; then
          echo "[$(ts)] OK   $tag ($(( $(date +%s) - t0 )) s)"
      else
          echo "[$(ts)] FAIL $tag ($(( $(date +%s) - t0 )) s)"; tail -20 "logs_lens/v12_smoke2_$tag.log"
      fi
    ) &
    pids+=($!)
    i=$(( i + 1 ))
    if [ $(( i % 4 )) -eq 0 ]; then wait; fi
done
wait
grep -hE "^\[v12\] (encoding|objective=spatial|spatial \+ flat|g:)" logs_lens/v12_smoke2_*.log | sort -u

echo "[$(ts)] 4/4 evaluate_v11"
mapfile -t RUNS < <(find "$SM" -mindepth 1 -maxdepth 1 -type d -name '*' | sort)
echo "scoring ${#RUNS[@]} run dirs"
"$PY" -u evaluate_v11.py --runs "${RUNS[@]}" --n-holdout 200 --no-legacy --quiet \
    --results "$SM/results_smoke2.jsonl" > logs_lens/v12_smoke2_eval.log 2>&1 \
    || { echo "[FATAL] evaluate_v11 failed"; tail -40 logs_lens/v12_smoke2_eval.log; exit 1; }
"$PY" - <<'PYEOF'
import json, os
p = "<REPO_ROOT>/code/outputs_v12_smoke2/results_smoke2.jsonl"
rows = [json.loads(l) for l in open(p) if l.strip()]
hdr = ("%-46s %9s %9s %9s %8s %7s %7s %8s" %
       ("arm (N=11, seed 0)", "PDF-JS", "MMD2", "MMD2_al", "proj", "gmax", "clip%", "params"))
print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda z: z["conv_mmd2_aligned"]):
    m = json.load(open(os.path.join(r["run_dir"], "meta.json")))
    print("%-46s %9.6f %9.5f %9.5f %8.4f %7.3f %7.3f %8d"
          % (r["arm"][:46], r["pdf_js"], r["conv_mmd2"], r["conv_mmd2_aligned"],
             m.get("projection_corr_mean") or float("nan"), r["gen_max"],
             100 * r["gen_clip_frac"], r["param_count"]))
print("-" * len(hdr))
print("%-46s %9.6f %9.5f %9.5f %8s %7s %7s %8d" % ("DCGAN (Table 2 v11, 5 seeds)", 0.001106, 0.05094, 0.03883, "--", "--", "--", 199837))
print("%-46s %9.6f %9.5f %9.5f %8s %7s %7s %8d" % ("RBM (Table 2 v11, 5 seeds)", 0.001669, 0.06537, 0.11060, "--", "--", "--", 11010))
print("%-46s %9.6f %9.5f %9.5f %8s %7s %7s %8d" % ("IQP M4 flat (Table 2 v11, 5 seeds)", 0.002979, 0.08914, 0.02737, "--", "--", "--", 5164))
PYEOF
du -sh "$SM"
echo "[$(ts)] SMOKE2 DONE"
