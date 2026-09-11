# Module interfaces (fixed 2026-09-04 before implementation; all agents code against this)

Directory: <REPO_ROOT>/code/
Python: /usr/bin/python3 = 3.9.21 (NO `X | None` syntax, no match statements), torch 2.8.0+cu128, numpy 1.26.
Login node: only unit tests that finish in seconds on CPU. All training/evaluation goes through SLURM
(account <YOUR_ACCOUNT>, partitions dev = 4 h / 8gpus = 2 d, `module load cuda/13.0`).

## Time index convention (all arms, fixed 2026-09-05 -- audit fix, see "Data-side single-source rules" below)
- t is 1-BASED throughout: t in {1, ..., N_snapshots} (N_snapshots = 1000 for both synthetic and D2). Array index
  of snapshot t is `t - 1`. This is what every `generated_t{tau}.npy` / `gt_t{tau}.npy` file name's `{tau}` is.
  (Earlier drafts of train_iqp.py used 0-based array indices as the tau label directly, which silently disagreed
  with controls.py/baseline_rbm.py/baseline_dcgan.py's 1-based convention -- fixed; do not reintroduce 0-based tau.)

## Snapshot representation (dataset-independent contract)
- A snapshot is a float32 numpy array `field` of shape (64, 64): the Z-channel (velocity magnitude) on a 2D grid.
  Row index = y (0..63), column index = x (0..63). data_d2.py is responsible for producing this from the raw D2 files.
- Tuples: `(kx, ky, kz)` integer bins in 0..63 each (6 bits). kx, ky are grid indices; kz = floor(64 * (z - a_z)/(b_z - a_z)),
  clipped to [0, 63]; a_z, b_z are the empirical min/max of z over the 11 TRAINING ANCHORS ONLY (not the whole dataset),
  computed ONCE and shared by every arm -- synthetic: min/max of the anchor fields (`synth_data.anchor_a_z_b_z`); D2:
  min/max of the NATIVE z_val (velocity magnitude) over the 11 anchors' native grids (`data_d2.get_z_range`), NOT the
  min/max of `.field` (`.field` is already a 64x64-bin aggregate, whose averaging narrows the range -- ~18.8% observed
  -- versus the native values used to encode; re-deriving a_z/b_z from `.field` was a 2026-09-05 bug, fixed in
  train_iqp.py/controls.py/baseline_rbm.py/baseline_dcgan.py/make_ground_truth.py) -- and recorded in every arm's meta.json.
- Bitstring code (int64): `code = (kx << 12) | (ky << 6) | kz`  -> 18 bits; qubit j == bit j of code (LSB = qubit 0).
  So kz occupies qubits 0-5, ky qubits 6-11, kx qubits 12-17.
- Empirical pmf of a snapshot: `p_hat = np.bincount(codes, minlength=2**18) / len(codes)` (float64), from N_t codes.
- **Code pool + sampler (audit fix, replaces per-draw re-quantization)**: every snapshot exposes a fixed "code pool" --
  synthetic: `codec.field_to_codes_full(field, a_z, b_z)` (every one of the 4096 grid cells, exactly once); D2:
  data_d2.py's native-grid pool (~1e5 points from the raw 308x328 slice, quantized point-by-point -- NOT re-derived
  from the 64x64 binned field, so it keeps a genuine spread of z per 64x64 bin). `codec.sample_codes(pool, n, rng)`
  draws n codes WITH replacement from that pool; this is what IQP, RBM, and CTRL-marginal all call for their N_t
  training tuples (N_t = 5e4 per the paper). `codec.sample_tuples(field, n, rng, a_z, b_z)` (uniform grid cell, then
  quantize) still exists but is no longer used by any arm's default path -- it is statistically equivalent to
  `sample_codes` on synthetic data (1 code per cell) but NOT on D2 (loses intra-bin spread), so do not reintroduce it
  for D2 codepaths.
- Decoding: `decode(codes) -> (kx, ky, kz)`; z value = a_z + (kz + 0.5) * (b_z - a_z) / 64 (bin centre).
- Reconstruction `aggregate(codes) -> field (64, 64)`: mean of decoded z over samples in each (ky, kx) cell;
  empty cells filled with the mean of their non-empty 8-neighbours (iterate until none empty; fallback global mean).
- **"Ground truth" / ​"ok to compare against" field convention**: NEVER the raw continuous field -- always
  `aggregate(pool, a_z, b_z)` (`synth_data.pool_and_quantized_field` for synthetic), i.e. z rounded to the 6-bit bin
  every generative arm is capped at. Used by: make_ground_truth.py's `gt_t{tau}.npy` files, CTRL-interp's
  interpolation inputs, and DCGAN's training fields (paper: "trained directly on quantized 2D turbulence fields").

## Parameter ordering (paper: {∅},{0},{1},…,{n−1},{0,1},{0,2},…)
- `subsets_upto(n, K)` returns the list of qubit-index tuples in itertools.combinations order for weight 0,1,...,K
  (i.e. sorted by Hamming weight, then lexicographic on the sorted tuple). Mask of a tuple = sum(1 << j).
- K = 6 -> 31,180 params ("IQP 31k"); K = 7 -> 63,004 ("IQP 63k"). theta_lat = first d_lat = 50 entries; theta_core = rest.

## Files and owners
- iqp_exact.py, test_iqp_exact.py, train_iqp.py, metrics.py, synth_data.py  -> agent CORE
- baseline_rbm.py, baseline_dcgan.py, test_baselines.py                    -> agent BASELINES
- diagnostics_qcli_cci.py, test_diagnostics.py                             -> agent DIAG
- controls.py, test_controls.py                                           -> agent CONTROLS
- evaluate.py, make_ground_truth.py, run_smoke_all.sh,
  submit_synthetic_full.sbatch, submit_d2_main.sbatch                      -> integration (2026-09-05)
- data_d2.py, docs/IQP_TURBULENCE_REPLICATION_PREREGISTRATION.md           -> written after dataset arrives (still
  blocked on the Dataverse download as of this update -- see the pre-registration Addendum)
- synth_data.py (CORE) provides `make_synthetic_snapshots(n_snapshots, seed) -> np.ndarray (n, 64, 64)` — smooth random
  fields (e.g. sum of a few Gaussian blobs drifting with t) so every module can be smoke-tested before real data exists.
  Also provides the single split/data-access sources every arm must use for `--data synthetic` (no arm may re-derive
  these independently -- see "Data-side single-source rules" below): `SYNTH_DATA_SEED` (fixed seed for data, decoupled
  from each arm's own `--seed`), `N_SNAPSHOTS_SYNTH` (1000), `make_split_labels(n_snapshots, seed, n_holdout,
  anchors_max) -> (anchors, unseen)` (1-based t arrays, the ONE split source), `anchor_a_z_b_z(all_snaps, anchors) ->
  (a_z, b_z)`, `pool_and_quantized_field(raw_field, a_z, b_z) -> (pool_codes, quantized_field)`.

## Data-side single-source rules (audit fix, fixed 2026-09-05; binding on every arm)
1. **Split**: for `--data synthetic`, anchors and the 200 held-out tau come ONLY from
   `synth_data.make_split_labels(N_SNAPSHOTS_SYNTH, seed=SYNTH_DATA_SEED, n_holdout=..., anchors_max=...)`. No file may
   draw its own independent held-out set with its own RNG offset (earlier drafts did: `seed+12345` duplicated inside
   train_iqp.py, `seed+999` in baseline_rbm.py, `seed+10_000` in controls.py's `make_holdout_taus` -- all three gave
   DIFFERENT held-out sets and even disagreed on 0-based vs 1-based tau; `make_holdout_taus` now exists only as a d2-
   without-its-own-split fallback, never exercised for `--data synthetic`). For `--data d2`, the split is `load_d2()`'s
   own pre-fixed `(anchors, holdout_taus)` -- also the ONE source, not re-derived by any arm.
2. **Data seed vs training seed**: the synthetic field VALUES always come from `SYNTH_DATA_SEED` (fixed), never from an
   arm's own `--seed` (which seeds only that run's model-training stochasticity: Adam/PCD/GAN init, sampling). This
   mirrors D2, where the dataset obviously does not change with a model's training seed. Every seed of every arm must
   therefore train/evaluate against the identical ground truth.
3. **a_z/b_z**: computed ONCE over the 11 training anchors -- synthetic: `synth_data.anchor_a_z_b_z` (raw anchor field
   values); D2: `data_d2.get_z_range(plane)` (native z_val, i.e. velocity magnitude on the raw grid -- NOT `snapshots[t].field`,
   which is already a 64x64-bin aggregate and narrows the range) -- and reused by every arm for every
   quantization/dequantization it does, INCLUDING DCGAN when it trains on a different `--n-train` regime (100-snapshot /
   scaling study) than the 11 canonical anchors. Record a_z, b_z in every meta.json.
4. **Code pool**: see the Snapshot representation section above (`codec.sample_codes`, `pool_and_quantized_field`).

## data_d2.py contract (landed 2026-09-05; binding)
```
load_d2() -> (snapshots, anchors, holdout_taus)
    snapshots:     dict[int, Snapshot], keyed by 1-BASED t, covering at least every anchor and held-out t
                   (every t any arm's --anchors-max/--n-holdout/--n-train could ask for)
    anchors:       list[int], the 11 anchor t (1-based, subset of {1, ..., 1000})
    holdout_taus:  list[int], the 200 held-out t (1-based) -- fixed once, the ONE split source for d2,
                   same role as synth_data.make_split_labels for synthetic (rule 1 above)
    Snapshot:      an object (namedtuple / SimpleNamespace / dataclass -- caller only reads attributes) with
        .field        (64, 64) float32 -- the binned Z-channel field (dataset-independent contract above)
        .pool_codes   int64 (~1e5,)    -- EVERY native-grid point (the raw ~308x328 slice) quantized to an
                                          18-bit code against the shared a_z/b_z (see "Code pool" above);
                                          NOT re-derived from `.field` (that would collapse to codec's
                                          `field_to_codes_full`, losing the intra-bin z spread D2 actually has)
load_d2_snapshot(t) -> Snapshot   # convenience single-snapshot accessor; must return the SAME object/values
                                  # as snapshots[t] from load_d2() (baseline_rbm.py / baseline_dcgan.py use this;
                                  # controls.py / train_iqp.py use load_d2()'s `snapshots` dict directly)
```
`get_z_range(plane="mid", cache_dir=None) -> (a_z, b_z)`: the ONE a_z/b_z source for D2 (rule 3 above) -- native z_val
min/max over the 11 anchors, cached to `<cache_dir>/z_range_<plane>.json`. Every d2-consuming file (train_iqp.py,
controls.py, baseline_rbm.py, baseline_dcgan.py, make_ground_truth.py) MUST call this directly and must NOT
re-derive a_z/b_z from `snapshots[t].field`'s own min/max (that was a real 2026-09-05 bug -- `.field` is already
codec.aggregate'd onto the 64x64 grid, which narrows the range vs. the bounds load_d2() used to encode `.pool_codes`,
so decoding/aggregating/quantizing against a `.field`-derived a_z/b_z silently used the wrong scale; fixed in all
five files above). `get_holdout_taus() -> list[int]` is likewise the ONE d2 holdout-tau source (rule 1 above); no
arm may re-derive its own d2 holdout set with a different RNG.

## Generated-sample contract (what evaluate.py reads)
Each time-conditioned model arm (IQP, RBM, CTRL-marginal, CTRL-interp) writes `outputs/<arm>/generated_t{tau}.npy`
(64, 64) float32 for each held-out tau (1-based, matching "Time index convention" above), plus `outputs/<arm>/meta.json`
with at least `arm` (a Table-2 row label, e.g. "IQP-31k", "RBM-12k", "DCGAN-51k", "CTRL-marginal"), `param_count`,
`seed`, `train_snapshots` (either the list of training t's or a bare count -- evaluate.py normalizes both), `a_z`,
`b_z`, `wallclock_s`. DCGAN (no time-conditioning mechanism in the paper) writes unconditional samples named
`generated_{i:03d}.npy` instead. Ground truth held-out fields are `outputs/ground_truth/gt_t{tau}.npy`, written by
make_ground_truth.py for `--data synthetic` (from the same synth_data split/a_z/b_z every arm uses) and by an
analogous d2 step once data_d2.py exists (from `load_d2()`'s own split). `evaluate.py` globs `outputs/*/` (or
`--runs`), matches each arm's generated files against ground truth by tau (or, for DCGAN, against the whole held-out
set distributionally), computes `metrics.pdf_js` / `metrics.conv_mmd2` (fixed encoder seed 0, same effective weights
for every arm), writes `results.jsonl` (one row per arm-seed run), and prints the Table 2 markdown (mean +/- std
across seeds sharing an (arm, train_snapshots) group).

## data_lens.py contract (landed 2026-09-06; binding for `--data lens`)
The DeepLense Model II pilot (docs/IQP_LENS_PILOT_PREREGISTRATION.md) reuses this whole
pipeline on 64x64 dark-matter lensing images. `data_lens.py` deliberately mirrors
`data_d2.py`'s contract attribute-for-attribute so each arm's `--data lens` branch is a
near-copy of its `--data d2` branch.
```
load_lens(cls, n_train, seed=0, n_holdout=200, sqrt=False) -> (snapshots, anchors, holdout_ids)
    cls          : class name, one of get_class_names() = ['axion', 'cdm', 'no_sub']
                   (read from model_II.npz's own class_names, index i == label i)
    snapshots    : dict[int, Snapshot] over anchors U holdout_ids
    anchors      : [1, ..., n_train]      -- the TRAINING images, in index order. Algorithm 1
                   trains theta_core on anchor 1 and warm-starts theta_lat through 2..n_train
                   in exactly this order (the lens analogue of the 11 anchor times).
    holdout_ids  : [1001, ..., 1000 + n_holdout] -- the HELD-OUT val images.
                   HOLDOUT_ID_OFFSET = 1000 keeps the two key ranges disjoint, so ONE dict
                   carries both (chosen over returning a separate holdout_snapshots dict:
                   it keeps load_d2()'s exact 3-tuple return shape, so no arm needs a new
                   unpacking path).
    Snapshot     : .t (the key), .field (64,64) float32, .pool_codes int64 (4096,)
                   -- the SAME attribute names as data_d2.Snapshot.
load_lens_snapshot(i, cls, n_train=100, sqrt=False) -> Snapshot   # same values as snapshots[i]
get_z_range(plane=None) -> (0.0, 1.0)     # the ONE a_z/b_z source; `plane` accepted, ignored
get_holdout_ids(n_holdout=200) -> list[int]   # the ONE held-out source (rule 1)
get_class_names() -> ['axion', 'cdm', 'no_sub']
raw_training_images(cls, n_train, sqrt=False) -> (n_train, 64, 64) QUANTIZED training fields
holdout_images(cls, n_holdout=200, sqrt=False) -> (n_holdout, 64, 64) QUANTIZED held-out fields
transform_intensity(x, sqrt) / invert_intensity(x, sqrt)   # the secondary arm's transform
```
Encoding (pre-registration "Data", applied literally): one image = one snapshot; the tuple
is (pixel column, pixel row, intensity), so `kx = column`, `ky = row`,
`kz = floor(64 * intensity)` clipped to [0, 63], i.e. **(a_z, b_z) = (0, 1) FIXED for every
class** -- there is no anchor-derived range here, and no arm may re-derive one (the d2
a_z/b_z bug in a different costume). An image's code pool is its 4,096 pixels exactly once
(`codec.field_to_codes_full`), so `.field == codec.aggregate(.pool_codes, 0, 1)` exactly
(test_lens.py checks this) and `.field` is the 6-bit-quantized image -- the same
"ground truth" convention every other dataset here uses.

Data-side single-source rules (the lens instances of the four rules above):
1. **Split**: `load_lens()`'s own `anchors` / `holdout_ids` (equivalently
   `get_holdout_ids()`), never an arm's own RNG draw.
2. **Data seed vs training seed**: the training/held-out image draw is fixed at
   `data_lens.SPLIT_SEED = 0`, decoupled from any arm's `--seed`. The N = 11 training set
   is a PREFIX of the N = 100 one (one permutation, two prefixes), which is what lets
   evaluate_lens.py score both regimes against one loaded training array.
3. **a_z/b_z**: `get_z_range()` -> (0, 1), recorded in every meta.json.
4. **Code pool**: `Snapshot.pool_codes` (4,096 pixels); arms draw N_t = 5e4 training
   tuples from it with `codec.sample_codes`, as on d2 (deliberately not `--full-grid`,
   so the data-side sampling noise matches d2's and is identical across arms).

Cache: `build_cache()` materializes ~15 MB of per-class .npy under `LENS_CACHE_DIR` (default
`<WORK>/data/lens_cache`) on first use, because the 1.4 GB source .npz must be
inflated whole-array-at-a-time. Same role as preprocess_d2.py's d2_cache.

### Generated-sample contract for `--data lens`
The lens data has NO time axis, so Eq. 31 latent interpolation has nothing to interpolate
along. Every generative lens arm therefore runs with `--synth-mode gaussian-latent`
(lens_synth.py): after the unchanged Algorithm 1 training, fit N(mu, diag sigma^2) to the
N trained latents (IQP: theta_lat; RBM: [b_v(18), b_h[:32]]; CTRL-marginal: the 18-dim
product-Bernoulli p-vector, clipped back into [0, 1]), draw `--n-gen` latents, generate one
image each. Consequences for the output contract:
- files are `generated_{i:03d}.npy` (evaluate.py's *unconditional* naming), never
  `generated_t{tau}.npy`; there is no `outputs/ground_truth/` step -- evaluate_lens.py
  takes the 200 held-out images from `data_lens.holdout_images()` directly;
- each such arm ALSO writes `<out>_last/` with the same samples under
  `agg_variants.aggregate_last` (LAST-WRITE-WINS), plus its own meta.json tagged
  `aggregation: "last"` and `arm: "<arm>-last"`, so the mean-vs-last aggregation choice is
  visible in the table instead of being a hidden lever;
- meta.json additionally carries `class_names`, `lens_class`, `lens_sqrt`, `n_train`,
  `synth_mode`, `aggregation`, and the fitted `latent_mu` / `latent_sigma`;
- the secondary `--lens-sqrt` arm quantizes sqrt(intensity) but writes its generated fields
  back through `data_lens.invert_intensity` (squaring) into INTENSITY space, so it is scored
  against the same held-out images as every other arm;
- `controls.py --arm interp` is refused for `--data lens` (no time axis); the lens no-model
  controls are `controls_lens.py --arm copy` (memorisation) and `--arm mean` (class blur).

### Round-2 lens arms (landed 2026-09-06; docs/IQP_LENS_PILOT_PREREGISTRATION.md Addendum)
Round 1 found IQP to be the worst trained model. Round 2 adds three levers, all opt-in, so
the round-1 code paths and every d2/synthetic arm are byte-for-byte unchanged when the new
flags are absent.

```
train_iqp.py --encoding {tuple,pmf}     # --data lens only; default tuple = round 1
             --stage1-train-all         # stage 1 trains theta_lat too (default: frozen)
             --arm-name LABEL           # meta.json's `arm` (several arms share a param count)
baseline_rbm.py --arm-name LABEL        # same
evaluate_lens.py --outputs-dir DIR [DIR ...]   # merges several trees into ONE table
```
- **`--encoding pmf` (codec12.py, 12 qubits)**: `code = (kx << 6) | ky`, flat index
  `kx*64 + ky`; the target distribution is the image itself, `p = field / field.sum()`, and
  the scale `S` lives OUTSIDE the circuit. `data_lens.load_lens(..., encoding="pmf")` adds
  `Snapshot.pmf` (4096, float64, sums to 1) and `.total_intensity`; `.field` -- the ground
  truth -- is byte-identical to the tuple encoding's, so both encodings are scored against
  the same held-out images. `Snapshot.pool_codes` becomes a `pool_size` (default
  `PMF_POOL_SIZE = 5e4`) draw from `.pmf` fixed at `data_lens.POOL_SEED`, NOT at an arm's
  `--seed` (same rule as the split): it is the paper-protocol arm's N_t sample.
  `data_lens.mean_total_intensity(cls, n_train)` is the ONE source of `S` and uses TRAINING
  images only.
- **`--full-grid` in lens mode = the exact data pmf.** For `--encoding tuple` the pool IS
  the image's 4,096 pixels each once, so `codec.codes_to_pmf(pool)` is exact. For
  `--encoding pmf` the flag bypasses codes entirely and takes `Snapshot.pmf`; the trainer
  prints `[data] built N EXACT image pmfs`. Without it a pmf arm trains on
  `codec12.codes_to_pmf(pool_codes)`, i.e. the paper's 5e4-sample estimate.
  `--init data` under the exact path uses `codec12.pmf_bit_moments` (exact weighted moments)
  instead of sample moments of a code draw.
- **Output contract for a pmf arm** (`lens_synth.write_pmf_pair` / `write_metas_pmf`):
  PRIMARY `<out>/generated_{i:03d}.npy` = `codec12.pmf_to_image(q_exact, S)` -- no
  Monte-Carlo step at all; CONTROL `<out>_sampled/` = the counting density of `--n-synth`
  circuit samples, tagged `reconstruction: "sampled"` and `arm: "<label>-sampled"`. There is
  no `_last/` sibling (last-write-wins is meaningless for a pmf reconstruction). Tuple arms
  keep the round-1 `_last/` contract.
- **meta.json gains** `encoding`, `stage1_train_all`, `full_grid`, `exact_data_pmf`,
  `stage1_latent_shift` (max |theta_lat - init| over stage 1: exactly 0.0 under the paper
  protocol, > 0 under `--stage1-train-all` -- the flag is auditable from the output alone),
  and for pmf arms `intensity_scale_S` and `n_tuples_effective` (0 when exact).
- **Scoring**: `evaluate_lens.py --outputs-dir outputs_lens_pilot/<cls> outputs_lens_round2/<cls>`
  produces one table holding the round-1 and round-2 arms against the same 200 held-out
  images and the same seed-0 Conv2D encoder; duplicate (arm, N) pairs are reported as a
  WARNING rather than silently averaged. `submit_lens_round2.sbatch` runs the 30 round-2
  jobs and this merged evaluation.
- Tests: `test_lens_round2.py` (codec12 round-trips and its agreement with codec.py's
  (kx, ky); the 12-qubit d_lat = 50 block = {} + 12 singles + 37 pairs; --full-grid really
  being the exact pmf; the --stage1-train-all A/B; all five arms end-to-end plus the merged
  table). `test_lens.py` and `test_d2_arms_smoke.py` still pass unchanged.

#### Range / total-intensity diagnostics (added 2026-09-06 after the round-2 code review)
`evaluate_lens.py` records six extra fields per row -- `gen_min`, `gen_max`,
`gen_mean_total`, `holdout_min`, `holdout_max`, `holdout_mean_total` -- and prints a NOTE
line for any arm that leaves the held-out pixel range or whose mean total intensity is off
by more than +25% / -20%. **No metric changed**; these only make an existing property of
the metrics visible.

Why it matters: `metrics.pdf_js` bins on the JOINT range of (held-out, that arm's output)
and `metrics.conv_mmd2` normalises its encoder input the same way, so an arm outside the
data's range is scored on a stretched scale and is not directly comparable to an in-range
arm. Re-scoring the round-1 outputs with the diagnostic on (cdm, measured 2026-09-06) shows
this is not hypothetical -- held-out mean total intensity is 149.3, and:

| round-1 arm | mean total | vs held-out |
|---|---|---|
| DCGAN-1.2m (N=100) | 899.2 | x6.02 |
| DCGAN-0.2m (N=100) | 786.1 | x5.27 |
| IQP-31k (N=100) | 222.9 | x1.49 |
| IQP-31k-sqrt (N=11) | 52.6 | x0.35 |
| DCGAN-51k (N=100) | 112.7 | x0.75 |
| CTRL-copy / CTRL-mean | 154.1 | x1.03 |

So a substantial part of round 1's PDF-JS / MMD2 / NN spread is a total-intensity scale
mismatch rather than spatial structure, and the two arms that look best on the round-1
table (CTRL-copy, CTRL-mean) are also the two whose totals match. **Round-2 pmf arms emit
exactly S by construction** (S = mean total intensity of the TRAINING images; for cdm at
N=3, S = 146.26 vs held-out 137.82 = x1.06), so they get that scale right for free. When
writing up P7, do NOT attribute a pmf arm's NN-holdout advantage to better spatial
modelling without first subtracting this effect -- and note that the converse also holds:
the pmf arms cannot express per-image total-intensity variation at all (held-out totals
span 116-162), so they carry a floor no spatial fit can remove.
