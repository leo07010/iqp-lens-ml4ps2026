# Exact IQP simulation for the 2603.06440 replication (n = 18)

Written 2026-09-04 before implementation. The paper (arXiv 2603.06440, Appendix G) trains
IQP generators with the Ref [38] (IQPopt, arXiv 2503.02934) estimator for <Z_s>. At n = 18 the
full state has only 2^18 = 262,144 amplitudes, so every quantity below is computed EXACTLY with
fast Walsh–Hadamard transforms (FWHT). No Monte-Carlo estimator, no sampling noise, ms per step.
IQPopt is used only as an optional cross-check of <Z_s> (needs a Python >= 3.10 venv).

## Notation
- Bitstrings x in {0,1}^n indexed as integers 0 .. 2^n - 1 (bit j of the integer = qubit j).
- Subsets s of [n] are also bitmasks. |s| = popcount(s). chi_s(x) = (-1)^{popcount(s & x)}.
- W = unnormalized Walsh–Hadamard matrix, W[y, s] = chi_s(y). W is symmetric, W W = 2^n I.
  FWHT(v) = W v costs n * 2^n. Implement in torch as n butterfly passes over a tensor of
  shape (2,)*n (or reshape (2^{j}, 2, 2^{n-1-j}) per pass). It is linear, autograd-safe.

## Circuit -> probabilities (paper eq. 25)
U(theta) = H^{(x)n} exp(i sum_s theta_s Z_s) H^{(x)n},   q_theta(x) = |<x| U |0^n>|^2.

1. Embed the trainable vector theta (only s with |s| <= K, K = 6 -> 31,180 params, K = 7 -> 63,004;
   canonical order = increasing Hamming weight, then lexicographic, matching the paper's
   "{}, {0}, {1}, ..., {n-1}, {0,1}, {0,2}, ..." so that theta_lat = first d_lat = 50 entries)
   into Theta in R^{2^n}, zeros elsewhere.
2. Phase function:  phi = W Theta            (phi(y) = sum_s theta_s chi_s(y); Z_s |y> = chi_s(y) |y>)
3. Amplitudes:      psi = 2^{-n} W exp(i phi)   (H^{(x)n}|0> = 2^{-n/2} sum_y |y>, then D, then H^{(x)n})
4. Probabilities:   q = |psi|^2                (must sum to 1 — assert)
5. All expectation values at once:  <Z_s>_theta = (W q)(s)  for every s in one FWHT.

Sanity: theta = 0 -> U = H H = I -> q = delta_{x, 0...0} (all-zeros string), <Z_s> = 1 for every s.
(Corrected 2026-09-04: an earlier draft wrongly said 'uniform'.)  theta_empty only adds a global phase.

## Data side
Empirical pmf p_hat over 18-bit tuples (x, y, z), 6 bits each (paper Appendix F, eq. 54–56:
k(v) = floor(2^N (v - a)/(b - a)) + 1, binary-encoded, concatenated [b(x) || b(y) || b(z)]).
<Z_s>_data = (W p_hat)(s). One FWHT per snapshot, cached.

## MMD loss in closed form (paper eq. 58–59; coefficients from Ref [38] App. A)
Gaussian kernel on bitstrings. If k(b, b') = exp(-d_H(b, b') / (2 sigma^2)) with Hamming distance d_H
(if Ref [38] uses +/-1 encoding, ||b - b'||^2 = 4 d_H — rescale sigma accordingly; CONFIRM against
ref38_method_details.md before coding), then per bit
   f(delta) = alpha + beta * (-1)^{b_j} (-1)^{b'_j},  r = exp(-1/(2 sigma^2)),  alpha = (1 + r)/2,  beta = (1 - r)/2
so  k(b, b') = sum_s c_s chi_s(b) chi_s(b'),   c_s = alpha^{n - |s|} beta^{|s|}   (depends only on |s|).
Hence  MMD^2(theta) = sum_s c_s ( <Z_s>_theta - <Z_s>_data )^2   exactly over all 2^n subsets.
Multi-scale kernel (if Ref [38] uses several sigma): c_s = sum_sigma c_s(sigma).
Per training step: FWHT(Theta) -> exp -> FWHT -> |.|^2 -> FWHT -> weighted squared error. Backprop through it.

## Training protocol (paper Sec. III.1 / Alg. 1 / line ~1356 of paper.txt)
- Stage 1 (t = 1): theta_lat^(1) random, fixed; train theta_core with Adam for 3e4 steps on MMD.
- Stage 2 (t >= 2): freeze theta_core^(1); per snapshot train only theta_lat^(t) (50 params), 3000 steps, warm-start
  from the previous snapshot's theta_lat.
- Unseen t: piecewise-linear interpolation of theta_lat between neighbouring trained snapshots.
- lr, init, sigma, seeds: take from replication_spec.md / ref38_method_details.md; where the paper is silent, record the
  choice in the pre-registration BEFORE running.

## Snapshot reconstruction & sampling
Exact q_theta is available, so a snapshot can be reconstructed either by torch.multinomial sampling of tuples
(mimicking hardware sampling; use the paper's number of draws) or from the exact conditional E[z | x, y].
Use the SAMPLED route for the headline numbers to stay faithful to the paper; report the exact route as a bonus.

## Diagnostics (paper Sec. II)
- QCLI: P(s) = 2^{-2n} (W p)(s)^2 ; m_k = sum_{|s|=k} P(s) / sum_s P(s) ; b_k = C(n,k)/2^n ; I_QCLI = JS(m || b).
- CCI: I_TC = sum_i H(X_i) - H(X) from the exact pmf; Chow–Liu tree from the 18x18 pairwise MI matrix;
  I_CCI = 1 - I_TC^tree / I_TC.

## Verification plan (must pass before any SLURM run)
1. n = 6..10: compare q and <Z_s> against a dense statevector built from explicit 2^n x 2^n unitaries (numpy).
2. n = 18: sum(q) == 1 to 1e-6 (complex64) ; theta = 0 gives q = delta at the all-zeros string.
3. MMD closed form vs brute-force MMD on n = 8 with explicit kernel matrix and exact q, p.
4. Gradient check (torch.autograd.gradcheck, float64, n = 8).
5. Optional: <Z_s> vs IQPopt estimator on n = 18, |s| <= 2, same theta (if py>=3.10 venv exists).
