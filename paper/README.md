# ML4PS 2026 submission — materials directory

Target venue: NeurIPS 2026 workshop on Machine Learning and the Physical Sciences
(https://ml4physicalsciences.github.io/2026/). Topic: IQP generative circuits applied to DeepLense
gravitational-lens images, replicating and stress-testing the methodology of arXiv:2603.06440.

This directory holds **prepared materials only** — no `.tex` yet. It was assembled by a materials-prep
pass (2026-09-08) that copied figures, extracted a citable bibliography, and built a number sheet so the
person/agent writing the actual paper body does not need to re-read the experiment docs from scratch.

## Contents

- `facts.md` — number sheet for the writer. Every number has a `file:line` or jsonl-field provenance.
  Read this before writing any results section; do not re-derive numbers from memory.
- `refs.bib` — 38 BibTeX entries, `bibtex`-validated (0 errors, 0 warnings against `plain.bst`). Key
  convention `author_year_short`. Several entries carry `note = {TODO: ...}` for incomplete author lists
  (some 2026 IQP preprints were only recorded as "first author et al." in the source survey) — resolve
  those via WebFetch/arXiv before camera-ready.
- `figures/` — 10 PNGs copied verbatim from `docs/figures/` (see table below). None exceed 3000 px in
  width, so no separately scaled versions were made.

## Figures

| File | Pixel size | Notes |
|---|---|---|
| `iqp_lens_map.png` | 2904×1056 | Correlation–complexity map (our recompute of paper Fig. 5) |
| `iqp_lens_round3.png` | 1624×2100 | Generated-image grid, round 3 + 3b arms |
| `iqp_lens_round3_scores.png` | 2184×1204 | Round 3/3b score bar charts |
| `iqp_lens_scaling.png` | 2131×579 | Data-scaling curves (DCGAN/RBM/IQP vs N) |
| `iqp_lens_pseudotime.png` | 2033×1038 | Pseudo-time (e1-conditioned) generation |
| `iqp_lens_mech_affine_latent.png` | 1320×1848 | Mechanism test: learned affine latent directions |
| `iqp_fig4_qcli_cci.png` | 1760×704 | Reproduction of paper Fig. 4 (QCLI-maximization → emergent CCI) |
| `iqp_fig3_support_mismatch.png` | 2624×640 | Reproduction of paper Fig. 3 — **v1, see caveat below** |
| `iqp_turbulence_replication.png` | 1800×1752 | Turbulence (D2) replication summary figure |
| `lens_characteristics_walsh.png` | 1581×973 | Walsh-spectrum characterization of lens images |

**Caveat on `iqp_fig3_support_mismatch.png`**: per `docs/IQP_LENS_PAPER_SUITE_RESULTS.md:54`, this is
explicitly the **v1** version, and the doc states v1's target-diversity construction failed (all 140
targets' sample QCLI capped at 0.063, below the i.i.d.-uniform null — the paper's claimed regime is
empty in this figure). A v2 rerun was in progress as of the source doc's writing. **Re-copy this file
from `docs/figures/` once v2 lands** — do not use the current copy in the submitted paper without
checking whether it has been superseded.

## Bib entries needing follow-up

Run `grep -n "TODO" refs.bib` to list them. As of this pass, 10 of 38 entries carry a TODO note (mostly
incomplete co-author lists on 2026 IQP-trainability preprints, plus the DeepLense dataset attribution and
two page-number placeholders for PRA/PRX Quantum publications still in press as of the source survey).

## What was not done in this pass

- No `.tex` file was written — that's the next step, out of scope for this materials-prep pass.
- Figure `iqp_fig3_support_mismatch.png` v2 was not available yet; only v1 was copied (see caveat above).
- Author-list TODOs in `refs.bib` were not resolved via live WebFetch (out of scope; flagged for the
  writer/fact-checker instead, consistent with `phd-skills:factcheck` being the recommended follow-up tool).
- No existing files under `docs/` or `experiments/` were modified.

---

## Build (added 2026-09-08 by the drafting pass)

- `main.tex` — the submission. `./compile.sh` runs
  `latexmk -g -pdf -interaction=nonstopmode main.tex` with `<HOME>/.local/bin` on PATH.
  TinyTeX packages installed for this build: `environ`, `lineno`.
- Style: `\usepackage[dblblindworkshop]{neurips_2026}` (double-blind, keeps line numbers and the
  anonymous author block). `natbib` is switched to numeric mode via
  `\PassOptionsToPackage{numbers,compress}{natbib}` because `plain.bst` is a numeric style — without
  it a clean build fails on the first pass. The first-page footer is set by redefining
  `\@noticestring` (the hook the style file itself provides); the `.sty` is unmodified.
- Figures are included **without a file extension** through the `\figboxw` wrapper, so a
  `figures/<name>.pdf` automatically supersedes a `figures/<name>.png` of the same basename, and a
  missing file renders a visible "figure pending" box instead of failing the build. Main text uses
  `fig1_map_samples` and `fig2_mechanism_scaling`; the appendix uses `fig_pipeline`, `figA_fig3`,
  `figA_fig4`.
- Page budget: body (title → end of Sec. 4) must stay at **4 pages**; references start at the top of
  page 5 and the appendix runs pages 6–8. After any edit, check with
  `python3 -c "import fitz; d=fitz.open('main.pdf'); print(d[4].get_text().startswith('References'))"`.
- `NUMBERS_PROVENANCE.md` lists the source of every number printed in the PDF, plus the items still
  needing Leo's confirmation.
- `refs.bib` was edited on 2026-09-08 (backup `refs.bib.backup-20260908`): all `note = {...}` fields
  removed so internal TODO annotations do not print in the bibliography, and the `...` placeholder
  author tokens removed so BibTeX does not typeset them.
