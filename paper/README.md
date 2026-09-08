# paper/ (placeholder)

This directory is intentionally empty in this snapshot of the release folder —
the paper source was still being edited at the time this folder was built and
is copied in separately. When populated, it should contain:

```
paper/
  main.tex                    # NeurIPS ML4PS 2026 submission source
  refs.bib                    # bibliography
  neurips_2026.sty            # venue style file
  compile.sh                  # build script (latexmk / pdflatex + biber-or-bibtex)
  figures/                    # figure-generation scripts + rendered PDFs/PNGs
    make_figures.py           # driver: regenerates every figure from results/
    fig_style.py
    fig_pipeline.py
    fig1_map_samples.py
    fig2_mechanism_scaling.py
    figA_fig3.py
    figA_fig4.py
    FIGURE_PROVENANCE.md      # file:line / json-key provenance for every panel
  main.pdf                    # compiled submission
  NUMBERS_PROVENANCE.md       # file:line provenance for every number in main.tex
  facts.md                    # the underlying number sheet (own file:line provenance)
```

To finish this release folder, copy the finished `paper/ml4ps2026/` tree here
(dropping any `*.backup-*`, `.aux`/`.log`/`.fls`/`.fdb_latexmk` build
byproducts, `main.bbl`/`.blg`, and the `.claude/` directory), then regenerate
the figures against this repository's `results/` directory:

```
cd paper/figures
python3 make_figures.py            # all figures, ~15 s CPU-only
python3 make_figures.py fig1_map_samples   # or one stem at a time
```

`NUMBERS_PROVENANCE.md` and `figures/FIGURE_PROVENANCE.md` trace every number
and every pixel in the submission back to a specific file in `results/` or
`code/` — read those two files first if you want to check a specific claim.
