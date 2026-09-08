# Compiling the paper locally

Everything needed to build `main.pdf` is in this folder.

```
main.tex                LaTeX source (NeurIPS 2026 template, dblblindworkshop option)
refs.bib                bibliography (all cited entries verified 2026-09-08)
neurips_2026.sty        style file, byte-identical to the official template
neurips_2026_template/  the unmodified template for reference
figures/                every figure as PDF (vector) and PNG, plus the Python scripts that draw them
main.bbl                pre-built bibliography, so the paper compiles even without running bibtex
compile.sh              one-line build (latexmk)
NUMBERS_PROVENANCE.md   every number in the text mapped to its source file
facts.md                the fact sheet the text was written from
```

## Build

With TeX Live 2023 or newer (pdflatex, bibtex, latexmk):

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

or without latexmk:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`compile.sh` does the same and prints the page count. The first-pass footer text
("Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute.")
is set in `main.tex` by redefining `\@noticestring`; do not edit `neurips_2026.sty`.

## Packages used

`inputenc`, `fontenc`, `hyperref`, `url`, `booktabs`, `amsfonts`, `amsmath`, `microtype`,
`graphicx`, `wrapfig`. All are in a standard TeX Live install.

## Page budget (ML4PS 2026)

Body (title through Discussion) must end on page 4; references start on page 5; appendices are
allowed but discouraged. After any edit run

```bash
python3 -c "import fitz; d=fitz.open('main.pdf'); print(len(d), [i+1 for i,p in enumerate(d) if 'References' in p.get_text()])"
```

(`pip install pymupdf`) to confirm the total page count and the page on which References starts.

## Regenerating figures

```bash
cd figures
python3 make_figures.py            # all figures, about 20 s on CPU
python3 make_figures.py fig1_compact
```

The scripts read the experiment outputs listed in `FIGURE_PROVENANCE.md`; in this release the
result files live under `../results/` and the cluster paths inside the scripts appear as `<HOME>`
and `<WORK>` placeholders (see the top-level README).

## Writing rules used in this draft

No em-dashes, no "label: explanation" sentences, British spelling, methods of the reference paper
named in parentheses on first mention, one home per idea, KL only in the Setup clause, the Fig. 2
caption and the appendix.
