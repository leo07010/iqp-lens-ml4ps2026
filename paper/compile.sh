#!/bin/bash
# Build the ML4PS 2026 submission PDF.
set -e
export PATH=<HOME>/.local/bin:$PATH
cd "$(dirname "$0")"
latexmk -g -pdf -interaction=nonstopmode main.tex
grep -E "Output written" main.log | tail -1
