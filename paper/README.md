# Paper: When rsLoRA Hurts

**Source**: `paper.md` (Markdown with YAML metadata + Pandoc citations)
**Bibliography**: `refs.bib` (BibTeX)
**Build**: `make` produces `rslora-imbalanced.pdf`

## Build requirements

- pandoc ≥ 3.1
- xelatex (TeX Live core or full)
- citeproc (bundled with pandoc)
- IEEE CSL style (auto-fetched from Zotero on first build)

All of the above are present on the dev machine. To build elsewhere:

```bash
sudo apt install -y texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra pandoc
make
```

## Files

- `paper.md` — paper source
- `refs.bib` — bibliography
- `Makefile` — build commands
- `data/` — eval results extracted from `../results/` for the paper
- `figures/` — generated figures (matplotlib output)
- `src/` — scripts to regenerate figures and tables from raw eval data

## Reproducibility

Every score in the paper is derivable from `../results/<arm>/<session>.graded.grader_*.json`. Scripts in `src/` consolidate those into the paper's tables and figures.
