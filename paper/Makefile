# Build the paper PDF via pandoc + LaTeX
# Usage:
#   make         — build paper.pdf
#   make clean   — remove generated PDF
#   make watch   — rebuild on every save (requires entr)

PAPER_NAME := rslora-imbalanced
SRC := paper.md
BIB := refs.bib
OUT := $(PAPER_NAME).pdf

PANDOC_FLAGS := \
	--from markdown+yaml_metadata_block+raw_tex+citations \
	--to pdf \
	--pdf-engine=xelatex \
	--citeproc \
	--bibliography=$(BIB) \
	--csl=https://www.zotero.org/styles/ieee \
	--variable=geometry:margin=1in \
	--variable=fontsize:11pt \
	--variable=linkcolor:blue \
	--variable=urlcolor:blue \
	--variable=papersize:letter \
	--standalone

$(OUT): $(SRC) $(BIB)
	pandoc $(PANDOC_FLAGS) -o $(OUT) $(SRC)

clean:
	rm -f $(OUT)

watch:
	echo "$(SRC) $(BIB)" | tr ' ' '\n' | entr -c make

.PHONY: clean watch
