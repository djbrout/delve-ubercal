PAPER = delve_ubercal

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib figures/*.pdf
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf

.PHONY: all clean
