# Build MLP architecture diagrams: TikZ -> PDF -> SVG into ../../fig/
#
# Diagrams adapted (theme-recoloured) from R. Tavenard, "Deep Learning Basics"
# (rtavenard/deep_book), chapter "MLP".
#
# Pipeline:  gen_mlp.py -> *.tex  ->(pdflatex)-> pdf/*.pdf ->(pdftocairo)-> ../../fig/*.svg

FIGDIR := ../../fig
NAMES  := mlp_1hidden mlp_2hidden mlp_2hidden_wide mlp_io mlp_deep
SVGS   := $(addprefix $(FIGDIR)/,$(addsuffix .svg,$(NAMES)))

all:
	python3 gen_mlp.py
	@for n in $(NAMES); do \
	  pdflatex -interaction=nonstopmode -halt-on-error -output-directory pdf $$n.tex > /dev/null && \
	  pdftocairo -svg pdf/$$n.pdf $(FIGDIR)/$$n.svg && echo "built $$n.svg"; \
	done

clean:
	rm -f pdf/*.pdf pdf/*.aux pdf/*.log *.tex

.PHONY: all tex clean
