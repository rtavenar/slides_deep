# Build Dumoulin & Visin convolution-arithmetic figures (numerical mode) and
# convert each chosen frame from PDF to SVG for the slides.
#
# Source: vdumoulin/conv_arithmetic (MIT). See LICENSE in this directory.
#   Dumoulin & Visin, "A guide to convolution arithmetic for deep learning",
#   arXiv:1603.07285.
#
# Pipeline per case:  produce_figure (TikZ) -> pdflatex -> pdf/  -> pdftocairo -> ../../fig/*.svg
#
# Usage:   make            # build all SVGs into ../../fig/
#          make clean      # remove generated pdf/ frames

FIGDIR := ../../fig
PF     := python3 produce_figure

# Each target: name STEP -i input -o output -p pad -k kernel -s stride
# Chosen STEP picks a representative (roughly centred / illustrative) frame.

SVGS := $(FIGDIR)/conv_no_pad.svg \
        $(FIGDIR)/conv_same.svg \
        $(FIGDIR)/conv_valid.svg \
        $(FIGDIR)/conv_stride1.svg \
        $(FIGDIR)/conv_stride2.svg

all: $(SVGS)

# --- helper: $(1)=jobname $(2)=step and conv args -> pdf, then svg ------------
# No padding, unit stride: 5x5, k=3 -> 3x3  (centre frame, step 4)
$(FIGDIR)/conv_no_pad.svg:
	$(PF) numerical no_pad 4 -i 5 -o 3 -p 0 -k 3 -s 1 -m convolution
	pdftocairo -svg pdf/no_pad_04.pdf $@

# "valid" padding (same geometry as no_pad, labelled differently): 5x5 -> 3x3
$(FIGDIR)/conv_valid.svg:
	$(PF) numerical valid 4 -i 5 -o 3 -p 0 -k 3 -s 1 -m convolution
	pdftocairo -svg pdf/valid_04.pdf $@

# "same"/half padding: 5x5, k=3, p=1 -> 5x5  (centre frame, step 12)
$(FIGDIR)/conv_same.svg:
	$(PF) numerical same 12 -i 5 -o 5 -p 1 -k 3 -s 1 -m convolution
	pdftocairo -svg pdf/same_12.pdf $@

# Stride 1: 5x5, k=3 -> 3x3  (step 3, kernel near top to show one step)
$(FIGDIR)/conv_stride1.svg:
	$(PF) numerical stride1 3 -i 5 -o 3 -p 0 -k 3 -s 1 -m convolution
	pdftocairo -svg pdf/stride1_03.pdf $@

# Stride 2: 5x5, k=3, s=2 -> 2x2  (step 1)
$(FIGDIR)/conv_stride2.svg:
	$(PF) numerical stride2 1 -i 5 -o 2 -p 0 -k 3 -s 2 -m convolution
	pdftocairo -svg pdf/stride2_01.pdf $@

clean:
	rm -f pdf/*.pdf pdf/*.aux pdf/*.log

.PHONY: all clean
