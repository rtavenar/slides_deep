"""Generate standalone LaTeX/TikZ for MLP architecture diagrams.

Adapted (recoloured to the slides theme) from the MLP network diagrams in
R. Tavenard, "Deep Learning Basics" (rtavenard/deep_book), chapter "MLP".

Each diagram is a fully-connected feed-forward graph: one column of nodes per
layer, all-to-all edges between consecutive layers, optional column titles and
weight-matrix labels. Emits one .tex per variant into this directory.
"""
import os

HERE = os.path.dirname(__file__)

# --- slides theme palette (RGB 0-255) ----------------------------------------
COLORS = r"""
\definecolor{mlpinput}{RGB}{131,109,169}   % primary purple   (input layer)
\definecolor{mlphidA}{RGB}{167,149,196}    % light purple     (hidden 1)
\definecolor{mlphidB}{RGB}{61,146,140}     % teal accent      (hidden 2)
\definecolor{mlpout}{RGB}{51,159,52}       % green            (output)
\definecolor{mlpedge}{RGB}{170,170,170}    % light grey edges
\definecolor{mlpstroke}{RGB}{90,90,90}
"""

def preamble(x_scale="1.6cm", y_scale="1.0cm"):
    return r"""\documentclass[border=8pt]{standalone}
\usepackage{tikz}
\usepackage{xcolor}
\usetikzlibrary{positioning}
""" + COLORS + r"""
\begin{document}
\begin{tikzpicture}[
    x=""" + x_scale + r""", y=""" + y_scale + r""",
    node distance=1cm,
    neuron/.style={draw=mlpstroke, circle, minimum size=14pt, inner sep=0pt, line width=0.5pt},
    every node/.style={font=\small},
]
"""


PREAMBLE = preamble()

POSTAMBLE = r"""
\end{tikzpicture}
\end{document}
"""


def diagram(layers, labels=None, titles=None, x_scale="1.6cm", y_scale="1.0cm"):
    """layers: list of (count, color). labels: weight labels between layers.
    titles: per-layer title strings (LaTeX) or None to omit.
    x_scale/y_scale: TikZ x=/y= unit lengths, tweak to control aspect ratio."""
    lines = []
    n_layers = len(layers)
    # node coordinates centred vertically around y=0
    coords = []  # coords[l] = list of (name, x, y)
    for li, (n, _) in enumerate(layers):
        x = li
        y0 = (n - 1) / 2.0
        col = []
        for i in range(n):
            name = f"n{li}_{i}"
            y = y0 - i
            col.append((name, x, y))
        coords.append(col)

    # nodes
    for li, (n, color) in enumerate(layers):
        for (name, x, y) in coords[li]:
            lines.append(
                f"  \\node[neuron, fill={color}] ({name}) at ({x},{y}) {{}};")

    # edges
    for li in range(n_layers - 1):
        for (a, _, _) in coords[li]:
            for (b, _, _) in coords[li + 1]:
                lines.append(f"  \\draw[->, mlpedge, line width=0.4pt] ({a}) -- ({b});")

    # titles above each column
    if titles:
        top = max(n for n, _ in layers) / 2.0 + 0.7
        for li, t in enumerate(titles):
            if t:
                lines.append(
                    f"  \\node[align=center, font=\\footnotesize] at ({li},{top:.2f}) {{{t}}};")

    # weight-matrix labels on the centre line, between columns
    if labels:
        for li, lbl in enumerate(labels):
            if lbl:
                x = li + 0.5
                lines.append(
                    f"  \\node[fill=white, inner sep=1pt] at ({x},0) {{{lbl}}};")

    return preamble(x_scale, y_scale) + "\n".join(lines) + POSTAMBLE


def write(name, tex):
    path = os.path.join(HERE, name + ".tex")
    with open(path, "w") as f:
        f.write(tex)
    print("wrote", name + ".tex")


IN, HA, HB, OUT = "mlpinput", "mlphidA", "mlphidB", "mlpout"

# 1. one hidden layer: x(5) -> h1(7) -> y(1)
write("mlp_1hidden", diagram(
    [(5, IN), (7, HA), (1, OUT)],
    labels=[r"$\mathbf{w^{(0)}}$", r"$\mathbf{w^{(1)}}$"],
    titles=[r"Input\\$\mathbf{x}$", r"Hidden 1\\$\mathbf{h^{(1)}}$",
            r"Output\\$\mathbf{\hat{y}}$"],
))

# 2. two hidden layers, titled + weight labels (the main architecture figure)
write("mlp_2hidden", diagram(
    [(5, IN), (7, HA), (7, HB), (1, OUT)],
    labels=[r"$\mathbf{w^{(0)}}$", r"$\mathbf{w^{(1)}}$", r"$\mathbf{w^{(2)}}$"],
    titles=[r"Input\\$\mathbf{x}$", r"Hidden 1\\$\mathbf{h^{(1)}}$",
            r"Hidden 2\\$\mathbf{h^{(2)}}$", r"Output\\$\mathbf{\hat{y}}$"],
))

# 2b. same architecture, wide/flat aspect ratio for full-width slide display
write("mlp_2hidden_wide", diagram(
    [(5, IN), (7, HA), (7, HB), (1, OUT)],
    labels=[r"$\mathbf{w^{(0)}}$", r"$\mathbf{w^{(1)}}$", r"$\mathbf{w^{(2)}}$"],
    titles=[r"Input\\$\mathbf{x}$", r"Hidden 1\\$\mathbf{h^{(1)}}$",
            r"Hidden 2\\$\mathbf{h^{(2)}}$", r"Output\\$\mathbf{\hat{y}}$"],
    x_scale="3.4cm", y_scale="0.6cm",
))

# 3. labeled input/output sizing: x(4 features) -> h1(7) -> h2(7) -> y(3 targets)
write("mlp_io", diagram(
    [(4, IN), (7, HA), (7, HB), (3, OUT)],
    titles=[r"Input\\(\# features)", r"$\mathbf{h^{(1)}}$",
            r"$\mathbf{h^{(2)}}$", r"Output\\(\# targets)"],
))

# 4. deep net (4 hidden layers) for the vanishing-gradient slide
write("mlp_deep", diagram(
    [(4, IN), (6, HA), (6, HB), (6, HA), (6, HB), (1, OUT)],
))

# --- French-labelled variants (only the ones actually used in the FR decks) --
write("mlp_2hidden_fr", diagram(
    [(5, IN), (7, HA), (7, HB), (1, OUT)],
    labels=[r"$\mathbf{w^{(0)}}$", r"$\mathbf{w^{(1)}}$", r"$\mathbf{w^{(2)}}$"],
    titles=[r"Entrée\\$\mathbf{x}$", r"Caché 1\\$\mathbf{h^{(1)}}$",
            r"Caché 2\\$\mathbf{h^{(2)}}$", r"Sortie\\$\mathbf{\hat{y}}$"],
))

write("mlp_2hidden_wide_fr", diagram(
    [(5, IN), (7, HA), (7, HB), (1, OUT)],
    labels=[r"$\mathbf{w^{(0)}}$", r"$\mathbf{w^{(1)}}$", r"$\mathbf{w^{(2)}}$"],
    titles=[r"Entrée\\$\mathbf{x}$", r"Caché 1\\$\mathbf{h^{(1)}}$",
            r"Caché 2\\$\mathbf{h^{(2)}}$", r"Sortie\\$\mathbf{\hat{y}}$"],
    x_scale="3.4cm", y_scale="0.6cm",
))
