"""Shared matplotlib style for the Deep Learning slides figures.

Deck accent: purple rgb(131, 109, 169).
All figures are saved as transparent PNGs into ../fig/.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- palette (matches the Typst deck) ---------------------------------------
PURPLE = "#836DA9"          # primary accent  rgb(131,109,169)
PURPLE_DARK = "#5E4F86"
PURPLE_LIGHT = "#B7A8D4"
GREY = "#C8C8C8"            # secondary rgb(200,200,200)
GREY_DARK = "#6E6E6E"
ORANGE = "#D98A3D"          # contrast / "fake"/"data" accent
TEAL = "#3D928C"            # second contrast (cyan-ish activation maps)
INK = "#2B2438"            # near-black text

CYCLE = [PURPLE, ORANGE, TEAL, GREY_DARK, PURPLE_LIGHT]

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "fig")

# ---- language switch ---------------------------------------------------------
# Set FIG_LANG=fr in the environment to render figure text in French and save
# outputs under a "_fr" suffix, so the English figures are never overwritten.
LANG = os.environ.get("FIG_LANG", "en")


def t(en, fr):
    """Pick the label matching the current FIG_LANG."""
    return fr if LANG == "fr" else en


def term_kw():
    """kwargs to italicize an untranslated English loanword (e.g. "epoch",
    "mini-batch") when rendering the French figures; a no-op in English."""
    return {"fontstyle": "italic"} if LANG == "fr" else {}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.edgecolor": GREY_DARK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": GREY_DARK,
        "ytick.color": GREY_DARK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "lines.linewidth": 2.2,
        "legend.frameon": False,
    })


def save(fig, name, **kw):
    """Save as vector SVG. `name` may be given with any extension; it is
    replaced with .svg. Under FIG_LANG=fr, a "_fr" suffix is inserted so the
    English figures are never overwritten."""
    os.makedirs(FIG_DIR, exist_ok=True)
    suffix = "_fr" if LANG == "fr" else ""
    name = os.path.splitext(name)[0] + suffix + ".svg"
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote", os.path.relpath(path))
