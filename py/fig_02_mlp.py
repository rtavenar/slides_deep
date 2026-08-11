"""Figures for deck 02 (MLP).

- activations.png       : sigmoid, tanh, ReLU (+ derivatives faint)
- activations_relu.png  : same trio, ReLU highlighted ("reign of ReLU")
- softmax.png           : softmax turning logits into a probability bar chart
- spiral.png            : 2-class spiral dataset (non-linearly separable)
"""
import numpy as np
import matplotlib.pyplot as plt
import style

style.apply()


def _plot_activations(highlight=None):
    x = np.linspace(-6, 6, 400)
    fns = {
        "sigmoid": 1 / (1 + np.exp(-x)),
        "tanh": np.tanh(x),
        "ReLU": np.maximum(0, x),
    }
    display_name = {"sigmoid": style.t("sigmoid", "sigmoïde"), "tanh": "tanh", "ReLU": "ReLU"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (name, y) in zip(axes, fns.items()):
        is_hi = (highlight == name)
        color = style.ORANGE if is_hi else style.PURPLE
        ax.axhline(0, color=style.GREY, lw=1)
        ax.axvline(0, color=style.GREY, lw=1)
        ax.plot(x, y, color=color, lw=3 if is_hi else 2.2)
        ax.set_title(display_name[name], color=color, fontweight="bold" if is_hi else "normal")
        ax.set_ylim(-1.4, 3)
        ax.set_xticks([-5, 0, 5])
    fig.tight_layout()
    return fig


style.save(_plot_activations(), "activations.png")
style.save(_plot_activations(highlight="ReLU"), "activations_relu.png")

# --- softmax -----------------------------------------------------------------
logits = np.array([2.0, 1.0, 0.1, -1.2])
probs = np.exp(logits) / np.exp(logits).sum()
labels = [style.t(f"class {i}", f"classe {i}") for i in range(len(logits))]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
a1.bar(labels, logits, color=style.GREY_DARK)
a1.set_title(style.t("logits $o_i$ (any real)", "logits $o_i$ (tout réel)"))
a1.axhline(0, color=style.GREY, lw=1)
a2.bar(labels, probs, color=style.PURPLE)
a2.set_title(style.t("softmax: >0, sums to 1", "softmax : >0, somme à 1"))
a2.set_ylim(0, 1)
for i, p in enumerate(probs):
    a2.text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=10)
fig.tight_layout()
style.save(fig, "softmax.png")

# --- spiral dataset ----------------------------------------------------------
rng = np.random.default_rng(1)
n = 200
theta = np.sqrt(rng.uniform(0, 1, n)) * 3.5 * np.pi
r = theta / (3.5 * np.pi)
x0 = np.c_[r * np.cos(theta), r * np.sin(theta)] + rng.normal(0, 0.03, (n, 2))
x1 = np.c_[r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)] + rng.normal(0, 0.03, (n, 2))
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.scatter(*x0.T, s=14, color=style.PURPLE, label="class 0", edgecolor="none")
ax.scatter(*x1.T, s=14, color=style.ORANGE, label="class 1", edgecolor="none")
ax.set_title("Spiral: not linearly separable")
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="upper right")
style.save(fig, "spiral.png")
