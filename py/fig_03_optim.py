"""Figures for deck 03 (loss / optimization / regularization).

- standardization.png : training loss with vs without input standardization
- gd_pitfalls.png     : LR too small / too large / local-minimum panels
- gd_vs_sgd.png       : GD smooth vs SGD noisy trajectory on a loss landscape
- early_stopping.png  : train vs validation error, best-model marker
- batchnorm.png       : training loss with vs without BatchNorm
- vanishing_grad.png  : gradient magnitude per layer (sigmoid vs ReLU)
"""
import numpy as np
import matplotlib.pyplot as plt
import style

style.apply()
rng = np.random.default_rng(2)


# --- 1. standardization -------------------------------------------------------
ep = np.arange(1, 41)
with_std = 0.1 + 0.9 * np.exp(-ep / 6)
without = 0.1 + 0.9 * np.exp(-ep / 22) + 0.04 * np.sin(ep / 2) * np.exp(-ep / 30)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(ep, with_std, color=style.PURPLE, label=style.t("with standardization", "avec standardisation"))
ax.plot(ep, without, color=style.ORANGE, ls="--", label=style.t("without", "sans"))
ax.set_xlabel("epoch", **style.term_kw())
ax.set_ylabel(style.t("training loss", "perte d'entraînement"))
ax.legend()
style.save(fig, "standardization.png")


# --- 2. GD pitfalls (3 panels) ------------------------------------------------
def bowl(x):
    return (x - 1) ** 2


fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
xs = np.linspace(-3, 5, 200)


def plot_steps(ax, pts, ys):
    """Color each descent step with the same start->end plasma gradient
    used in gd_steps.png, so the direction of the optimization is visible."""
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(pts)))
    for i in range(len(pts) - 1):
        ax.plot(pts[i:i + 2], ys[i:i + 2], "-", color=cmap[i], lw=2.2, zorder=2)
    ax.scatter(pts, ys, c=cmap, s=30, zorder=3, edgecolor="none")


# too small
ax = axes[0]; ax.plot(xs, bowl(xs), color=style.GREY_DARK)
x = -2.5; pts = [x]
for _ in range(6):
    x -= 0.08 * 2 * (x - 1); pts.append(x)
pts = np.array(pts)
plot_steps(ax, pts, bowl(pts))
ax.set_title(style.t("LR too small\n(slow)", "Taux d'apprentissage trop petit\n(lent)"))

# too large
ax = axes[1]; ax.plot(xs, bowl(xs), color=style.GREY_DARK)
x = -2.0; pts = [x]
for _ in range(6):
    x -= 0.95 * 2 * (x - 1); pts.append(x)
pts = np.array(pts)
plot_steps(ax, pts, bowl(pts))
ax.set_title(style.t("LR too large\n(oscillates)", "Taux d'apprentissage trop grand\n(oscille)"))

# local minimum
ax = axes[2]
xs2 = np.linspace(-3.5, 4.5, 300)
f = 0.5 * xs2**2 + 3 * np.sin(1.3 * xs2)
ax.plot(xs2, f, color=style.GREY_DARK)
x = 3.8; pts = [x]
for _ in range(20):
    g = xs2 - 1  # placeholder, recompute numeric grad
    gi = (0.5 * (x + 1e-3) ** 2 + 3 * np.sin(1.3 * (x + 1e-3)) -
          (0.5 * x**2 + 3 * np.sin(1.3 * x))) / 1e-3
    x -= 0.08 * gi; pts.append(x)
pts = np.array(pts)
fp = 0.5 * pts**2 + 3 * np.sin(1.3 * pts)
plot_steps(ax, pts, fp)
ax.set_title(style.t("Local minimum\n/ plateau", "Minimum local\n/ plateau"))
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
style.save(fig, "gd_pitfalls.png")


# --- 3. GD vs SGD trajectory on a loss landscape ------------------------------
def loss_xy(X, Y):
    return 0.6 * X**2 + 2.2 * Y**2


gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-2, 2, 200))
Z = loss_xy(gx, gy)
fig, ax = plt.subplots(figsize=(6, 4))
ax.contour(gx, gy, Z, levels=12, colors=style.GREY, linewidths=0.8)

# GD smooth
p = np.array([-2.6, 1.6]); gd = [p.copy()]
for _ in range(18):
    g = np.array([1.2 * p[0], 4.4 * p[1]]); p = p - 0.12 * g; gd.append(p.copy())
gd = np.array(gd)
ax.plot(gd[:, 0], gd[:, 1], "o-", color=style.PURPLE, ms=4, label=style.t("GD (smooth)", "GD (lisse)"))

# SGD noisy
p = np.array([-2.6, 1.6]); sgd = [p.copy()]
for _ in range(40):
    g = np.array([1.2 * p[0], 4.4 * p[1]]) + rng.normal(0, 1.6, 2)
    p = p - 0.06 * g; sgd.append(p.copy())
sgd = np.array(sgd)
ax.plot(sgd[:, 0], sgd[:, 1], "-", color=style.ORANGE, lw=1.3, alpha=0.9,
        label=style.t("SGD (noisy)", "SGD (bruitée)"))
ax.plot(0, 0, "*", color=style.INK, ms=15)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="upper right")
ax.set_title(style.t("Parameter trajectories", "Trajectoires des paramètres"))
style.save(fig, "gd_vs_sgd.png")


# --- 4. early stopping --------------------------------------------------------
ep = np.arange(1, 61)
train = 0.05 + 0.8 * np.exp(-ep / 10)
val = 0.12 + 0.7 * np.exp(-ep / 9) + 0.0025 * (ep - 18) ** 2 * (ep > 18)
best = ep[np.argmin(val)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ep, val, color=style.PURPLE, label=style.t("validation", "validation"))
ax.plot(ep, train, color=style.ORANGE, ls="--", label=style.t("training", "entraînement"))
ax.axvline(best, color=style.GREY_DARK, ls=":", lw=1.5)
ax.scatter([best], [val.min()], color=style.INK, zorder=5)
ax.annotate(style.t("best model", "meilleur modèle"), (best, val.min()), xytext=(best - 13, val.min() + 0.55),
            arrowprops=dict(arrowstyle="->", color=style.INK), fontsize=11)
ax.set_xlabel("epoch", **style.term_kw()); ax.set_ylabel(style.t("error (RMSE)", "erreur (RMSE)"))
ax.legend()
style.save(fig, "early_stopping.png")


# --- 5. batchnorm -------------------------------------------------------------
ep = np.arange(1, 41)
bn = 0.08 + 0.9 * np.exp(-ep / 5)
nobn = 0.08 + 0.9 * np.exp(-ep / 16) + 0.05 * np.sin(ep / 1.5) * np.exp(-ep / 20)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(ep, bn, color=style.PURPLE, label=style.t("with BatchNorm", "avec BatchNorm"))
ax.plot(ep, nobn, color=style.ORANGE, ls="--", label=style.t("without", "sans"))
ax.set_xlabel("epoch", **style.term_kw()); ax.set_ylabel(style.t("training loss", "perte d'entraînement"))
ax.legend()
style.save(fig, "batchnorm.png")


# --- 6. vanishing gradient: |grad| per layer ----------------------------------
layers = np.arange(1, 11)
# sigmoid: derivative <= 0.25 -> product shrinks ~0.25^depth-from-output
sig = 0.25 ** (layers[::-1] - 1) * 0.8
relu = np.full_like(layers, 0.8, dtype=float) * (0.92 ** (layers[::-1] - 1))
fig, ax = plt.subplots(figsize=(6.2, 4))
w = 0.4
ax.bar(layers - w / 2, sig, width=w, color=style.ORANGE, label=style.t("sigmoid", "sigmoïde"))
ax.bar(layers + w / 2, relu, width=w, color=style.PURPLE, label="ReLU")
ax.set_yscale("log")
ax.set_xlabel(style.t("layer (1 = closest to input)", "couche (1 = la plus proche de l'entrée)"))
ax.set_ylabel(style.t("|gradient|  (log scale)", "|gradient|  (échelle log)"))
ax.set_title(style.t("Gradient magnitude across depth", "Magnitude du gradient selon la profondeur"))
ax.set_xticks(layers)
ax.legend()
style.save(fig, "vanishing_grad.png")
