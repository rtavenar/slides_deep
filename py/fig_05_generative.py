"""Figures for deck 05 (generative models).

- gmm.png        : mixture p(x) = sum of two Gaussians
- gmm_2d.png     : 2D, 2-mode GMM — level lines (model) vs. crosses (samples)
- flow_field.png : flow-matching velocity field + a transported trajectory
- diffusion_strip.png : forward noising of a simple shape, x0 -> xT
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal
import style

style.apply()
rng = np.random.default_rng(4)


# --- 1. GMM density -----------------------------------------------------------
def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


x = np.linspace(-6, 8, 500)
c1 = 0.6 * gauss(x, -1.5, 1.0)
c2 = 0.4 * gauss(x, 3.5, 1.3)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(x, c1, color=style.PURPLE_LIGHT, ls="--", label="component 1")
ax.plot(x, c2, color=style.TEAL, ls="--", label="component 2")
ax.plot(x, c1 + c2, color=style.ORANGE, lw=2.6, label="mixture p(x)")
ax.fill_between(x, c1 + c2, color=style.ORANGE, alpha=0.08)
ax.set_xlabel("x"); ax.set_ylabel("density"); ax.legend()
style.save(fig, "gmm.png")

# --- 1b. GMM density in 2D: level lines (the model) vs. samples (the data) ----
# Illustrates the duality: we *observe* samples (crosses) and *seek* a model
# of the density that produced them (the level lines).
mu1, cov1, w1 = np.array([-2.0, 1.0]), np.array([[1.4, 0.55], [0.55, 0.9]]), 0.55
mu2, cov2, w2 = np.array([2.6, -1.4]), np.array([[1.0, -0.35], [-0.35, 1.25]]), 0.45
comp1, comp2 = multivariate_normal(mu1, cov1), multivariate_normal(mu2, cov2)

xg, yg = np.linspace(-6, 6.5, 250), np.linspace(-5, 5, 250)
Xg, Yg = np.meshgrid(xg, yg)
grid_pts = np.dstack([Xg, Yg])
Zg = w1 * comp1.pdf(grid_pts) + w2 * comp2.pdf(grid_pts)

n = 90
n1 = rng.binomial(n, w1)
samples = np.vstack([
    rng.multivariate_normal(mu1, cov1, n1),
    rng.multivariate_normal(mu2, cov2, n - n1),
])

fig, ax = plt.subplots(figsize=(6.4, 5.2))
purple_cmap = mcolors.LinearSegmentedColormap.from_list("purple_soft", ["#FFFFFF", style.PURPLE])
fill_levels = np.linspace(Zg.max() * 0.06, Zg.max(), 9)
line_levels = np.linspace(Zg.max() * 0.06, Zg.max(), 8)
ax.contourf(Xg, Yg, Zg, levels=fill_levels, cmap=purple_cmap, alpha=0.55, extend="neither")
ax.contour(Xg, Yg, Zg, levels=line_levels, colors=style.PURPLE, linewidths=1.1)
ax.scatter(samples[:, 0], samples[:, 1], marker="x", s=42, linewidths=1.6,
           color=style.INK, zorder=5)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
handles = [
    Line2D([0], [0], marker="x", color=style.INK, linestyle="none", markersize=9,
           markeredgewidth=1.8, label="observed samples $x_i$"),
    Line2D([0], [0], color=style.PURPLE, lw=2, label="model density $p(x)$ (level lines)"),
]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5)
style.save(fig, "gmm_2d.png")

# --- 2. flow matching velocity field -----------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4))
# noise (t=0) left blob, data (t=1) right blob
n = 160
x0 = rng.normal([-3, 0], [0.6, 0.9], (n, 2))
x1 = rng.normal([3, 0], [0.6, 0.9], (n, 2))
ax.scatter(*x0.T, s=10, color=style.PURPLE, alpha=0.5, label="noise (t=0)")
ax.scatter(*x1.T, s=10, color=style.ORANGE, alpha=0.5, label="data (t=1)")
# velocity field = constant-ish transport to the right
gx, gy = np.meshgrid(np.linspace(-4, 4, 14), np.linspace(-2.5, 2.5, 9))
u = np.ones_like(gx) * 1.0
v = -0.15 * gy
ax.quiver(gx, gy, u, v, color=style.TEAL, alpha=0.6, width=0.004)
# one trajectory
p0 = np.array([-3, 1.2]); p1 = np.array([3, -0.5])
t = np.linspace(0, 1, 30)[:, None]
traj = (1 - t) * p0 + t * p1
ax.plot(traj[:, 0], traj[:, 1], color=style.INK, lw=2.4, label="trajectory")
ax.scatter(*p0, color=style.PURPLE, edgecolor="k", zorder=5, s=40)
ax.scatter(*p1, color=style.ORANGE, edgecolor="k", zorder=5, s=40)
ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="upper center", ncol=3,
                                                 fontsize=9)
ax.set_title("Flow matching: transport noise → data")
style.save(fig, "flow_field.png")

# --- 3. diffusion forward noising strip --------------------------------------
# A real MNIST digit as x0, so "data" reads as an actual image rather than an
# abstract blob.
mnist = np.load(os.path.expanduser("~/.keras/datasets/mnist.npz"))
x_train, y_train = mnist["x_train"], mnist["y_train"]
base = x_train[np.where(y_train == 8)[0][0]].astype(float) / 255.0
steps = [0.0, 0.2, 0.45, 0.7, 1.0]
fig, axes = plt.subplots(1, len(steps), figsize=(11, 2.6))
for ax, b in zip(axes, steps):
    img = (1 - b) * base + b * rng.normal(0.5, 0.25, base.shape)
    ax.imshow(img, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t = {b:.2f}", fontsize=11)
axes[0].set_ylabel("$x_0$ (data)", fontsize=11)
fig.suptitle("Forward noising:  data  →  pure noise", y=1.02)
fig.tight_layout()
style.save(fig, "diffusion_strip.png")
