"""Figures for deck 05 (generative models).

- gmm.png        : mixture p(x) = sum of two Gaussians
- flow_field.png : flow-matching velocity field + a transported trajectory
- diffusion_strip.png : forward noising of a simple shape, x0 -> xT
"""
import numpy as np
import matplotlib.pyplot as plt
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
size = 32
yy, xx = np.mgrid[0:size, 0:size]
base = np.exp(-(((xx - 16) ** 2 + (yy - 16) ** 2) / 40))  # a blob "image"
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
