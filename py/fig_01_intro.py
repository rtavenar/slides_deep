"""Figures for deck 01 (Introduction).

- housing_scatter.png : RM vs PRICE scatter + fitted line
- gd_steps.png        : scatter+fit and the loss curve with gradient-descent steps
"""
import numpy as np
import matplotlib.pyplot as plt
import style

style.apply()
rng = np.random.default_rng(0)

# --- synthetic "Boston housing"-like data: rooms (RM) vs price -----------------
RM = rng.uniform(4.0, 8.5, 120)
true_w = 9.0
PRICE = true_w * RM + rng.normal(0, 8, RM.size)
PRICE = np.clip(PRICE, 5, None)


def loss(w):
    return np.mean((w * RM - PRICE) ** 2)


# --- 1. scatter + fitted line -------------------------------------------------
w_hat = np.sum(RM * PRICE) / np.sum(RM * RM)  # no-intercept OLS
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(RM, PRICE, s=22, color=style.PURPLE, alpha=0.7, edgecolor="none")
xs = np.linspace(RM.min(), RM.max(), 50)
ax.plot(xs, w_hat * xs, color=style.ORANGE, label=f"fit: price = {w_hat:.1f}·RM")
ax.set_xlabel("RM (avg rooms / dwelling)")
ax.set_ylabel("PRICE")
ax.legend(loc="upper left")
style.save(fig, "housing_scatter.png")

# --- 2. scatter+fit progression  AND  loss curve with GD steps ----------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 4))

# gradient descent on w
w = 2.0
rho = 0.004
hist = [w]
for _ in range(8):
    grad = np.mean(2 * RM * (w * RM - PRICE))
    w = w - rho * grad
    hist.append(w)
hist = np.array(hist)

# left: data + a few candidate lines along the descent
axL.scatter(RM, PRICE, s=18, color=style.PURPLE, alpha=0.55, edgecolor="none")
cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(hist)))
for wi, c in zip(hist, cmap):
    axL.plot(xs, wi * xs, color=c, lw=1.3, alpha=0.9)
axL.plot(xs, hist[-1] * xs, color=style.ORANGE, lw=2.4, label="final fit")
axL.set_xlabel("RM")
axL.set_ylabel("PRICE")
axL.set_title("Candidate lines along descent")
axL.legend(loc="upper left")

# right: loss curve with descent steps (line segments and markers colored
# like the candidate lines on the left, so a step can be matched across
# both panels)
ww = np.linspace(0, 14, 200)
ll = [loss(x) for x in ww]
hist_loss = [loss(x) for x in hist]
axR.plot(ww, ll, color=style.GREY_DARK, lw=2)
for i in range(len(hist) - 1):
    axR.plot(hist[i:i + 2], hist_loss[i:i + 2], "-", color=cmap[i], lw=1.3,
              zorder=2)
axR.scatter(hist, hist_loss, c=cmap, s=36, zorder=3, edgecolor="none")
axR.set_xlabel("w (slope)")
axR.set_ylabel(r"loss $\mathcal{L}(w)$")
axR.set_title("Gradient descent on the loss")
legend_proxy = plt.Line2D([0], [0], color=style.ORANGE, marker="o",
                           markersize=6, lw=1.3, label="GD steps")
axR.legend(handles=[legend_proxy])
fig.tight_layout()
style.save(fig, "gd_steps.png")
