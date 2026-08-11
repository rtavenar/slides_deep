"""Figures for deck 04 (CNNs).

- lsvrc.svg : ImageNet top-5 error by year (shallow vs deep)
- rgb_pixels.svg : an image as a grid of pixels, each pixel as 3 RGB values
- aug_original.svg / aug_grid.svg : data augmentation, original vs. 6 variants
- conv_multichannel.svg : multi-channel conv — per-kernel sum over input
  channels, then feature maps stacked into the output tensor
- mnist_grid.svg : one real MNIST sample per digit class, black on white
- batchnorm_cube.svg : (N, C, H×W) tensor cube, one channel slice highlighted
  to show batch norm's normalization axes

Convolution-arithmetic figures (no-pad, padding, strides) live in
fig_04_conv_arithmetic.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables projection="3d")
from scipy import ndimage
import style

style.apply()

# --- LSVRC error by year ------------------------------------------------------
years = ["2010", "2011", "2012\n(AlexNet)", "2013", "2014", "2015\n(ResNet)"]
err = [28.2, 25.8, 16.4, 11.7, 7.3, 3.6]
deep = [False, False, True, True, True, True]
colors = [style.ORANGE if not d else style.PURPLE for d in deep]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(years, err, color=colors)
ax.axhline(5.1, color=style.GREY_DARK, ls=":", lw=1.5)
ax.text(5.4, 5.6, style.t("human ~5%", "humain ~5%"), ha="right", color=style.GREY_DARK, fontsize=10)
for b, e in zip(bars, err):
    ax.text(b.get_x() + b.get_width() / 2, e + 0.4, f"{e:.1f}", ha="center",
            fontsize=10)
ax.set_ylabel(style.t("top-5 error (%)", "erreur top-5 (%)"))
ax.set_title(style.t("ImageNet (LSVRC): shallow (orange) → deep (purple)",
                      "ImageNet (LSVRC) : peu profond (orange) → profond (violet)"))
style.save(fig, "lsvrc.png")

# --- image = grid of pixels, pixel = 3 RGB values -----------------------------
H, W = 12, 12
heart_rows = [
    "............",
    "..XX....XX..",
    ".XXXX..XXXX.",
    "XXXXXXXXXXXX",
    "XXXXXXXXXXXX",
    "XXXXXXXXXXXX",
    ".XXXXXXXXXX.",
    "..XXXXXXXX..",
    "...XXXXXX...",
    "....XXXX....",
    ".....XX.....",
    "............",
]
heart_mask = np.array([[c == "X" for c in row] for row in heart_rows])
leaf_mask = np.zeros((H, W), dtype=bool)
leaf_mask[0:3, 0:3] = True

bg_color = np.array([90, 150, 225])
heart_color = np.array([215, 45, 55])
leaf_color = np.array([60, 165, 90])

img = np.tile(bg_color, (H, W, 1)).astype(np.uint8)
img[heart_mask] = heart_color
img[leaf_mask] = leaf_color

hi_r, hi_c = 4, 5  # a plain heart pixel to highlight
hi_val = img[hi_r, hi_c]

fig = plt.figure(figsize=(9.5, 4.6))
fig.text(0.03, 0.95, style.t("An image = a grid of pixels", "Une image = une grille de pixels"),
          fontsize=14, ha="left", color=style.INK)
fig.text(0.98, 0.95, style.t("A pixel = 3 luminance values (R, G, B)", "Un pixel = 3 valeurs de luminance (R, V, B)"),
          fontsize=14, ha="right", color=style.INK)

# panel A: pixel grid, one pixel highlighted with its (R, G, B) triplet
axL = fig.add_axes([0.03, 0.04, 0.42, 0.82])
axL.imshow(img, extent=(0, W, 0, H), origin="upper", interpolation="nearest")
for x in range(W + 1):
    axL.axvline(x, color="white", lw=0.8, alpha=0.8)
for y in range(H + 1):
    axL.axhline(y, color="white", lw=0.8, alpha=0.8)

rx, ry = hi_c, H - hi_r - 1
axL.add_patch(plt.Rectangle((rx, ry), 1, 1, fill=False, edgecolor=style.INK, lw=2.4))
axL.annotate(
    style.t(f"one pixel =\n(R, G, B) = ({hi_val[0]}, {hi_val[1]}, {hi_val[2]})",
            f"un pixel =\n(R, V, B) = ({hi_val[0]}, {hi_val[1]}, {hi_val[2]})"),
    xy=(rx + 1, ry + 1), xycoords="data",
    xytext=(W + 0.6, H - 1.5), textcoords="data",
    fontsize=12, color=style.INK,
    arrowprops=dict(arrowstyle="-", color=style.INK, lw=1.2, shrinkA=0, shrinkB=4),
)
axL.set_xlim(-0.3, W + 5.5)
axL.set_ylim(-0.3, H + 0.3)
axL.set_xticks([])
axL.set_yticks([])
for s in axL.spines.values():
    s.set_visible(False)

# panel B: the image exploded into its 3 color-tinted R/G/B planes
axR = fig.add_axes([0.44, -0.06, 0.54, 0.92], projection="3d")


def plane_rgba(channel_idx):
    plane = np.zeros((H, W, 4), dtype=float)
    plane[:, :, channel_idx] = img[:, :, channel_idx].astype(float) / 255.0
    plane[:, :, 3] = 1.0
    return plane


X, Y = np.meshgrid(np.arange(W + 1), np.arange(H + 1))
offsets = [0, 1.4, 2.8]  # R front, G middle, B back
channel_labels = style.t("RGB", "RVB")
for i, label in enumerate(channel_labels):
    Z = np.full_like(X, -offsets[i], dtype=float)
    axR.plot_surface(
        X, Y, Z, rstride=1, cstride=1,
        facecolors=np.flipud(plane_rgba(i)), shade=False,
        linewidth=0.15, edgecolor=(1, 1, 1, 0.35), antialiased=True,
    )
    axR.text(W + 0.4, H + 0.3, -offsets[i], label, fontsize=13, color=style.INK, weight="bold")

axR.set_box_aspect((W, H, 7))
axR.view_init(elev=18, azim=-60)
axR.set_axis_off()
axR.text2D(0.78, 0.10, style.t("height × width × 3 channels", "hauteur × largeur × 3 canaux"),
           transform=axR.transAxes, ha="center", fontsize=11, color=style.GREY_DARK)

style.save(fig, "rgb_pixels.svg")

# --- data augmentation: original cat vs. 6 augmented variants -----------------
CAT_FUR = "#E3A857"
CAT_PATCH = "#8C5A2B"
CAT_PINK = "#E79EA6"


def draw_cat(ax):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    # ears
    ax.add_patch(plt.Polygon([(-0.95, 0.55), (-0.55, 1.45), (-0.15, 0.75)],
                              closed=True, facecolor=CAT_FUR, edgecolor=style.INK, lw=1.5))
    ax.add_patch(plt.Polygon([(0.95, 0.55), (0.55, 1.45), (0.15, 0.75)],
                              closed=True, facecolor=CAT_FUR, edgecolor=style.INK, lw=1.5))
    ax.add_patch(plt.Polygon([(-0.78, 0.65), (-0.58, 1.18), (-0.34, 0.78)],
                              closed=True, facecolor=CAT_PINK, edgecolor="none"))
    ax.add_patch(plt.Polygon([(0.78, 0.65), (0.58, 1.18), (0.34, 0.78)],
                              closed=True, facecolor=CAT_PINK, edgecolor="none"))
    # head
    ax.add_patch(plt.Circle((0, 0), 1.0, facecolor=CAT_FUR, edgecolor=style.INK, lw=1.8))
    # asymmetric marking, so flips/rotations are visually obvious
    ax.add_patch(plt.Circle((0.55, 0.35), 0.32, facecolor=CAT_PATCH, edgecolor="none", alpha=0.9))
    # eyes
    for sx in (-1, 1):
        ax.add_patch(plt.Circle((sx * 0.38, 0.12), 0.16, facecolor=style.INK))
        ax.add_patch(plt.Circle((sx * 0.33, 0.17), 0.045, facecolor="white"))
    # nose
    ax.add_patch(plt.Polygon([(-0.10, -0.15), (0.10, -0.15), (0, -0.32)],
                              closed=True, facecolor=CAT_PINK, edgecolor=style.INK, lw=1.0))
    # mouth
    t = np.linspace(0, np.pi, 30)
    for sx in (-1, 1):
        ax.plot(sx * 0.22 * np.sin(t), -0.32 - 0.16 * (1 - np.cos(t)), color=style.INK, lw=1.6)
    # whiskers
    for sx in (-1, 1):
        for dy in (-0.05, -0.16, -0.27):
            ax.plot([sx * 0.35, sx * 1.35], [-0.20 + dy, -0.26 + dy], color=style.INK, lw=1.0, alpha=0.8)


def render_rgba(res=400):
    rfig = plt.figure(figsize=(res / 100, res / 100), dpi=100)
    rax = rfig.add_axes([0, 0, 1, 1])
    draw_cat(rax)
    rfig.canvas.draw()
    buf = np.asarray(rfig.canvas.buffer_rgba()).copy()
    plt.close(rfig)
    return buf


def zoom_crop(arr, factor):
    h, w = arr.shape[:2]
    out = ndimage.zoom(arr, (factor, factor, 1), order=1)
    hh, ww = out.shape[:2]
    if factor >= 1:
        y0, x0 = (hh - h) // 2, (ww - w) // 2
        return out[y0:y0 + h, x0:x0 + w]
    canvas = np.zeros_like(arr)
    y0, x0 = (h - hh) // 2, (w - ww) // 2
    canvas[y0:y0 + hh, x0:x0 + ww] = out
    return canvas


def brightness(arr, factor):
    out = arr.astype(float)
    out[:, :, :3] = np.clip(out[:, :, :3] * factor, 0, 255)
    return out.astype(np.uint8)


base = render_rgba()

fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.imshow(base)
ax.axis("off")
style.save(fig, "aug_original.svg")

augs = [
    (np.fliplr(base), style.t("horizontal flip", "retournement horizontal")),
    (ndimage.rotate(base, 18, reshape=False, order=1, cval=0), style.t("rotation", "rotation")),
    (zoom_crop(base, 1.35), style.t("random crop / zoom", "recadrage / zoom aléatoire")),
    (brightness(base, 1.35), style.t("brightness +", "luminosité +")),
    (brightness(base, 0.65), style.t("brightness -", "luminosité -")),
    (ndimage.rotate(ndimage.shift(base, (-10, 25, 0), order=1, cval=0), -15,
                     reshape=False, order=1, cval=0), style.t("shift + rotation", "translation + rotation")),
]

fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.2))
for ax, (arr, label) in zip(axes.flat, augs):
    ax.imshow(arr)
    ax.axis("off")
    ax.set_title(label, fontsize=12, color=style.GREY_DARK, pad=6)
fig.subplots_adjust(wspace=0.05, hspace=0.15)
style.save(fig, "aug_grid.svg")

# --- multi-channel convolution: sum over channels, then stack feature maps ----
# Same solarized palette as the Dumoulin & Visin conv-arithmetic figures:
# blue input, gray kernel (their shaded "base02" overlay), cyan output.
# Flat 2D (axis-aligned), with a thin diagonal offset per plane to suggest the
# channel stack, rather than a full 3D/isometric perspective.
DUMOULIN_BLUE = "#268BD2"
DUMOULIN_GRAY = "#586E75"
DUMOULIN_CYAN = "#2AA198"
DUMOULIN_EDGE = "#002B36"

N_IN, N_OUT = 3, 2  # input channels, kernels / output channels


def _draw_stack(ax, cx, cy, n, size, color, off, alpha=0.95, lw=1.3, zorder=2):
    for i in range(n - 1, -1, -1):
        x, y = cx - size / 2 + i * off, cy - size / 2 + i * off
        a = alpha if i == 0 else alpha * 0.75
        ax.add_patch(plt.Rectangle((x, y), size, size, facecolor=color, alpha=a,
                                    edgecolor=DUMOULIN_EDGE, lw=lw, zorder=zorder + (n - i)))


def _arrow(ax, p0, p1, lw=1.3, shrink=10):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=DUMOULIN_EDGE, lw=lw,
                                shrinkA=shrink, shrinkB=shrink))


fig, ax = plt.subplots(figsize=(7.4, 6.2))

SIZE, OFF = 2.2, 0.35
Y_IN, Y_K, Y_FM, Y_OUT = 0, 5.2, 9.6, 14.4
x_centers = [-4.2, 4.2]

_draw_stack(ax, 0, Y_IN, N_IN, SIZE, DUMOULIN_BLUE, OFF)

for i, xc in enumerate(x_centers):
    _arrow(ax, (0, Y_IN + SIZE / 2 + 0.3), (xc, Y_K - SIZE / 2 - 0.2))
    _draw_stack(ax, xc, Y_K, N_IN, SIZE * 0.7, DUMOULIN_GRAY, OFF)
    _arrow(ax, (xc, Y_K + SIZE * 0.35 + 0.2), (xc, Y_FM - SIZE / 2 - 0.2))
    _draw_stack(ax, xc, Y_FM, 1, SIZE, DUMOULIN_CYAN, OFF)
    _arrow(ax, (xc, Y_FM + SIZE / 2 + 0.2), (0, Y_OUT - SIZE / 2 - 0.2 + i * OFF))

_draw_stack(ax, 0, Y_OUT, N_OUT, SIZE, DUMOULIN_CYAN, OFF, zorder=10)

ax.text(-6.6, Y_IN, style.t(f"input\n({N_IN} channels)", f"entrée\n({N_IN} canaux)"), fontsize=12, color=style.INK,
        ha="right", va="center")
ax.text(-6.6, Y_K, style.t("kernels", "noyaux"), fontsize=11, color=style.GREY_DARK,
        ha="right", va="center")
ax.text(-6.6, Y_FM, style.t("feature maps", "cartes d'activation"), fontsize=11, color=style.GREY_DARK,
        ha="right", va="center")
ax.text(-6.6, Y_OUT, style.t(f"output\n({N_OUT} channels)", f"sortie\n({N_OUT} canaux)"), fontsize=12, color=style.INK,
        ha="right", va="center")

ax.set_xlim(-7.4, 6.5)
ax.set_ylim(Y_IN - SIZE, Y_OUT + SIZE + 1)
ax.set_aspect("equal")
ax.axis("off")

style.save(fig, "conv_multichannel.svg")

# --- MNIST digit grid: one real sample per class, black on white --------------
mnist = np.load(os.path.expanduser("~/.keras/datasets/mnist.npz"))
x_train, y_train = mnist["x_train"], mnist["y_train"]

n_cols, n_rows = 5, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4))
for digit, ax in enumerate(axes.flat):
    idx = np.where(y_train == digit)[0][0]
    ax.imshow(x_train[idx], cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(style.GREY)
        s.set_linewidth(1.2)
fig.subplots_adjust(wspace=0.08, hspace=0.08)
style.save(fig, "mnist_grid.svg")

# --- batch norm: (N, C, H*W) tensor cube, one channel slice highlighted -------
# Homemade version of the classic "normalization axes" cube figure (as in
# Wu & He, "Group Normalization", 2018): batch norm normalizes each channel
# across the batch and spatial dimensions, i.e. one C-slice spans all of N, H*W.
N_b, C_b, HW_b = 5, 4, 6  # batch, channels, flattened spatial (H x W)

filled = np.ones((C_b, N_b, HW_b), dtype=bool)
facecolors = np.empty(filled.shape + (4,), dtype=float)
facecolors[...] = (0.93, 0.93, 0.93, 1.0)
r, g, b = mcolors.to_rgb(style.PURPLE)
facecolors[C_b - 1, :, :] = (r, g, b, 1.0)  # one channel, all N and H, W

fig = plt.figure(figsize=(5.6, 5.6))
ax = fig.add_axes([-0.05, -0.05, 1.1, 1.05], projection="3d")
ax.voxels(filled, facecolors=facecolors, edgecolors=style.GREY_DARK, linewidth=0.6,
          shade=False)

ax.set_box_aspect((C_b, N_b, HW_b))
ax.view_init(elev=18, azim=-60)
ax.set_axis_off()

ax.text2D(0.06, 0.5, "H, W", transform=ax.transAxes, fontsize=15, color=style.INK,
          ha="center", va="center", rotation=90)
ax.text2D(0.30, 0.04, "C", transform=ax.transAxes, fontsize=15, color=style.INK,
          ha="center", va="center")
ax.text2D(0.78, 0.04, "N", transform=ax.transAxes, fontsize=15, color=style.INK,
          ha="center", va="center")

style.save(fig, "batchnorm_cube.svg")
