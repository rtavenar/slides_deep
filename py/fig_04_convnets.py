"""Figures for deck 04 (CNNs).

- lsvrc.svg : ImageNet top-5 error by year (shallow vs deep)

Convolution-arithmetic figures (no-pad, padding, strides) live in
fig_04_conv_arithmetic.py.
"""
import matplotlib.pyplot as plt
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
ax.text(5.4, 5.6, "human ~5%", ha="right", color=style.GREY_DARK, fontsize=10)
for b, e in zip(bars, err):
    ax.text(b.get_x() + b.get_width() / 2, e + 0.4, f"{e:.1f}", ha="center",
            fontsize=10)
ax.set_ylabel("top-5 error (%)")
ax.set_title("ImageNet (LSVRC): shallow (orange) → deep (purple)")
style.save(fig, "lsvrc.png")
