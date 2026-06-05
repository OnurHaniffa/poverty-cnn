"""Slide-ready methodology diagrams for the progress deck.

Three clean, high-DPI diagrams (no data load needed):
  1. big_picture  — how DHS + satellite + CNN fit together
  2. wealth_pipeline — survey -> pooled PCA -> village wealth target
  3. ee_bridge   — DHS GPS point -> Earth Engine -> 8-channel tile

Run: python scripts/05_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("results/figures/diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# Palette
NAVY = "#1f3a5f"
BLUE = "#2e86ab"
GREEN = "#178a7a"
GREEN_L = "#d8efe9"
GOLD = "#e0922f"
GOLD_L = "#fbe9cf"
INK = "#16202b"
GREY = "#6b7682"
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 12})


def box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=12, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.14",
        fc=fc, ec=ec, lw=2.2, mutation_aspect=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=2.4, ls="-", label=None, lc=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20,
        lw=lw, color=color, ls=ls, shrinkA=3, shrinkB=3, zorder=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", va="bottom",
                fontsize=9.5, color=lc or color, style="italic", zorder=3)


def new(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def chain(ax, steps, y, h, theme_fc, theme_l, x0=0.3, total=13.5, gap=0.22):
    n = len(steps)
    w = (total - x0 * 2 - gap * (n - 1)) / n
    xs = [x0 + i * (w + gap) for i in range(n)]
    for i, (x, t) in enumerate(zip(xs, steps)):
        fc = theme_fc if i in (0, n - 1) else theme_l
        tc = "white" if i in (0, n - 1) else INK
        box(ax, x, y, w, h, t, fc=fc, ec=theme_fc, tc=tc,
            fs=11, bold=i in (0, n - 1))
        if i < n - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2, color=theme_fc)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", OUT / name)


# --- 1. Big picture ---------------------------------------------------------
fig, ax = new(13.5, 5.6)
box(ax, 0.4, 3.6, 3.0, 1.2, "Satellite imagery\nLandsat + nightlights", GREEN_L, GREEN, fs=12, bold=True)
box(ax, 5.0, 3.6, 2.7, 1.2, "8-channel tile\n(input  x)", GREEN_L, GREEN, fs=12)
arrow(ax, 3.4, 4.2, 5.0, 4.2, color=GREEN, label="Earth Engine")

box(ax, 0.4, 0.7, 3.0, 1.2, "DHS household\nsurveys", GOLD_L, GOLD, fs=12, bold=True)
box(ax, 5.0, 0.7, 2.7, 1.2, "Wealth index\n(target  y)", GOLD_L, GOLD, fs=12)
arrow(ax, 3.4, 1.3, 5.0, 1.3, color=GOLD, label="pooled PCA")

# GPS bridge (dotted): DHS location picks the satellite tile
arrow(ax, 1.9, 1.9, 4.8, 3.7, color=BLUE, lw=1.8, ls=(0, (4, 3)),
      label="GPS coordinate\nlocates the tile", lc=BLUE)

box(ax, 9.0, 2.0, 2.7, 1.5, "CNN\n8-channel\nResNet-18", NAVY, NAVY, tc="white", fs=13, bold=True)
arrow(ax, 7.7, 4.2, 9.2, 3.4, color=NAVY)               # tile -> CNN
arrow(ax, 7.7, 1.3, 9.2, 2.3, color=NAVY, label="supervises", lc=NAVY)  # y -> CNN
box(ax, 12.0, 2.2, 1.2, 1.1, "ŷ", BLUE, BLUE, tc="white", fs=20, bold=True)
arrow(ax, 11.7, 2.75, 12.0, 2.75, color=NAVY)
save(fig, "01_big_picture.png")

# --- 2. Wealth pipeline -----------------------------------------------------
fig, ax = new(13.5, 4.0)
chain(ax, [
    "15 asset features\nper household",
    "Pool 355,445\nhouseholds\n(23 countries)",
    "Standardize\n(z-score)",
    "PCA\nPC1 = wealth axis",
    "Re-standardize\n+ sign-fix",
    "Average to\n13,634 villages\n=  y",
], y=0.85, h=2.3, theme_fc=GOLD, theme_l=GOLD_L)
save(fig, "02_wealth_pipeline.png")

# --- 3. Earth Engine bridge -------------------------------------------------
fig, ax = new(13.5, 4.2)
chain(ax, [
    "DHS cluster\nGPS point\n(lat, lon)",
    "Earth Engine\ncenters a\n6.72 km box",
    "Landsat L8/L7/L5\n+ nightlights\n3-year median",
    "8-channel tile\n224×224 px\n@ 30 m",
    "linked by\ncluster_id\nto wealth  y",
], y=1.15, h=2.4, theme_fc=GREEN, theme_l=GREEN_L)
ax.text(6.75, 0.5, "RGB · NIR · SWIR1 · SWIR2 · thermal · nightlights",
        ha="center", fontsize=10, color=GREY, style="italic")
save(fig, "03_ee_bridge.png")

print("done")
