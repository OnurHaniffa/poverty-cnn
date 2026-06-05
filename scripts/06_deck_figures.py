"""Restyle the deck's data figures in the light-premium brand style (Avenir Next).

Reads precomputed PCA diagnostics so it runs instantly (no DHS reload):
  results/eda/pca_evr.npy, results/eda/pc1_loadings.csv,
  data/processed/wealth_index_clusters.csv
All figures saved transparent (they sit on white cards in the deck), title-free
(the slide provides the title), and de-cluttered.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#1f3a5f"; TEAL = "#178a7a"; AMBER = "#e0922f"
SLATE = "#2b3440"; GREY = "#9aa4ae"; GRID = "#e6ebf0"
plt.rcParams.update({
    "font.family": "Avenir Next", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c7d0d9", "axes.linewidth": 1.0,
    "text.color": SLATE, "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE,
})
E = Path("results/figures/eda")


def save(fig, name):
    fig.savefig(E / name, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", name)


# --- scree (PC1 highlighted) ---
evr = np.load("results/eda/pca_evr.npy") * 100
n = len(evr); null = 100 / n
fig, ax = plt.subplots(figsize=(7.4, 5.0))
ax.bar(range(1, n + 1), evr, color=[AMBER] + [TEAL] * (n - 1), zorder=3, width=0.78)
ax.axhline(null, color=GREY, ls=(0, (5, 4)), lw=1.6, zorder=2)
ax.text(n, null + 0.7, f"random baseline  {null:.1f}%", ha="right", va="bottom", color=GREY, fontsize=12)
ax.text(3.4, evr[0] - 2.5, f"PC1 = {evr[0]:.1f}%", ha="left", va="center", color=AMBER, fontweight="bold", fontsize=15)
ax.set_xticks(range(1, n + 1)); ax.set_xlabel("Principal component"); ax.set_ylabel("% of variance explained")
ax.set_axisbelow(True); ax.yaxis.grid(True, color=GRID, lw=1)
save(fig, "01_pca_scree.png")

# --- PC1 loadings ---
ld = pd.read_csv("results/eda/pc1_loadings.csv").sort_values("PC1")
fig, ax = plt.subplots(figsize=(7.0, 6.2))
ax.barh(ld["feature"], ld["PC1"], color=[AMBER if v < 0.02 else TEAL for v in ld["PC1"]], zorder=3, height=0.74)
ax.axvline(0, color="#c7d0d9", lw=1)
ax.set_xlabel("PC1 loading (the wealth recipe)")
ax.set_axisbelow(True); ax.xaxis.grid(True, color=GRID, lw=1)
save(fig, "02_pca_loadings.png")

# --- urban vs rural ---
c = pd.read_csv("data/processed/wealth_index_clusters.csv")
fig, ax = plt.subplots(figsize=(7.4, 5.0))
ax.hist(c.loc[c.urban == "R", "wealth_index_mean"], bins=46, color=AMBER, alpha=0.78, label="Rural", zorder=3)
ax.hist(c.loc[c.urban == "U", "wealth_index_mean"], bins=46, color=TEAL, alpha=0.78, label="Urban", zorder=3)
ax.set_xlabel("cluster wealth index"); ax.set_ylabel("clusters")
ax.legend(frameon=False, fontsize=13)
ax.set_axisbelow(True); ax.yaxis.grid(True, color=GRID, lw=1)
save(fig, "07_urban_rural.png")

# --- cluster map ---
m = c[(c.lat != 0) | (c.lon != 0)]
fig, ax = plt.subplots(figsize=(6.6, 6.6))
sc = ax.scatter(m.lon, m.lat, c=m.wealth_index_mean, cmap="RdYlBu_r", s=5, alpha=0.72,
                vmin=-1.5, vmax=1.5, linewidths=0)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
cb = fig.colorbar(sc, fraction=0.046, pad=0.04); cb.set_label("wealth index"); cb.outline.set_visible(False)
save(fig, "10_cluster_map.png")
print("done")
