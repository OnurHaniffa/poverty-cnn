"""Measure the root-cause claim: is the night-light signal flat/absent for the poorest?

For a large tile sample, extract the night-lights band (channel 7), compute each tile's mean
intensity, and bin by TRUE wealth decile. If the poorest deciles have near-zero, flat
night-light intensity (and night-lights barely correlate with wealth WITHIN the poor), then the
satellite literally cannot separate the extreme poor — converting our central thesis from
*asserted* to *measured*. Also reports the label-noise alternative proxy (within-decile spread).

Channel order: 0-2 RGB, 3 NIR, 4-5 SWIR, 6 thermal, 7 nightlights.
Run on PC: PPY scripts/49_nightlight_by_wealth.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CACHE = "data/processed/tile_cache_full"; NL = 7; NSAMP = 14000
NAVY = "#1f3a5f"; TEAL = "#178a7a"; AMBER = "#e0922f"; GREY = "#9aa4ae"
OUT = Path("results/figures/teaching"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    cache = np.load(f"{CACHE}/cache.npy", mmap_mode="r")
    meta = pd.read_csv(f"{CACHE}/cache_metadata.csv")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(meta), min(NSAMP, len(meta)), replace=False))
    nl_mean = np.array([float(np.asarray(cache[int(i), NL]).mean()) for i in idx])
    y = meta.wealth_index_mean.values[idx]
    dec = pd.qcut(y, 10, labels=False)

    g = pd.DataFrame({"y": y, "nl": nl_mean, "dec": dec}).groupby("dec")
    nl_by = g["nl"].mean().values; nl_sd = g["nl"].std().values; y_by = g["y"].mean().values
    # discriminative power of nightlights: Spearman(nl, wealth) overall vs within poorest 30%
    poor = dec <= 2
    sp_all = float(spearmanr(nl_mean, y)[0])
    sp_poor = float(spearmanr(nl_mean[poor], y[poor])[0])
    sp_rich = float(spearmanr(nl_mean[dec >= 7], y[dec >= 7])[0])

    print("=== night-light intensity by TRUE wealth decile ===")
    for d in range(10):
        print(f"  decile {d} (wealth {y_by[d]:+.2f}): NL mean {nl_by[d]:+.3f} ± {nl_sd[d]:.3f}")
    print(f"\n  Spearman(nightlights, wealth):  overall {sp_all:+.3f} | poorest-30% {sp_poor:+.3f} | richest-30% {sp_rich:+.3f}")
    print("  → if poorest-30% ≈ 0 while richest-30% is high, the NL signal is ABSENT for the poor (measured, not asserted)")
    json.dump({"nl_by_decile": nl_by.tolist(), "wealth_by_decile": y_by.tolist(),
               "spearman_nl_wealth_overall": sp_all, "spearman_poorest30": sp_poor,
               "spearman_richest30": sp_rich, "n": int(len(idx))},
              open("results/nightlight_by_wealth.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.errorbar(range(10), nl_by, yerr=nl_sd, fmt="-o", color=TEAL, lw=2.3, ms=7, capsize=4, label="mean tile night-light")
    ax.axhspan(-99, nl_by[0] + nl_sd[0], xmin=0, xmax=0.30, color=AMBER, alpha=0.10)
    ax.set_xticks(range(10)); ax.set_xlabel("true-wealth decile  (0 = poorest)")
    ax.set_ylabel("night-light intensity (normalised)")
    ax.set_ylim(min(nl_by) - max(nl_sd) - 0.05, max(nl_by) + max(nl_sd) + 0.05)
    ax.set_title("The signal isn't there: night-lights are flat across the bottom deciles",
                 color=NAVY, fontweight="bold")
    ax.text(0.4, nl_by[0], f"poorest-30%: NL↔wealth ρ={sp_poor:+.2f}\n(richest-30% ρ={sp_rich:+.2f})",
            fontsize=10, color=AMBER, va="center")
    ax.grid(True, color="#eee"); fig.tight_layout()
    fig.savefig(OUT / "nightlight_by_wealth.png", dpi=170, bbox_inches="tight", facecolor="white")
    print("\nwrote results/nightlight_by_wealth.json + nightlight_by_wealth.png")


if __name__ == "__main__":
    main()
