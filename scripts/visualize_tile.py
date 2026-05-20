"""Quick visualization of the Kenya test tiles to see what the CNN actually sees.

For each tile, renders:
  1. Composite view  — true-color RGB, false-color (NIR-R-G), nightlights side by side.
  2. All 8 bands     — 2x4 grid of grayscale panels, one per band.

Each band is independently percentile-stretched (2nd→98th) so its dynamic range
is visible. Without this, the thermal and nightlights bands would look completely
black or completely white because their natural value ranges are very different
from the visible bands.

Outputs land in results/figures/tile_viz/ (gitignored).

Run with the env's python:
    /opt/homebrew/Caskroom/miniforge/base/envs/poverty-cnn/bin/python scripts/visualize_tile.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# Order matches the 8-channel canonical order from earth_engine.py
BAND_NAMES = ["RED", "GREEN", "BLUE", "NIR", "SWIR1", "SWIR2", "TEMP1", "NL"]

TILES = {
    "Nairobi (urban, wealthy)": Path("data/processed/test_kenya/KE_urban_rich_cluster47.tif"),
    "Turkana (rural, poor)":    Path("data/processed/test_kenya/KE_rural_poor_cluster1015.tif"),
}

OUTPUT_DIR = Path("results/figures/tile_viz")


def stretch(arr: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Linearly stretch an array to [0, 1] using percentile bounds.

    Percentile clipping prevents a handful of outlier pixels from dominating
    the displayed dynamic range — without it, one bright cloud or saturated
    pixel can flatten the rest of the image into a gray smear.
    """
    a = arr.astype(np.float32)
    valid = a[np.isfinite(a)]
    if valid.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(valid, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def center_crop(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """Center-crop a (bands, H, W) array to (bands, size, size).

    EE returns tiles at 225x224 or 226x224 due to bounds rounding;
    we force the canonical 224x224 for consistency with the CNN's input shape.
    """
    _, h, w = arr.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[:, top:top + size, left:left + size]


def load_tile(path: Path) -> np.ndarray:
    """Load a multi-band GeoTIFF as (bands, H, W) and center-crop to 224x224."""
    with rasterio.open(path) as src:
        arr = src.read()
    if arr.shape[1] != 224 or arr.shape[2] != 224:
        arr = center_crop(arr, 224)
    return arr


def plot_composites(tile: np.ndarray, title: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. True-color RGB — what your eyes would see from space
    rgb = np.stack([stretch(tile[0]), stretch(tile[1]), stretch(tile[2])], axis=-1)
    axes[0].imshow(rgb)
    axes[0].set_title("True-color RGB\n(what your eyes would see)", fontsize=11)
    axes[0].axis("off")

    # 2. False-color NIR→R, RED→G, GREEN→B — vegetation glows red
    false_color = np.stack([stretch(tile[3]), stretch(tile[0]), stretch(tile[1])], axis=-1)
    axes[1].imshow(false_color)
    axes[1].set_title("False-color (NIR→R, RED→G, GREEN→B)\n(healthy vegetation → red)", fontsize=11)
    axes[1].axis("off")

    # 3. Nightlights — electricity at night
    axes[2].imshow(stretch(tile[7]), cmap="inferno")
    axes[2].set_title("Nightlights (NL band)\n(brightness at night ≈ electrification)", fontsize=11)
    axes[2].axis("off")

    fig.suptitle(f"{title} — composite views", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all_bands(tile: np.ndarray, title: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(stretch(tile[i]), cmap="gray")
        ax.set_title(f"Band {i}: {BAND_NAMES[i]}", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{title} — all 8 bands, each rendered as grayscale", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_band_stats(tile: np.ndarray) -> None:
    for i, name in enumerate(BAND_NAMES):
        valid = tile[i].astype(float)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            print(f"    {name:6s}: no valid pixels")
            continue
        print(f"    {name:6s}  min={valid.min():>12.2f}  max={valid.max():>12.2f}  "
              f"mean={valid.mean():>10.2f}  std={valid.std():>10.2f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, path in TILES.items():
        if not path.exists():
            print(f"!! missing: {path}")
            continue
        print(f"\n=== {label} ===")
        print(f"    path: {path}")
        tile = load_tile(path)
        print(f"    shape after crop: {tile.shape}  dtype: {tile.dtype}")
        print_band_stats(tile)

        slug = label.split(" ")[0].lower()
        comp_out = OUTPUT_DIR / f"{slug}_composites.png"
        bands_out = OUTPUT_DIR / f"{slug}_all_bands.png"
        plot_composites(tile, label, comp_out)
        plot_all_bands(tile, label, bands_out)
        print(f"    saved → {comp_out}")
        print(f"    saved → {bands_out}")


if __name__ == "__main__":
    main()
