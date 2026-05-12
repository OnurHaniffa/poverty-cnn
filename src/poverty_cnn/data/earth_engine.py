"""Earth Engine pipeline for Landsat + nighttime-light extraction.

Replicates Yeh et al. (2020) protocol: 8-channel 224×224 tiles, 30 m/pixel,
6.72 km × 6.72 km on the ground, 3-year median composites.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import ee
from dotenv import load_dotenv


# Yeh 2020 protocol constants
PATCH_HALF_SIDE_M = 3360   # 6.72 km / 2 = 3360 m radius
PATCH_SCALE_M = 30          # Landsat surface-reflectance native resolution
PATCH_SIZE_PX = 224         # Final patch dimensions (224 × 30 = 6720 m ≈ 6.72 km)

# Cloud-cover threshold for individual Landsat scenes (matches Yeh 2020)
MAX_CLOUD_COVER = 30        # percent

# 3-year composite windows (matches Yeh 2020 §Methods)
# Maps survey year → (start_year, end_year) for the median composite
COMPOSITE_WINDOWS = {
    "P1": (2009, 2011),
    "P2": (2012, 2014),
    "P3": (2015, 2017),
}

# Per-sensor band ordering. We emit in this canonical order:
#   RED, GREEN, BLUE, NIR, SWIR1, SWIR2, TEMP1
# Sensor-specific band IDs come from the Landsat Collection 2 Level-2 docs.
LANDSAT_BANDS = {
    8: ["SR_B4", "SR_B3", "SR_B2", "SR_B5", "SR_B6", "SR_B7", "ST_B10"],
    7: ["SR_B3", "SR_B2", "SR_B1", "SR_B4", "SR_B5", "SR_B7", "ST_B6"],
    5: ["SR_B3", "SR_B2", "SR_B1", "SR_B4", "SR_B5", "SR_B7", "ST_B6"],
}

LANDSAT_COLLECTIONS = {
    8: "LANDSAT/LC08/C02/T1_L2",
    7: "LANDSAT/LE07/C02/T1_L2",
    5: "LANDSAT/LT05/C02/T1_L2",
}

CANONICAL_BAND_NAMES = ["RED", "GREEN", "BLUE", "NIR", "SWIR1", "SWIR2", "TEMP1"]
NIGHTLIGHTS_BAND_NAME = "NL"

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_ee(env_path: str | Path | None = None) -> None:
    """Initialize Earth Engine using the project ID stored in .env.

    Args:
        env_path: Optional explicit path to a .env file. If omitted, looks
            for a .env in the current working directory or any parent.
    """
    if env_path is not None:
        load_dotenv(env_path)
    else:
        load_dotenv()

    project = os.environ.get("GEE_PROJECT_ID")
    if not project:
        raise RuntimeError(
            "GEE_PROJECT_ID not set in environment. Put it in .env or export it."
        )
    ee.Initialize(project=project)


# ---------------------------------------------------------------------------
# Sensor and window selection
# ---------------------------------------------------------------------------

def landsat_sensor_for_year(year: int) -> int:
    """Choose the Landsat sensor appropriate to the survey year.

    Landsat 8 launched 2013; Landsat 7 from 1999 (SLC-off since 2003);
    Landsat 5 retired 2013. We pick the newest sensor with full coverage for
    the year's composite window.
    """
    if year >= 2013:
        return 8
    if year >= 1999:
        return 7
    return 5


def composite_window_for_year(year: int) -> tuple[int, int]:
    """Map a DHS survey year to its 3-year composite window."""
    if year <= 2011:
        return COMPOSITE_WINDOWS["P1"]
    if year <= 2014:
        return COMPOSITE_WINDOWS["P2"]
    return COMPOSITE_WINDOWS["P3"]


# ---------------------------------------------------------------------------
# Composite builders
# ---------------------------------------------------------------------------

def multispectral_composite(point: ee.Geometry.Point, year: int) -> ee.Image:
    """Build a 7-band median multispectral composite for the cluster's window.

    Bands are output in canonical order: RED, GREEN, BLUE, NIR, SWIR1,
    SWIR2, TEMP1.
    """
    sensor = landsat_sensor_for_year(year)
    start, end = composite_window_for_year(year)
    collection_id = LANDSAT_COLLECTIONS[sensor]
    sensor_bands = LANDSAT_BANDS[sensor]

    composite = (
        ee.ImageCollection(collection_id)
        .filterBounds(point)
        .filterDate(f"{start}-01-01", f"{end}-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_COVER))
        .median()
        .select(sensor_bands, CANONICAL_BAND_NAMES)
    )
    return composite


def nightlights_composite(point: ee.Geometry.Point, year: int) -> ee.Image:
    """Build a 1-band median nighttime-lights composite for the year's window.

    Uses DMSP-OLS for surveys before 2012, VIIRS for 2012 and later
    (matches Yeh 2020).
    """
    start, end = composite_window_for_year(year)

    if year < 2012:
        # DMSP-OLS stable-lights band, 2009-2011 era
        return (
            ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS")
            .filterDate(f"{start}-01-01", f"{end}-12-31")
            .select("stable_lights")
            .median()
            .rename(NIGHTLIGHTS_BAND_NAME)
        )

    # VIIRS DNB monthly composites
    return (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(f"{start}-01-01", f"{end}-12-31")
        .select("avg_rad")
        .median()
        .rename(NIGHTLIGHTS_BAND_NAME)
    )


def cluster_image(lat: float, lon: float, year: int) -> ee.Image:
    """Build the full 8-channel image for a cluster (no export yet)."""
    point = ee.Geometry.Point([lon, lat])
    multispectral = multispectral_composite(point, year)
    nightlights = nightlights_composite(point, year)
    return multispectral.addBands(nightlights)


def cluster_region(lat: float, lon: float) -> ee.Geometry.Polygon:
    """The 6.72 km × 6.72 km square around a cluster's GPS point."""
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(PATCH_HALF_SIDE_M).bounds()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@dataclass
class TileExportTask:
    """Handle for an in-flight EE export to Google Drive."""

    task: ee.batch.Task
    cluster_id: str
    description: str
    drive_folder: str


def export_cluster_to_drive(
    *,
    lat: float,
    lon: float,
    year: int,
    cluster_id: str,
    drive_folder: str = "poverty_cnn_data",
) -> TileExportTask:
    """Kick off an asynchronous export of one cluster's 8-channel tile.

    Returns immediately with a task handle. The actual export runs on
    Google's servers, the result lands in the user's Google Drive.

    Tile naming: `tile_<cluster_id>.tif`.
    """
    description = f"tile_{cluster_id}"
    image = cluster_image(lat, lon, year)
    region = cluster_region(lat, lon)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=drive_folder,
        fileNamePrefix=description,
        region=region,
        scale=PATCH_SCALE_M,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    task.start()
    return TileExportTask(task, cluster_id, description, drive_folder)


# ---------------------------------------------------------------------------
# Small-batch download (no Drive round-trip)
# ---------------------------------------------------------------------------

def download_cluster_tile_direct(
    *,
    lat: float,
    lon: float,
    year: int,
    output_path: str | Path,
    timeout: int = 600,
) -> Path:
    """Download a single cluster tile directly as a local GeoTIFF.

    Uses `ee.Image.getDownloadURL`, which is fine for small numbers of
    tiles (debugging, sanity checks). For full 19 000-cluster extraction,
    use `export_cluster_to_drive` instead — much higher throughput.

    The default 10-minute timeout accommodates Earth Engine compute time
    on the server side, which can be slow for the first request to a
    new geographic area or busy clusters.
    """
    import requests

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cluster_image(lat, lon, year)
    region = cluster_region(lat, lon)

    url = image.getDownloadURL(
        {
            "scale": PATCH_SCALE_M,
            "crs": "EPSG:4326",
            "region": region,
            "format": "GEO_TIFF",
        }
    )

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    return output_path
