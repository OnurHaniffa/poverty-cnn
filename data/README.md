# Data

This directory holds raw and processed data. **Nothing in this directory is committed to git** (see `.gitignore`).

## Layout

```
data/
├── raw/
│   ├── dhs/         # DHS survey CSVs and shapefiles, organized by country
│   ├── landsat/     # 8-channel multispectral tiles from Earth Engine
│   └── nightlights/ # DMSP-OLS / VIIRS nighttime light tiles
└── processed/
    ├── tensors/     # 8-channel float32 .npy files, one per cluster
    ├── metadata.csv # cluster_id, country, year, urban, wealth_index, lat, lon
    └── splits/      # 5-fold cross-country split files (Yeh 2020 Supp Table S2)
```

## How to obtain the data

### DHS surveys

1. Register at [dhsprogram.com](https://dhsprogram.com/data/new-user-registration.cfm).
2. Submit a project description (1-3 day approval).
3. Request access to all 23 sub-Saharan African countries used in this project.
4. Download the household recode files (`*HR*.DTA`) and GPS shapefiles (`*GE*.SHP`) for the most recent round per country.
5. Place under `data/raw/dhs/<country_code>/`.

### Landsat + nighttime lights

Run the Earth Engine download script:

```bash
earthengine authenticate
python scripts/03_download_imagery.py
```

Will pull Landsat 5/7/8 surface reflectance + DMSP-OLS / VIIRS nightlights for every DHS cluster. Expects 24-48h wall-clock for all 23 countries.

### WILDS PovertyMap (sanity check)

```python
import wilds
dataset = wilds.get_dataset("poverty", download=True, root_dir="data/raw/wilds_poverty")
```

Used to verify that our raw-pipeline output matches the published benchmark. ~13 GB.
