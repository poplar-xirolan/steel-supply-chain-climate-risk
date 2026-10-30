#!/usr/bin/env python
# coding: utf-8

"""Extract ISIMIP tropical-cyclone wind-lifetime maxima at facilities."""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
import os
from pathlib import Path

ROOT = bundle_root()
parser = argparse.ArgumentParser()
parser.add_argument("--facilities-dir", type=Path, default=data_path('input/climate_facilities'))
parser.add_argument("--netcdf-dir", type=Path, default=data_path('input/isimip/wind'))
parser.add_argument("--output-dir", type=Path, default=data_path('work/climate_at_facilities/wind'))
parser.add_argument("--shard-count", type=int, default=1)
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--resume-output", action="store_true")
parser.add_argument(
    "--include-nonformal-inputs",
    action="store_true",
    help="Also extract source years/model members that step 08 never uses.",
)
parser.add_argument(
    "--selection-only",
    action="store_true",
    help="Report the selected formal NetCDF inputs and exit before extraction.",
)
args = parser.parse_args()
if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
    parser.error("--shard-index must be in [0, --shard-count)")

# 读取`projects/2_research/个人研究项目/高时空分辨率金属风险影响范围、路径研究与对策分析/子部分 MFA网络空间化/2_数据整理/2_规范范式整理/SCHEMA.md`处定义的 `basic_info` 范式的 parquet 数据，根据坐标信息，获取`./1.2_wind_download`处的 isimip3b 的 windlifetimemax 变量，并存储为 parquet 文件。
# 

# In[1]:


import pandas as pd
import xarray as xr
import glob
from tqdm import tqdm
import dask
from dask.distributed import Client

FORMAL_YEAR_POLICY = {
    "picontrol": frozenset(range(2015, 2101)),
    "ssp126": frozenset(range(2061, 2101)),
    "ssp370": frozenset(range(2015, 2101)),
    "ssp585": frozenset(range(2015, 2101)),
}


def parse_wind_source(path):
    """Parse the annual wind filename contract also used by step 08."""

    parts = Path(path).stem.split("_")
    if len(parts) < 6:
        return None
    try:
        year = int(parts[5])
    except ValueError:
        return None
    return {
        "path": path,
        "model": parts[2],
        "forcing_member": f"{parts[0]}_{parts[1]}",
        "scenario": parts[3],
        "variable": parts[4],
        "year": year,
    }


def select_formal_wind_sources(paths):
    """Keep only complete model-member series that can contribute to step 08."""

    parsed = [item for path in paths if (item := parse_wind_source(path)) is not None]
    available = {}
    for item in parsed:
        key = (item["scenario"], item["model"], item["forcing_member"])
        available.setdefault(key, set()).add(item["year"])

    valid = {
        key
        for key, years in available.items()
        if key[0] in FORMAL_YEAR_POLICY and FORMAL_YEAR_POLICY[key[0]].issubset(years)
    }
    return [
        item["path"]
        for item in parsed
        if (
            (item["scenario"], item["model"], item["forcing_member"]) in valid
            and item["year"] in FORMAL_YEAR_POLICY[item["scenario"]]
            and item["variable"] == "windlifetimemax"
        )
    ]

# --- Settings ---
# Folder containing your MULTIPLE input parquet files
input_parquet_folder = str(args.facilities_dir)

# Folder containing your NetCDF files
nc_folder = str(args.netcdf_dir)

# Resolve the output location without mutating it during a selection query.
output_folder = str(args.output_dir)

# Get file lists
# 1. List all parquet files to process
loc_files = sorted(glob.glob(os.path.join(input_parquet_folder, "*.parquet")))

# 2. List only the NetCDF files that can contribute to the formal step-08 output.
all_nc_files = sorted(glob.glob(os.path.join(nc_folder, "*.nc")))
nc_files = all_nc_files if args.include_nonformal_inputs else select_formal_wind_sources(all_nc_files)
selected_total = len(nc_files)
shard_size, remainder = divmod(selected_total, args.shard_count)
shard_start = args.shard_index * shard_size + min(args.shard_index, remainder)
shard_stop = shard_start + shard_size + int(args.shard_index < remainder)
nc_files = nc_files[shard_start:shard_stop]

print(
    f"Found {len(loc_files)} location files; selected {selected_total} of "
    f"{len(all_nc_files)} NetCDF files; shard {args.shard_index + 1}/"
    f"{args.shard_count} contains {len(nc_files)} files."
)

if args.selection_only:
    raise SystemExit(0)

if Path(output_folder).exists() and any(Path(output_folder).iterdir()) and not args.resume_output:
    raise FileExistsError("Output exists; use the managed --resume/--retry-failed workflow or a new destination")
os.makedirs(output_folder, exist_ok=True)

client = Client(processes=False)
print(client.dashboard_link)


# In[2]:


# Create a "Cache" of location targets
# This is a list of dictionaries containing the xarray Targets for each file
location_targets_cache = []

print("Pre-loading location data into RAM...")

for loc_path in tqdm(loc_files):
    try:
        df_loc = pd.read_parquet(loc_path)

        # Store everything needed for extraction in a lightweight dictionary
        loc_data = {
            "stem": os.path.splitext(os.path.basename(loc_path))[0],  # "my_locs"
            "lats": xr.DataArray(df_loc["Geo_lat"], dims="pt"),
            "lons": xr.DataArray(df_loc["Geo_lon"], dims="pt"),
            "ids": df_loc["ID_internal"].values,
        }
        location_targets_cache.append(loc_data)

    except Exception as e:
        raise RuntimeError(f"Error reading facility source {loc_path}") from e

print(f"Loaded {len(location_targets_cache)} location groups into memory.")


# In[3]:


# --- CELL 3: Bulk Dask Extraction (Reads File Once, Exports Many) ---

def process_storm_netcdf_dask_bulk(nc_path, all_location_targets, out_dir):

    nc_fname = os.path.basename(nc_path)
    nc_stem = os.path.splitext(nc_fname)[0]

    # Open with Dask (Standard lazy loading)
    with xr.open_dataset(nc_path, chunks="auto") as ds:
        # 1. Load Shared Coordinates ONCE (Crucial Optimization)
        time_values = ds["time"].values.astype(str)
        storm_ids = ds["storm"].values
        time_mapper = pd.Series(time_values, index=storm_ids)

        # 2. Build Lazy Tasks for ALL locations
        lazy_tasks = []
        task_metadata = [] # Stores (loc_data, out_path) to match results later

        for loc_data in all_location_targets:
            out_name = f"result_{loc_data['stem']}_{nc_stem}.parquet"
            out_path = os.path.join(out_dir, out_name)

            if os.path.exists(out_path):
                continue

            # Create Lazy Selection (Does not read data yet)
            subset = ds["windlifetimemax"].sel(
                lat=loc_data["lats"],
                lon=loc_data["lons"],
                method="nearest",
            )
            lazy_tasks.append(subset)
            task_metadata.append((loc_data, out_path))

        # 3. Bulk Compute (reads the file once and returns both facility arrays)
        if lazy_tasks:
            computed_results = dask.compute(*lazy_tasks)

            # 4. Process and export individually.
            for subset_values, (loc_data, out_path) in zip(computed_results, task_metadata):
                assert len(storm_ids) != len(loc_data["ids"]), (
                    f"Ambiguous shape: Storms {len(storm_ids)} == Points {len(loc_data['ids'])}"
                )

                if subset_values.shape[0] != len(storm_ids):
                    subset_values = subset_values.T

                df_wide = pd.DataFrame(subset_values, index=storm_ids, columns=loc_data["ids"])
                df_long = df_wide.melt(
                    ignore_index=False,
                    var_name="id_internal",
                    value_name="windspeed",
                ).reset_index()
                df_long.rename(columns={"index": "storm"}, inplace=True)
                df_long = df_long[df_long["windspeed"] > 0].copy()
                df_long["time"] = df_long["storm"].map(time_mapper)
                df_long = df_long.sort_values("windspeed", ascending=False)
                df_long = df_long.drop_duplicates(
                    subset=["id_internal", "time"], keep="first"
                )
                df_long["source_nc"] = nc_fname
                keep_cols = ["id_internal", "source_nc", "time", "windspeed", "storm"]

                # A zero-event year is valid data, not missing data.  Atomic replace
                # makes interrupted shards safely resumable.
                partial_path = f"{out_path}.partial"
                df_long[keep_cols].to_parquet(partial_path, index=False)
                os.replace(partial_path, out_path)


# In[4]:


print("Starting Bulk Dask Extraction...")

for nc_file in tqdm(nc_files):
    process_storm_netcdf_dask_bulk(nc_file, location_targets_cache, output_folder)

print("All processing complete.")
client.close()

