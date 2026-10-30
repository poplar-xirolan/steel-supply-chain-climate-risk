#!/usr/bin/env python
# coding: utf-8

"""Extract ISIMIP daily relative humidity (hurs) at facility locations."""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
import re
from pathlib import Path

ROOT = bundle_root()
parser = argparse.ArgumentParser()
parser.add_argument("--facilities-dir", type=Path, default=data_path('input/climate_facilities'))
parser.add_argument("--netcdf-dir", type=Path, default=data_path('input/isimip/hurs'))
parser.add_argument("--output-dir", type=Path, default=data_path('work/climate_at_facilities/hurs'))
parser.add_argument("--start-year", type=int, default=2018)
parser.add_argument("--end-year", type=int, default=2100)
parser.add_argument("--scenarios", nargs="+", default=["picontrol", "ssp126", "ssp370", "ssp585"])
parser.add_argument("--shard-count", type=int, default=1)
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--resume-output", action="store_true")
args = parser.parse_args()
if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
    parser.error("--shard-index must be in [0, --shard-count)")

# 读取`projects/2_research/个人研究项目/高时空分辨率金属风险影响范围、路径研究与对策分析/子部分 MFA网络空间化/2_数据整理/2_规范范式整理/SCHEMA.md`处定义的 `basic_info` 范式的 parquet 数据，根据坐标信息，获取`./1.5_humid_download`处的 isimip3b 的 hurs 变量，并存储为 parquet 文件。
# 

# In[1]:


import pandas as pd
import xarray as xr
import os
import glob
import dask
from dask.distributed import Client
from dask import delayed
from tqdm import tqdm

# --- Settings ---
# Folder containing your MULTIPLE input parquet files
input_parquet_folder = str(args.facilities_dir)

# Folder containing your NetCDF files (hurs)
nc_folder = str(args.netcdf_dir)

# Folder to save the results. Remove if exists.
output_folder = str(args.output_dir)
if Path(output_folder).exists() and any(Path(output_folder).iterdir()) and not args.resume_output:
    raise FileExistsError("Output exists; use the managed --resume/--retry-failed workflow or a new destination")
os.makedirs(output_folder, exist_ok=True)

# Get file lists
# 1. List all parquet files to process
loc_files = sorted(glob.glob(os.path.join(input_parquet_folder, "*.parquet")))

# 2. List all NetCDF files (Climate Data)
def required_climate_file(path):
    stem = Path(path).stem
    parts = stem.split("_")
    year_match = re.search(r"_(\d{4})_(\d{4})$", stem)
    if len(parts) <= 3 or year_match is None:
        return False
    start_year, end_year = map(int, year_match.groups())
    return (
        parts[3] in set(args.scenarios)
        and end_year >= args.start_year
        and start_year <= args.end_year
    )


nc_files = [
    path
    for path in sorted(glob.glob(os.path.join(nc_folder, "*.nc")))
    if required_climate_file(path)
]
shard_size, remainder = divmod(len(nc_files), args.shard_count)
shard_start = args.shard_index * shard_size + min(args.shard_index, remainder)
shard_stop = shard_start + shard_size + int(args.shard_index < remainder)
nc_files = nc_files[shard_start:shard_stop]

print(f"Found {len(loc_files)} location files and {len(nc_files)} NetCDF files.")

client = Client(processes=False)
print(client.dashboard_link)


# In[2]:


# Create a "Cache" of location targets
location_targets_cache = []

print("Pre-loading location data into RAM...")

for loc_path in tqdm(loc_files):
    try:
        df_loc = pd.read_parquet(loc_path)

        # Store everything needed for extraction in a lightweight dictionary
        loc_data = {
            "stem": os.path.splitext(os.path.basename(loc_path))[0], 
            "lats": xr.DataArray(df_loc["Geo_lat"], dims="pt"),
            "lons": xr.DataArray(df_loc["Geo_lon"], dims="pt"),
            "ids": df_loc["ID_internal"].values,
        }
        location_targets_cache.append(loc_data)

    except Exception as e:
        raise RuntimeError(f"Error reading facility source {loc_path}") from e

print(f"Loaded {len(location_targets_cache)} location groups into memory.")

# --- OPTIMIZATION: Scatter data to workers ---
# This moves the location data to the cluster memory once, 
# preventing us from re-sending it for every single task.
loc_future = client.scatter(location_targets_cache, broadcast=True)
print("Location data scattered to workers.")


# In[3]:


def process_netcdf_centric(nc_path, all_location_targets, out_dir):
    """
    Reads ONE NetCDF file.
    Optimized: Batches ALL location extractions into a single dask.compute() call 
    to ensure shared chunks are read from disk only once.
    """
    try:
        nc_fname = os.path.basename(nc_path)
        nc_stem = os.path.splitext(nc_fname)[0]
        target_var = "hurs"

        # 1. Open Lazily with Dask
        with xr.open_dataset(nc_path, chunks="auto") as ds:
            
            if target_var not in ds:
                raise ValueError(f"Required variable {target_var} not found in {nc_fname}")

            da = ds[target_var]

            # Load time index once (small)
            time_index = da["time"].astype(str).compute().values

            # --- PREPARE LAZY BATCH ---
            lazy_results = []
            meta_data_list = []

            for loc_data in all_location_targets:
                out_name = f"result_{loc_data['stem']}_{nc_stem}.parquet"
                out_path = os.path.join(out_dir, out_name)

                # Skip existing but do not break, we need to maintain index alignment 
                # if we were strictly mapping. But here we just won't add it to the batch.
                if os.path.exists(out_path):
                    continue

                # Lazy Selection (No I/O yet)
                subset = da.sel(lat=loc_data["lats"], lon=loc_data["lons"], method="nearest")
                
                # Force dimensions (Time, Pt)
                subset = subset.transpose("time", "pt")
                
                # Append the Dask Array to our batch list
                lazy_results.append(subset)
                
                # Store metadata so we know who this result belongs to later
                meta_data_list.append({
                    "ids": loc_data["ids"],
                    "out_path": out_path
                })

            # If nothing to do, exit
            if not lazy_results:
                return f"Skipped {nc_fname} (All exist)"

            # --- THE CRITICAL STEP: BATCH COMPUTE ---
            # We pass the list of dask arrays to compute().
            # Dask sees the shared ancestry and optimizes the read.
            # computed_arrays is a tuple of Numpy Arrays.
            computed_arrays = dask.compute(lazy_results)[0]

            # --- POST-PROCESS & SAVE ---
            # Now we have the data in RAM, we just format and save.
            for i, raw_values in enumerate(computed_arrays):
                meta = meta_data_list[i]
                
                # Create DataFrame
                df_wide = pd.DataFrame(
                    raw_values, # This is now a numpy array
                    index=time_index, 
                    columns=meta["ids"]
                )

                # Melt
                df_chunk = df_wide.melt(
                    ignore_index=False, 
                    var_name="id_internal", 
                    value_name=target_var
                ).reset_index()

                df_chunk.rename(columns={"index": "time"}, inplace=True)
                df_chunk["source_nc"] = nc_fname

                # Save
                keep_cols = ["id_internal", "source_nc", "time", target_var]
                df_chunk[keep_cols].to_parquet(meta["out_path"])

        return f"Done {nc_fname} ({len(lazy_results)} loc groups)"
    
    except Exception as e:
        raise RuntimeError(f"Extraction failed for {nc_path}") from e


# In[4]:


print("Starting Sequential Processing (One file at a time)...")

# We loop through files sequentially to avoid "Disk Thrashing"
# This is often FASTER than parallel processing for large NetCDF files
# because it allows the hard drive to read continuously without jumping around.

results = []

for nc_file in tqdm(nc_files):
    # 1. Call the function DIRECTLY (No delayed, no scatter)
    # We use 'location_targets_cache' (the local list) instead of 'loc_future'
    res = process_netcdf_centric(nc_file, location_targets_cache, output_folder)
    
    results.append(res)
    
    # Optional: Print error immediately if it occurs
    if "Error" in res:
        print(f"Warning: {res}")

print("\nAll processing complete.")

# Summary of any errors
errors = [r for r in results if "Error" in r]
if errors:
    print(f"Encountered {len(errors)} errors.")
else:
    print("No errors reported.")

