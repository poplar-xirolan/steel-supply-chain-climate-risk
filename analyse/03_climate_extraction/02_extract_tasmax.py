#!/usr/bin/env python
# coding: utf-8

"""Extract ISIMIP daily tasmax values at the two facility point datasets."""

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
parser.add_argument("--netcdf-dir", type=Path, default=data_path('input/isimip/tasmax'))
parser.add_argument("--output-dir", type=Path, default=data_path('work/climate_at_facilities/tasmax'))
parser.add_argument("--start-year", type=int, default=2018)
parser.add_argument("--end-year", type=int, default=2100)
parser.add_argument("--scenarios", nargs="+", default=["picontrol", "ssp126", "ssp370", "ssp585"])
parser.add_argument("--shard-count", type=int, default=1)
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--resume-output", action="store_true")
args = parser.parse_args()
if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
    parser.error("--shard-index must be in [0, --shard-count)")

# 读取`projects/2_research/个人研究项目/高时空分辨率金属风险影响范围、路径研究与对策分析/子部分 MFA网络空间化/2_数据整理/2_规范范式整理/SCHEMA.md`处定义的 `basic_info` 范式的 parquet 数据，根据坐标信息，获取`./1.1_heatwave_download`处的 isimip3b 的 tasmax 变量，并存储为 parquet 文件。
# 

# In[3]:


import pandas as pd
import xarray as xr
import os
import glob
from tqdm import tqdm

# --- Settings ---
# Folder containing your MULTIPLE input parquet files
input_parquet_folder = str(args.facilities_dir)

# Folder containing your NetCDF files (tasmax)
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


# In[4]:


# Create a "Cache" of location targets
# This is a list of dictionaries containing the xarray Targets for each file
location_targets_cache = []

print("Pre-loading location data into RAM...")

for loc_path in tqdm(loc_files):
    try:
        df_loc = pd.read_parquet(loc_path)
        
        # Store everything needed for extraction in a lightweight dictionary
        loc_data = {
            "stem": os.path.splitext(os.path.basename(loc_path))[0], # "my_locs"
            "lats": xr.DataArray(df_loc["Geo_lat"], dims="pt"),
            "lons": xr.DataArray(df_loc["Geo_lon"], dims="pt"),
            "ids":  df_loc["ID_internal"].values
        }
        location_targets_cache.append(loc_data)
        
    except Exception as e:
        raise RuntimeError(f"Error reading facility source {loc_path}") from e

print(f"Loaded {len(location_targets_cache)} location groups into memory.")


# In[5]:


# --- CELL 3: The Optimized Extraction Function (Matrix Strategy) ---

def process_netcdf_centric(nc_path, all_location_targets, out_dir):
    """
    Reads ONE NetCDF file, and extracts data for ALL location groups.
    Optimized: Uses Matrix->Melt strategy instead of .to_dataframe()
    """
    
    nc_fname = os.path.basename(nc_path)
    nc_stem = os.path.splitext(nc_fname)[0]
    
    # 1. Open the NetCDF lazily.  Loading the complete global grid before
    # selecting facility points is both slower and much more memory hungry.
    with xr.open_dataset(nc_path, chunks="auto") as ds:
        da = ds["tasmax"]
        
        # Optimization: Pre-extract time as string array ONCE
        # This prevents converting it 1000 times inside the loop
        time_index = da['time'].astype(str).values

        # 2. Inner Loop: Iterate through cached location groups
        for loc_data in all_location_targets:
            
            out_name = f"result_{loc_data['stem']}_{nc_stem}.parquet"
            out_path = os.path.join(out_dir, out_name)
            
            # Skip if output already exists
            if os.path.exists(out_path):
                continue
            
            # --- Extract (Instant) ---
            subset = da.sel(lat=loc_data['lats'], lon=loc_data['lons'], method="nearest")
            
            # --- SAFETY LOCK: FORCE DIMENSION ORDER ---
            # IMPORTANT: Ensure shape is strictly (Time, Points) before converting to numpy.
            # If we skip this, some NetCDF files might give us (Points, Time) 
            # which would scramble your data completely.
            # We assume your location array dim is named "pt" (from Cell 2).
            subset = subset.transpose("time", "pt").load()
            
            # --- THE OPTIMIZATION (Matrix -> Melt) ---
            
            # 1. Get raw numpy matrix (Rows=Time, Cols=Points)
            raw_values = subset.values 
            
            # 2. Create a "Wide" DataFrame directly
            # Index = Time, Columns = ID_internal
            df_wide = pd.DataFrame(
                raw_values, 
                index=time_index, 
                columns=loc_data['ids']
            )
            
            # 3. "Melt" to Long Format (Rows = Time * Points)
            # This bypasses the slow xarray MultiIndex creation
            df_chunk = df_wide.melt(
                ignore_index=False,      # Keep the 'time' index
                var_name="id_internal", 
                value_name="tasmax"
            ).reset_index()              # Move 'time' index to column
            
            # Rename 'index' (which held the time) to 'time'
            df_chunk.rename(columns={'index': 'time'}, inplace=True)
            
            # --------------------------------
            
            # Add static source column
            df_chunk["source_nc"] = nc_fname
            
            # Save
            keep_cols = ["id_internal", "source_nc", "time", "tasmax"]
            df_chunk[keep_cols].to_parquet(out_path)
        
        # Clean up NetCDF from RAM
        del da


# In[6]:


print("Starting Inverted Extraction Loop...")

# We loop through NetCDF files (the heavy files)
# Inside this function, it loops through the locations
for nc_file in tqdm(nc_files):
    process_netcdf_centric(nc_file, location_targets_cache, output_folder)

print("All processing complete.")

