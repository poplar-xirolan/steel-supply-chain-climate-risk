#!/usr/bin/env python
# coding: utf-8

"""Build daily heatwave labor-productivity losses for 2018-2100."""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
from pathlib import Path

ROOT = bundle_root()
parser = argparse.ArgumentParser()
parser.add_argument("--start-year", type=int, default=2018)
parser.add_argument("--end-year", type=int, default=2100)
parser.add_argument("--tas-dir", type=Path, default=data_path('work/climate_at_facilities/tas'))
parser.add_argument("--tasmax-dir", type=Path, default=data_path('work/climate_at_facilities/tasmax'))
parser.add_argument("--hurs-dir", type=Path, default=data_path('work/climate_at_facilities/hurs'))
parser.add_argument("--output-dir", type=Path, default=data_path('work/hazard_facility_loss/heatwave_gross_loss'))
parser.add_argument("--shard-count", type=int, default=1)
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--resume-output", action="store_true")
parser.add_argument(
    "--selection-only",
    action="store_true",
    help="Report the selected year shard and exit before calculating losses.",
)
args = parser.parse_args()
if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
    parser.error("--shard-index must be in [0, --shard-count)")
args.output_dir.mkdir(parents=True, exist_ok=True)

# # 数据检查部分
# 

# In[9]:


import pandas as pd
import numpy as np
from pathlib import Path
from scipy.special import erf


def heatwave_production_loss_extract(
    target_year: int, data_source: str, scenario: str, output_parquet_path: str, tas_folder: str, tasmax_folder: str, hurs_folder: str
):
    """
    Extracts ISIMIP climate data for a specific Year/Source/Scenario, calculates
    labor productivity loss (WBGT), and exports to Parquet.
    """

    # --- 1. Helper Functions (Nested to keep scope clean) ---

    def parse_isimip_filenames_to_df(folder_path):
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Error: Folder '{folder_path}' not found.")
            return pd.DataFrame()

        custom_prefixes = ["result_global_energy_monitor_iron_steel_basic_info_", "result_spglobal_iron_mine_basic_info_"]

        data_rows = []
        files = list(folder.glob("*.parquet"))
        # print(f"Found {len(files)} files in {folder.name}. Parsing...")

        for file_path in files:
            raw_filename = file_path.name
            clean_name = raw_filename.replace(".parquet", "")

            dataset_source_extracted = "Unknown"
            remainder = clean_name

            for prefix in custom_prefixes:
                if clean_name.startswith(prefix):
                    dataset_source_extracted = prefix.rstrip("_")
                    remainder = clean_name.replace(prefix, "")
                    break

            parts = remainder.split("_")

            row_data = {
                "raw_filename": raw_filename,
                "filepath": str(file_path.resolve()),
                "dataset_source": dataset_source_extracted,
                "model": parts[0] if len(parts) > 0 else None,
                "forcing_member": parts[1] if len(parts) > 1 else None,
                "bias_adjustment": parts[2] if len(parts) > 2 else None,
                "climate_scenario": parts[3] if len(parts) > 3 else None,
                "variable": parts[4] if len(parts) > 4 else None,
                "region": parts[5] if len(parts) > 5 else None,
                "time_step": parts[6] if len(parts) > 6 else None,
                "start_year": parts[7] if len(parts) > 7 else None,
                "end_year": parts[8] if len(parts) > 8 else None,
            }
            data_rows.append(row_data)

        return pd.DataFrame(data_rows)

    def get_single_year_data(row, variable_name, target_year):
        filepath = row["filepath"]
        try:
            df = pd.read_parquet(filepath)
            df["dt"] = pd.to_datetime(df["time"])
            df_year = df[df["dt"].dt.year == target_year].copy()

            if df_year.empty:
                raise ValueError(f"Selected input has no observations for {target_year}: {filepath}")

            df_year["day_of_year"] = df_year["dt"].dt.dayofyear
            return df_year[["id_internal", "day_of_year", variable_name]]
        except Exception as e:
            raise RuntimeError(f"Cannot read selected heatwave input {filepath}") from e

    def calculate_ensemble_mean(metadata_df, variable_name, target_year):
        data_from_all_models = []
        print(f"  > Processing {variable_name} for {target_year}...")

        for index, row in metadata_df.iterrows():
            model_df = get_single_year_data(row, variable_name, target_year)
            if model_df is not None:
                data_from_all_models.append(model_df)

        if not data_from_all_models:
            return pd.DataFrame()

        combined_df = pd.concat(data_from_all_models, ignore_index=True)
        ensemble_mean_df = combined_df.groupby(["id_internal", "day_of_year"])[variable_name].mean().reset_index()
        ensemble_mean_df = ensemble_mean_df.sort_values(by=["id_internal", "day_of_year"])

        return ensemble_mean_df

    def calc_loss(wbgt_val):
        """Calculates loss fraction L based on the S-curve formula."""
        mu = 32.47
        sigma = 4.16
        return 0.5 * (1 + erf((wbgt_val - mu) / (sigma * np.sqrt(2))))

    # --- 2. Main Execution Logic ---

    print(f"--- Starting Extraction: {target_year} | {scenario} | {data_source} ---")

    # Parse directories
    tas_param_df = parse_isimip_filenames_to_df(tas_folder)
    tasmax_param_df = parse_isimip_filenames_to_df(tasmax_folder)
    hurs_param_df = parse_isimip_filenames_to_df(hurs_folder)

    if tas_param_df.empty or tasmax_param_df.empty or hurs_param_df.empty:
        raise ValueError("One or more required heatwave input folders yielded no data")

    # Filter by Year
    tas_sel = tas_param_df[(tas_param_df["start_year"].astype(int) <= target_year) & (tas_param_df["end_year"].astype(int) >= target_year)]
    tasmax_sel = tasmax_param_df[(tasmax_param_df["start_year"].astype(int) <= target_year) & (tasmax_param_df["end_year"].astype(int) >= target_year)]
    hurs_sel = hurs_param_df[(hurs_param_df["start_year"].astype(int) <= target_year) & (hurs_param_df["end_year"].astype(int) >= target_year)]

    # Filter by Scenario
    tas_sel = tas_sel[tas_sel["climate_scenario"] == scenario]
    tasmax_sel = tasmax_sel[tasmax_sel["climate_scenario"] == scenario]
    hurs_sel = hurs_sel[hurs_sel["climate_scenario"] == scenario]

    # Filter by Data Source
    tas_sel = tas_sel[tas_sel["dataset_source"] == data_source]
    tasmax_sel = tasmax_sel[tasmax_sel["dataset_source"] == data_source]
    hurs_sel = hurs_sel[hurs_sel["dataset_source"] == data_source]

    # Find Common Models
    common_models = set(tas_sel["model"]).intersection(set(tasmax_sel["model"])).intersection(set(hurs_sel["model"]))

    if not common_models:
        raise ValueError(f"No common models found for {scenario} in {target_year}, source {data_source}")

    print(f"Found {len(common_models)} common models: {common_models}")

    tas_sel = tas_sel[tas_sel["model"].isin(common_models)]
    tasmax_sel = tasmax_sel[tasmax_sel["model"].isin(common_models)]
    hurs_sel = hurs_sel[hurs_sel["model"].isin(common_models)]

    # Calculate Ensembles
    tas_ensemble = calculate_ensemble_mean(tas_sel, "tas", target_year)
    tasmax_ensemble = calculate_ensemble_mean(tasmax_sel, "tasmax", target_year)
    hurs_ensemble = calculate_ensemble_mean(hurs_sel, "hurs", target_year)

    if tas_ensemble.empty or tasmax_ensemble.empty or hurs_ensemble.empty:
        raise ValueError("Heatwave ensemble calculation resulted in empty data")

    # Merge
    final_risk_df = hurs_ensemble.merge(tas_ensemble, on=["id_internal", "day_of_year"], how="inner").merge(
        tasmax_ensemble, on=["id_internal", "day_of_year"], how="inner"
    )

    # --- 3. Physical Calculations ---

    # Unit Conversion (Kelvin -> Celsius)
    T_mean_C = final_risk_df["tas"] - 273.15
    T_max_C = final_risk_df["tasmax"] - 273.15
    RH_mean = final_risk_df["hurs"]

    # Water Vapor Pressure (E)
    E = (RH_mean / 100) * 6.105 * np.exp((17.27 * T_mean_C) / (237.7 + T_mean_C))

    # WBGT
    wbgt_mean = 0.567 * T_mean_C + 0.393 * E + 3.94
    wbgt_max = 0.567 * T_max_C + 0.393 * E + 3.94
    wbgt_half = (wbgt_max + wbgt_mean) / 2

    # Production Loss (L)
    L_max = calc_loss(wbgt_max)
    L_half = calc_loss(wbgt_half)
    L_mean = calc_loss(wbgt_mean)

    final_risk_df["gross_production_loss_fraction"] = (L_max + L_half + L_mean) / 3

    # Add metadata columns for the parquet file
    final_risk_df["year"] = target_year
    final_risk_df["scenario"] = scenario
    final_risk_df["source"] = data_source

    # --- 4. Export ---
    print(f"Exporting to {output_parquet_path}...")
    final_risk_df.to_parquet(output_parquet_path, index=False)
    print("Done.")


# In[ ]:


from tqdm import tqdm


DIRS = {
    "tas": str(args.tas_dir),
    "tasmax": str(args.tasmax_dir),
    "hurs": str(args.hurs_dir),
}

all_years = list(range(args.start_year, args.end_year + 1))
shard_size, remainder = divmod(len(all_years), args.shard_count)
shard_start = args.shard_index * shard_size + min(args.shard_index, remainder)
shard_stop = shard_start + shard_size + int(args.shard_index < remainder)
years = all_years[shard_start:shard_stop]
print(
    f"Selected {len(years)} of {len(all_years)} years for shard "
    f"{args.shard_index + 1}/{args.shard_count}: "
    f"{years[0] if years else 'EMPTY'}..{years[-1] if years else 'EMPTY'}",
    flush=True,
)
if args.selection_only:
    raise SystemExit(0)

for year in tqdm(years):
    for scenario in ["picontrol", "ssp126", "ssp370", "ssp585"]:
        for data_source in ["result_spglobal_iron_mine_basic_info", "result_global_energy_monitor_iron_steel_basic_info"]:
            output_path = args.output_dir / f"production_loss_{data_source.replace('_', '-')}_{year}_{scenario}.parquet"
            if args.resume_output and output_path.exists():
                continue
            heatwave_production_loss_extract(
                target_year=year,
                data_source=data_source,
                scenario=scenario,
                output_parquet_path=str(output_path),
                tas_folder=DIRS["tas"],
                tasmax_folder=DIRS["tasmax"],
                hurs_folder=DIRS["hurs"],
            )

