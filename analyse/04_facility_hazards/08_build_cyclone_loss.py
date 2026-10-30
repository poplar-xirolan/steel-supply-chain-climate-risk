#!/usr/bin/env python
# coding: utf-8

"""Build tropical-cyclone disruption windows from extracted wind fields."""

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
parser.add_argument("--input-dir", type=Path, default=data_path('work/climate_at_facilities/wind'))
parser.add_argument("--output-dir", type=Path, default=data_path('work/hazard_facility_loss/tropical_cyclone_loss'))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

# - 解析文件列表
# - 筛选符合年份序列要求的数据（picontrol和ssp370、ssp585情景下，要求数据覆盖2015-2100年连续年份；ssp126下，要求数据覆盖2061-2100年连续年份）
# - 指定生产点位数据源，根据窗口期大小与定义，生成长时间影响序列后进行平均，最终为每一年，导出各天、各生产点的production_loss

# In[14]:


import pandas as pd
from pathlib import Path
def display(*_objects):
    return None
from tqdm import tqdm

# 解析文件列表
def parse_isimip_wind_filenames_to_df(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' not found.")
        return pd.DataFrame()

    custom_prefixes = [
        "result_global_energy_monitor_iron_steel_basic_info_", 
        "result_spglobal_iron_mine_basic_info_"
    ]

    data_rows = []
    files = list(folder.glob("*.parquet"))

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
        
        # Structure: er11_mit_ipsl-cm6a-lr_historical_windlifetimemax_1940
        # parts[0]: er11 (Event Set ID)
        # parts[1]: mit  (Institution/Generator)
        # parts[2]: ipsl-cm6a-lr (GCM Model)
        # parts[3]: historical (Scenario)
        # parts[4]: windlifetimemax (Variable)
        # parts[5]: 1940 (Year)
        
        parts = remainder.split("_")
        
        # Ensure we have enough parts to avoid IndexError
        if len(parts) >= 6:
            row_data = {
                "raw_filename": raw_filename,
                "filepath": str(file_path.resolve()),
                "dataset_source": dataset_source_extracted,
                
                # Mapping specific to the wind/cyclone file structure
                "model": parts[2],                          # e.g., ipsl-cm6a-lr
                "forcing_member": f"{parts[0]}_{parts[1]}", # e.g., er11_mit
                "climate_scenario": parts[3],               # e.g., historical
                "variable": parts[4],                       # e.g., windlifetimemax
                "start_year": parts[5],                     # e.g., 1940
                "end_year": parts[5],                       # Same as start for annual files
                
                # Fields not present in this specific filename format:
                "bias_adjustment": None,
                "region": None,
                "time_step": "annual"                       # Inferring annual from single year suffix
            }
            data_rows.append(row_data)
        else:
            print(f"Skipping malformed file: {raw_filename}")

    return pd.DataFrame(data_rows)

# 筛选符合年份序列要求的数据（picontrol和ssp370、ssp585情景下，要求数据覆盖2015-2100年连续年份；ssp126下，要求数据覆盖2061-2100年连续年份）
def filter_valid_combinations(parsed_df, target_year, climate_scenario):
    """
    Selects model and forcing_member combinations that completely cover the target year sequence.
    
    Args:
        folder_path (str): Path to the folder containing the ISIMIP wind data.
        target_year (list or range): Sequence of years required (e.g., range(2015, 2101)).
        climate_scenario (str): The climate scenario to filter (e.g., 'ssp126', 'historical').

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the models/forcing members 
                      that have data for every year in target_year.
    """
    
    # 1. Get the initial DataFrame using your parsing function
    # Note: Ensure 'parse_isimip_wind_filenames_to_df' is defined/imported
    df = parsed_df
    
    # 2. Filter by the requested climate scenario first
    scenario_df = df[df['climate_scenario'] == climate_scenario].copy()
    
    # 3. Ensure the year column is numeric for comparison
    # We use 'start_year' as the representative year (since start_year == end_year in annual files)
    scenario_df['year'] = pd.to_numeric(scenario_df['start_year'])
    scenario_df = scenario_df.dropna(subset=['year']) # Drop rows where year is invalid
    scenario_df['year'] = scenario_df['year'].astype(int)
    
    # 4. Define the required years as a set
    required_years = set(target_year)
    
    # 5. Group by model and forcing_member to check coverage
    # We identify which (model, forcing_member) pairs contain the full set of target years
    valid_combinations = []
    
    # Grouping gives us the subset of data for each unique simulation
    groups = scenario_df.groupby(['model', 'forcing_member'])
    
    for (model, forcing), group_data in groups:
        available_years = set(group_data['year'])
        
        # Check if the required years are a SUBSET of the available years
        if required_years.issubset(available_years):
            valid_combinations.append((model, forcing))
            
    # 6. Filter the DataFrame to return only the valid simulations
    if not valid_combinations:
        print(f"No models found in '{climate_scenario}' covering the full range {min(required_years)}-{max(required_years)}.")
        return pd.DataFrame(columns=df.columns) 
    
    # Create a mask to keep only rows belonging to valid (model, forcing_member) pairs
    # We use a MultiIndex match for efficiency
    mask = scenario_df.set_index(['model', 'forcing_member']).index.isin(valid_combinations)
    result_df = scenario_df[mask].reset_index(drop=True)
    
    # Optional: Sort by model and year for cleaner viewing
    result_df = result_df.sort_values(by=['model', 'forcing_member', 'year']).reset_index(drop=True)
    
    return result_df

# 指定生产点位数据源，根据窗口期大小与定义，生成长时间影响序列后进行平均，最终为每一年，导出各天、各生产点的production_loss
def generate_production_loss_timeseries(filtered_df, data_source, full_year_range, climate_scenario, export_folder):
    """
    Generates impact windows for tropical cyclones, averages losses across models,
    and exports annual production loss files.
    """
    export_path = Path(export_folder)

    # 1. Filter for the specific data source and scenario
    source_scenario_df = filtered_df[
        (filtered_df['dataset_source'] == data_source) & 
        (filtered_df['climate_scenario'] == climate_scenario)
    ].copy()

    if source_scenario_df.empty:
        raise ValueError(f"No data found for source: {data_source} and scenario: {climate_scenario}")

    # 2. Identify unique model/forcing combinations to iterate over
    all_models = source_scenario_df[['model', 'forcing_member']].drop_duplicates().copy()
    num_models = len(all_models)
    print(f"Processing {num_models} model combination(s) for aggregation.")

    model_loss_dfs = []

    # 3. Iterate through each model combination
    # Wrap iterrows with tqdm for a progress bar
    for idx, row in tqdm(all_models.iterrows(), total=num_models, desc="Processing Models"):
        model = row['model']
        forcing_member = row['forcing_member']
        
        # Get all files for this specific model simulation
        full_ts_data_df = source_scenario_df[
            (source_scenario_df['model'] == model) & 
            (source_scenario_df['forcing_member'] == forcing_member)
        ]
        
        # Basic validation to ensure annual files are consistent
        assert (full_ts_data_df['start_year'] == full_ts_data_df['end_year']).all()

        # Load all parquet files for this model into one dataframe
        data_frames = []
        for _, data_row in full_ts_data_df.iterrows():
            file_path = data_row['filepath']
            try:
                df = pd.read_parquet(file_path)
                data_frames.append(df)
            except Exception as e:
                raise RuntimeError(f"Cannot read selected cyclone input {file_path}") from e
        
        if not data_frames:
            continue
            
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # Filter for tropical cyclone threshold (≥ 64 knots as defined in previous cells)
        # Using the threshold value from your provided code snippet: 32.9244 m/s (approx 64 knots)
        cyclone_events_df = combined_df[combined_df['windspeed'] >= 32.9244].copy()
        
        if cyclone_events_df.empty:
            # If no cyclones hit this model, skip impact window generation
            continue

        # --- PROCEDURE 1: Define 31-day Impact Window ---
        
        # Ensure 'time' is datetime object for date arithmetic
        cyclone_events_df['event_time_dt'] = pd.to_datetime(cyclone_events_df['time']).dt.normalize()
        
        # Create a DataFrame representing the +/- 15 day offsets
        offsets = [pd.Timedelta(i, unit="days") for i in range(-15, 16)]
        offset_df = pd.DataFrame({'offset': offsets})
        
        # Perform a cross-join to expand every cyclone event row into 31 rows (one for each offset day)
        # Note: 'how="cross"' requires pandas >= 1.2.0
        expanded_df = cyclone_events_df.merge(offset_df, how='cross')
        
        # Calculate the actual impact date by adding the offset to the event date
        expanded_df['impact_date'] = expanded_df['event_time_dt'] + expanded_df['offset']
        
        # Select relevant columns: id_internal and the calculated impact date
        daily_loss_df = expanded_df[['id_internal', 'impact_date']].copy()
        
        # Deduplicate: If overlapping windows affect the same day, it still counts as just 1 loss day.
        daily_loss_df = daily_loss_df.drop_duplicates()
        
        # Set production loss to 1 (complete loss) for these days
        daily_loss_df['production_loss'] = 1.0
        
        model_loss_dfs.append(daily_loss_df)

    # --- PROCEDURE 2: Average over all models ---
    print("Aggregating and averaging across models...")
    
    if not model_loss_dfs:
         print("No cyclone events detected across any models matching criteria.")
         # Create an empty placeholder if needed, or just return.
         total_loss_df = pd.DataFrame(columns=['id_internal', 'impact_date', 'avg_production_loss'])
    else:
        # Concatenate all individual model loss dataframes
        concat_loss_df = pd.concat(model_loss_dfs, ignore_index=True)
        
        # Group by site (id_internal) and date. Sum the losses (1s) and divide by total number of models.
        # This gives the average loss ratio across the ensemble.
        total_loss_df = concat_loss_df.groupby(['id_internal', 'impact_date'])['production_loss'].sum().reset_index()
        total_loss_df.rename(columns={'production_loss': 'sum_loss'}, inplace=True)
        total_loss_df['avg_production_loss'] = total_loss_df['sum_loss'] / num_models

    # Ensure impact_date is datetime for subsequent filtering
    total_loss_df['impact_date'] = pd.to_datetime(total_loss_df['impact_date'])

    # --- PROCEDURE 3: Export annual files ---
    print(f"Starting export to {export_path}...")
    
    target_years = sorted(list(full_year_range))
    
    for year in tqdm(target_years, desc="Exporting Years"):
        # Filter the aggregated data for the current year
        year_df = total_loss_df[total_loss_df['impact_date'].dt.year == year].copy()
            
        # Calculate 'day_of_year' to match the target format         
        year_df['day_of_year'] = year_df['impact_date'].dt.dayofyear
        
        # Select and reorder columns for final output
        final_output_df = year_df[['id_internal', 'day_of_year', 'avg_production_loss']].copy()
        
        # Rename the loss column to match the implied format if necessary (e.g., 'gross_production_loss_fr...')
        # Using a generic name based on the prompt for now.
        final_output_df.rename(columns={'avg_production_loss': 'production_loss_fraction'}, inplace=True)

        # Construct filename
        filename = f"production_loss_{data_source.replace("_", "-")}_{year}_{climate_scenario}.parquet"
        full_export_path = export_path / filename
        
        # Export to parquet
        final_output_df.to_parquet(full_export_path, index=False)

    print("Processing complete.")


# 年份附注
# 
# - picontrol、ssp370、ssp585的年份均覆盖2015-2100的全部年份（但是请注意，部分模型没有覆盖全部年份）；
# - ssp245只有2061-2100年份的数据。

# In[15]:


# --- Configuration ---
input_folder = str(args.input_dir)
export_folder = str(args.output_dir)

# Define the two data sources
data_sources = [
    "result_global_energy_monitor_iron_steel_basic_info",
    "result_spglobal_iron_mine_basic_info"
]

# Define the scenarios to iterate over
climate_scenarios = ["picontrol", "ssp126", "ssp370", "ssp585"]

# --- Step 1: Parse all filenames once ---
# It is more efficient to parse the folder once and filter later
print("Parsing file list...")
all_files_df = parse_isimip_wind_filenames_to_df(input_folder)

if not all_files_df.empty:
    # --- Step 2: Iterate over Sources and Scenarios ---
    for source in data_sources:
        for scenario in climate_scenarios:
            print(f"\n{'='*60}")
            print(f"Processing Source: {source} | Scenario: {scenario}")
            print(f"{'='*60}")
            
            # 1. Determine the correct year range based on your rules
            if scenario == "ssp126":
                target_year_range = range(2061, 2101) # 2061-2100
                print(f"-> Time span set to: 2061-2100")
            else:
                # picontrol, ssp370, ssp585
                target_year_range = range(2015, 2101) # 2015-2100
                print(f"-> Time span set to: 2015-2100")

            # 2. Filter valid model combinations for this specific scenario
            # Note: We pass the FULL parsed dataframe here; the function handles filtering
            valid_models_df = filter_valid_combinations(
                all_files_df[all_files_df['dataset_source'] == source],
                target_year_range, 
                scenario
            )

            if valid_models_df.empty:
                raise ValueError(f"No complete cyclone model combination for {source}, {scenario}")

            # 3. Generate and export the time series
            try:
                generate_production_loss_timeseries(
                    filtered_df=valid_models_df,
                    data_source=source,
                    full_year_range=target_year_range,
                    climate_scenario=scenario,
                    export_folder=export_folder
                )
            except Exception as e:
                raise RuntimeError(f"Error processing {source} - {scenario}") from e

    print("\nAll tasks completed.")
else:
    raise ValueError("No cyclone files were parsed; check the input folder path")

