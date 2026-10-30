"""Prepare the extraction-location files from the initial source facility tables.

The original climate-location inputs are byte-identical copies of these two
source basic_info tables. No proprietary attributes are added or redistributed.
"""
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'library'))
from steel_supply.pipeline_paths import data_path


def main():
    output=data_path('work/prepared_climate_facilities')
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    sources={'global_energy_monitor_iron_steel':'global_energy_monitor_iron_steel_basic_info.parquet',
             'spglobal_iron_mine':'spglobal_iron_mine_basic_info.parquet'}
    paths=[(data_path('input/spatialization')/name/'basic_info.parquet',output/dest) for name,dest in sources.items()]
    for source,_ in paths:
        if not source.is_file():raise FileNotFoundError(source)
    output.mkdir(parents=True,exist_ok=True)
    for source,dest in paths:shutil.copy2(source,dest)


if __name__=='__main__':main()
