#!/usr/bin/env python
# coding: utf-8

"""Step 2: map Comtrade reporter/partner identifiers to ISO alpha-3 codes."""

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
parser.add_argument("--work-dir", type=Path, default=data_path('work/mfa_trade'))
parser.add_argument("--workbook", type=Path, default=data_path('input/mfa/mfa_inputs.xlsx'))
args = parser.parse_args()

TRADE_WORK_DIR = args.work_dir.resolve()
INPUT_WORKBOOK = args.workbook.resolve()
COUNTRY_MAPPED_PATH = TRADE_WORK_DIR / "2_country_mapped" / "trade_data.parquet"
COUNTRY_MAPPED_PATH.parent.mkdir(parents=True, exist_ok=True)

# In[ ]:


import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
def display(*_objects):
    return None


# In[ ]:


# 辅助函数，获取某一目录下的所有具有某一扩展名的文件路径
def get_file_paths_with_extension(directory, extension):
    import os

    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths


# 辅助函数，读取某一目录下的所有parquet文件，如果column均一致，合并到一个DataFrame
def read_and_merge_parquet_files(directory):
    import pandas as pd
    import os

    file_paths = get_file_paths_with_extension(directory, "parquet")
    dataframes = []
    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()  # 返回空DataFrame

    # 检查所有DataFrame的列是否一致
    first_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        if set(df.columns) != first_columns:
            raise ValueError("Not all parquet files have the same columns")

    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


# In[ ]:


# 读取改正后的贸易数据
trade_df = read_and_merge_parquet_files(str(TRADE_WORK_DIR / "1_corrected"))


# In[ ]:


# 读取表格映射数据
country_mapping_df = pd.read_excel(
    INPUT_WORKBOOK, sheet_name="Comtrade国家编号对照表"
)


# In[ ]:


# 辅助函数，获取某个comtrade_code的iso_3166_alpha_3代号
def get_iso_a3_from_comtrade_code(comtrade_code):
    # 如果need_discard列为Yes，则返回None
    row = country_mapping_df[country_mapping_df["comtrade_code"] == comtrade_code]
    if row.empty:
        raise ValueError(f"comtrade_code {comtrade_code} not found in mapping table")
    elif row.iloc[0]["need_discard"] == "Yes":
        return None

    # 如果corrected_iso_alpha_3列不为空，则返回该列的值
    elif pd.notna(row.iloc[0]["corrected_iso_alpha_3"]):
        return row.iloc[0]["corrected_iso_alpha_3"]

    # 否则返回iso_3166_alpha_3列的值
    else:
        return row.iloc[0]["iso_3166_alpha_3"]


# In[ ]:


# 获取trade_df的所有reporter_comtrade_id和partner_comtrade_id的唯一值
unique_reporter_ids = trade_df["reporter_comtrade_id"].unique()
unique_partner_ids = trade_df["partner_comtrade_id"].unique()
unique_ids = np.unique(np.concatenate((unique_reporter_ids, unique_partner_ids)))
display(unique_ids)

# 为trade_df添加新列，reporter_iso_a3和partner_iso_a3
for comtrade_id in tqdm(unique_ids):
    iso_a3 = get_iso_a3_from_comtrade_code(comtrade_id)
    trade_df.loc[trade_df["reporter_comtrade_id"] == comtrade_id, "reporter_iso_a3"] = (
        iso_a3
    )
    trade_df.loc[trade_df["partner_comtrade_id"] == comtrade_id, "partner_iso_a3"] = (
        iso_a3
    )

# 删除reporter_iso_a3或partner_iso_a3为None的行
trade_df = trade_df[
    trade_df["reporter_iso_a3"].notna() & trade_df["partner_iso_a3"].notna()
]

# 删除reporter_comtrade_id和partner_comtrade_id列
trade_df = trade_df.drop(columns=["reporter_comtrade_id", "partner_comtrade_id"])

# 删除reporter_iso_a3和partner_iso_a3相同的行（即自贸）
trade_df = trade_df[trade_df["reporter_iso_a3"] != trade_df["partner_iso_a3"]]

# 导出数据
trade_df.to_parquet(COUNTRY_MAPPED_PATH, index=False)

