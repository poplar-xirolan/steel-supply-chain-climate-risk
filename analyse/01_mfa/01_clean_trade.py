#!/usr/bin/env python
# coding: utf-8

"""Step 1: clean and interpolate the raw UN Comtrade tables.

The numerical procedure is unchanged from the final research notebook. Only
paths were moved to the public repository layout.
"""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
from pathlib import Path
import sys

ROOT = bundle_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steel_supply.mfa.trade_peak_filter import correct_single_and_double_peaks

parser = argparse.ArgumentParser()
parser.add_argument("--raw-data-dir", type=Path, default=data_path('input/mfa/trade_raw'))
parser.add_argument("--work-dir", type=Path, default=data_path('work/mfa_trade'))
args = parser.parse_args()

RAW_DATA_DIR = args.raw_data_dir.resolve()
TRADE_WORK_DIR = args.work_dir.resolve()
if not RAW_DATA_DIR.exists():
    raise FileNotFoundError(
        f"Raw Comtrade directory not found: {RAW_DATA_DIR}. "
        "Place the downloaded annual Parquet files there before running this step."
    )
(TRADE_WORK_DIR / "1_ext_uncorrected").mkdir(parents=True, exist_ok=True)
(TRADE_WORK_DIR / "1_corrected").mkdir(parents=True, exist_ok=True)

# # 复制文件
# 
# 将`projects\2_research\2025.8.20 UN Comtrade贸易数据MFA耦合`获得的所有最终的 parquet 文件，复制到`raw_data`文件夹。
# 

# # 数据清洗
# 

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


# In[ ]:


import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
def display(*_objects):
    return None

# 获取研究时间范围
START_YEAR = 9999
END_YEAR = 0
raw_parquets = get_file_paths_with_extension(str(RAW_DATA_DIR), "parquet")
if not raw_parquets:
    raise FileNotFoundError(f"No Parquet files were found under {RAW_DATA_DIR}")
for file_path in raw_parquets:
    df = pd.read_parquet(file_path)
    START_YEAR = min(START_YEAR, df["trade_year"].min())
    END_YEAR = max(END_YEAR, df["trade_year"].max())
display(START_YEAR, END_YEAR)


# In[ ]:


# 读取全部dataframe，concat到一起
df_trade = pd.concat(
    [pd.read_parquet(file_path) for file_path in tqdm(raw_parquets)], ignore_index=True
)
df_trade["commodity_code"] = df_trade["commodity_code"].astype(int)
df_trade


# In[ ]:


# 数据规范校验：重复值、数据类型、NaN值、年份范围

# 检查是否有重复的项，尤其是在v1、v2数据杂交的情况下更需要注意
print("去重前：", df_trade.shape)
print(
    "假如去重后：",
    df_trade.drop_duplicates(
        subset=[
            "trade_year",
            "flow_code",
            "reporter_comtrade_id",
            "partner_comtrade_id",
            "commodity_code",
        ]
    ).shape,
)
# 如果行列没有发生变化，说明没有重复项


# In[ ]:


for col in df_trade.columns:
    print(col, df_trade[col].dtype)
# 除了flow_code是string（object）类型之外，其它的都是int64/float64


# In[ ]:


for col in df_trade.columns:
    print(col, df_trade[col].isna().any())

# 价格和重量可能存在na值，以前的屎山代码是填0处理的，这里也会这么做
df_trade["monetary_usd"] = df_trade["monetary_usd"].fillna(0)
df_trade["weight_unified_kg"] = df_trade["weight_unified_kg"].fillna(0)

print("---")
for col in df_trade.columns:
    print(col, df_trade[col].isna().any())


# In[ ]:


assert df_trade["flow_code"].unique().tolist() == ["Import"]
# 保证只有进口数据


# In[ ]:


# 基于矩阵运算的一阶、二阶异常值滤波
year_max = END_YEAR
year_min = START_YEAR

LOW_THRESHOLD_FACTOR = 10  # 小贸易量高容错
HIGH_THRESHOLD_FACTOR = 3  # 大贸易量低容错

SMALL_VALUE = 1e-6  # 防止除零
SMALL_VALUE_PLACEBACK = SMALL_VALUE * 10

EXPORT_DIR_PATH = str(TRADE_WORK_DIR)


# In[ ]:


# 矩阵改正函数。导出处理后的结果到result目录
def correction_vectorized(df_trade):
    # 从之前运转良好的屎山代码复制过来的，不想重写了，做一下兼容处理
    df_trade_comtrade = df_trade.copy()
    df_trade_comtrade = df_trade_comtrade.rename(
        columns={
            "trade_year": "Year",
            "reporter_comtrade_id": "Importer",
            "partner_comtrade_id": "Exporter",
            "commodity_code": "CommodityCode",
            "weight_unified_kg": "NetWeight",
            "monetary_usd": "TradeValue",
        }
    )
    # 只保留需要的列
    df_trade_comtrade = df_trade_comtrade[
        [
            "Year",
            "Importer",
            "Exporter",
            "CommodityCode",
            "NetWeight",
            "TradeValue",
        ]
    ].copy()

    commodity_codes = list(df_trade_comtrade["CommodityCode"].unique())

    # == 之前的代码 ==
    # 主体计算逻辑
    for commodity_code in tqdm(commodity_codes):

        # 某一个CommodityCode的所有贸易关系
        df_commodity_c = df_trade_comtrade[
            df_trade_comtrade.CommodityCode == commodity_code
        ]
        df_commodity_c = df_commodity_c.sort_values(by=["Year", "Importer", "Exporter"])

        # 进口-出口国配对
        imp_exp_list = sorted(
            set(zip(df_commodity_c["Importer"], df_commodity_c["Exporter"]))
        )
        imp_exp_pair_to_index = {pair: i for i, pair in enumerate(imp_exp_list)}
        tqdm.write("进口-出口国配对数量: " + str(len(imp_exp_list)))

        # 构建贸易矩阵，X轴和Y轴分别为进口出口国配对的list index和以起始年份为起点的年份
        trade_matrix_amount = np.full(
            (len(imp_exp_list), year_max - year_min + 1), SMALL_VALUE
        )
        trade_matrix_trade_value = np.full(
            (len(imp_exp_list), year_max - year_min + 1), SMALL_VALUE
        )

        # 填充矩阵
        for row in tqdm(df_commodity_c.itertuples()):
            trade_matrix_amount[
                imp_exp_pair_to_index[(row.Importer, row.Exporter)],
                row.Year - year_min,
            ] = row.NetWeight
            if row.NetWeight == 0:
                trade_matrix_amount[
                    imp_exp_pair_to_index[(row.Importer, row.Exporter)],
                    row.Year - year_min,
                ] = SMALL_VALUE
            trade_matrix_trade_value[
                imp_exp_pair_to_index[(row.Importer, row.Exporter)],
                row.Year - year_min,
            ] = row.TradeValue
            if row.TradeValue == 0:
                trade_matrix_trade_value[
                    imp_exp_pair_to_index[(row.Importer, row.Exporter)],
                    row.Year - year_min,
                ] = SMALL_VALUE

        # 建立改正后的矩阵
        trade_matrix_amount_corrected = trade_matrix_amount.copy()
        trade_matrix_trade_value_corrected = trade_matrix_trade_value.copy()

        # Correct only single- and double-year peaks on both value paths.
        trade_matrix_amount_corrected, _amount_peak_counts = (
            correct_single_and_double_peaks(trade_matrix_amount_corrected)
        )
        trade_matrix_trade_value_corrected, _trade_value_peak_counts = (
            correct_single_and_double_peaks(trade_matrix_trade_value_corrected)
        )

        # 导出数据
        year_array_list = []
        importer_array_list = []
        exporter_array_list = []
        amount_array_list = []
        amount_corrected_list = []
        trade_value_array_list = []
        trade_value_corrected_list = []
        commodity_code_array_list = []
        for i in range(len(imp_exp_list)):
            year_array_list.append(np.arange(year_min, year_max + 1))
            importer_array_list.append(
                np.full(year_max - year_min + 1, imp_exp_list[i][0])
            )
            exporter_array_list.append(
                np.full(year_max - year_min + 1, imp_exp_list[i][1])
            )
            amount_array_list.append(trade_matrix_amount[i])
            amount_corrected_list.append(trade_matrix_amount_corrected[i])
            trade_value_array_list.append(trade_matrix_trade_value[i])
            trade_value_corrected_list.append(trade_matrix_trade_value_corrected[i])
            commodity_code_array_list.append(
                np.full(year_max - year_min + 1, commodity_code)
            )
        year_array = np.concatenate(year_array_list, axis=0)
        importer_array = np.concatenate(importer_array_list, axis=0)
        exporter_array = np.concatenate(exporter_array_list, axis=0)
        amount_array = np.concatenate(amount_array_list, axis=0)
        amount_corrected = np.concatenate(amount_corrected_list, axis=0)
        trade_value_array = np.concatenate(trade_value_array_list, axis=0)
        trade_value_corrected = np.concatenate(trade_value_corrected_list, axis=0)
        commodity_code_array = np.concatenate(commodity_code_array_list, axis=0)
        df_commodity_full = pd.DataFrame(
            {
                "CommodityCode": commodity_code_array,
                "Importer": importer_array,
                "Exporter": exporter_array,
                "Year": year_array,
                "Amount": amount_array,
                "TradeValue": trade_value_array,
            }
        )
        df_commodity_corrected = pd.DataFrame(
            {
                "CommodityCode": commodity_code_array,
                "Importer": importer_array,
                "Exporter": exporter_array,
                "Year": year_array,
                "Amount": amount_corrected,
                "TradeValue": trade_value_corrected,
            }
        )
        df_commodity_full["Amount"] = df_commodity_full["Amount"].where(
            df_commodity_full["Amount"] > SMALL_VALUE_PLACEBACK, 0
        )
        df_commodity_full["TradeValue"] = df_commodity_full["TradeValue"].where(
            df_commodity_full["TradeValue"] > SMALL_VALUE_PLACEBACK, 0
        )
        df_commodity_corrected["Amount"] = df_commodity_corrected["Amount"].where(
            df_commodity_corrected["Amount"] > SMALL_VALUE_PLACEBACK, 0
        )
        df_commodity_corrected["TradeValue"] = df_commodity_corrected[
            "TradeValue"
        ].where(df_commodity_corrected["TradeValue"] > SMALL_VALUE_PLACEBACK, 0)

        # == 之前的代码 结束 ==

        # 转换回目前的列名
        df_commodity_full = df_commodity_full.rename(
            columns={
                "Year": "trade_year",
                "Importer": "reporter_comtrade_id",
                "Exporter": "partner_comtrade_id",
                "CommodityCode": "commodity_code",
                "Amount": "weight_unified_kg",
                "TradeValue": "monetary_usd",
            }
        )
        df_commodity_corrected = df_commodity_corrected.rename(
            columns={
                "Year": "trade_year",
                "Importer": "reporter_comtrade_id",
                "Exporter": "partner_comtrade_id",
                "CommodityCode": "commodity_code",
                "Amount": "weight_unified_kg",
                "TradeValue": "monetary_usd",
            }
        )
        # 导出
        df_commodity_full.to_parquet(
            os.path.join(
                EXPORT_DIR_PATH, "1_ext_uncorrected", str(commodity_code) + ".parquet"
            ),
            index=False,
        )
        df_commodity_corrected.to_parquet(
            os.path.join(
                EXPORT_DIR_PATH, "1_corrected", str(commodity_code) + ".parquet"
            ),
            index=False,
        )


correction_vectorized(df_trade)

