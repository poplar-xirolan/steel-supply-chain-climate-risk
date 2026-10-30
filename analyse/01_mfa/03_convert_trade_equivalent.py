#!/usr/bin/env python
# coding: utf-8

"""Step 3: fill trade weights from unit prices and convert to iron equivalent."""

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
parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, default=data_path('work/mfa_trade'))
parser.add_argument("--workbook", type=Path, default=data_path('input/mfa/mfa_inputs.xlsx'))
parser.add_argument(
    "--plot-unit-prices",
    action="store_true",
    help="Opt in to non-blocking unit-price diagnostic plots; batch default is off.",
)
args = parser.parse_args()

TRADE_WORK_DIR = args.work_dir.resolve()
INPUT_WORKBOOK = args.workbook.resolve()
PRICE_DIR = TRADE_WORK_DIR / "3_1_price_table"
FILLED_TRADE_PATH = TRADE_WORK_DIR / "3_2_weight_filled_trade" / "trade_data.parquet"
PRICE_DIR.mkdir(parents=True, exist_ok=True)
FILLED_TRADE_PATH.parent.mkdir(parents=True, exist_ok=True)

# 一般当量换算，不包括铁矿石

# # 单价计算与处理

# In[ ]:


import pandas as pd
from steel_supply.mfa.trade_equivalent_conversion import (
    build_price_table,
    deterministic_trade_order,
    fill_trade_values,
    interpolate_price_table,
    plot_unit_price_trends,
    sum_weight_and_money,
)


def display(*_objects):
    return None

df_trade_unfilled = deterministic_trade_order(
    pd.read_parquet(TRADE_WORK_DIR / "2_country_mapped" / "trade_data.parquet")
)
display(df_trade_unfilled)

SMALL_VALUE = 1e-6 * 10  # 与 1_数据清洗.ipynb 的SMALL_VALUE_PLACEBACK一致


# In[ ]:


# 获取某一年、某一个商品，所有同时拥有价格、重量的行的重量和价格总和。返回一个元组 (weight_sum, price_sum)
# 正常情况返回两个(float, float)，如果没有符合条件的行，返回(None, None)
def sum_weight_price_by_commodity_year(commodity_code, trade_year):
    return sum_weight_and_money(
        df_trade_unfilled, commodity_code, trade_year, SMALL_VALUE
    )


sum_weight_price_by_commodity_year(671, 1963)


# In[ ]:


# 构建单价表
df_unit_price = build_price_table(df_trade_unfilled, SMALL_VALUE)
df_unit_price.to_excel(PRICE_DIR / "unit_price.xlsx", index=False)


# In[ ]:


df_unit_price


# In[ ]:


# 人工诊断图仅在显式 opt-in 时非阻塞显示，并逐图关闭。
plot_unit_price_trends(df_unit_price, enabled=args.plot_unit_prices)

# 查阅后文的图表，确保感觉上进行插值没问题，而且总体比较合理


# In[ ]:


# 对每个commodity_code的unit_price进行中间线性插值，首尾端简单延伸插值
df_unit_price = interpolate_price_table(df_unit_price)

display(df_unit_price)
df_unit_price.to_excel(
    PRICE_DIR / "unit_price_interpolated.xlsx", index=False
)


# # 贸易重量填充

# In[ ]:


# 动态规划缓存
_cache_dict = {}


# 辅助函数，获取某一年、某一个商品的插值后单价
def get_usd_per_kg(commodity_code, trade_year):
    global _cache_dict
    key = (commodity_code, trade_year)
    if key in _cache_dict:
        return _cache_dict[key]

    row = df_unit_price[
        (df_unit_price["commodity_code"] == commodity_code)
        & (df_unit_price["trade_year"] == trade_year)
    ]
    if row.empty:
        raise ValueError(
            f"No data for commodity_code {commodity_code} in year {trade_year}"
        )
    _cache_dict[key] = row.iloc[0]["unit_price_usd_per_kg_interpolated"]
    return _cache_dict[key]


# In[ ]:


# 计算重量或者价格，对于缺失的重量或者价格，使用单价进行换算

# 合并价格和贸易表，计算缺失的重量或者价格
df_trade_filled = fill_trade_values(
    df_trade_unfilled, df_unit_price, SMALL_VALUE
)

df_trade_filled


# # 贸易当量换算

# In[ ]:


IRON_EQ_SHEET_PATH = INPUT_WORKBOOK
SHEET_NAME = "<参考资料>铁SITC 1贸易代码"
df_iron_eq = pd.read_excel(IRON_EQ_SHEET_PATH, sheet_name=SHEET_NAME)

df_trade_filled["iron_eq_ratio_unit_1"] = None

for idx, row in df_iron_eq.iterrows():
    sitc_code = row["SITCCode"]
    iron_concentration = row["IronConcentration (Unit:1)"]
    df_trade_filled.loc[
        df_trade_filled["commodity_code"] == sitc_code, "iron_eq_ratio_unit_1"
    ] = iron_concentration

df_trade_filled["iron_eq_weight_unified_kg"] = (
    df_trade_filled["weight_unified_kg"] * df_trade_filled["iron_eq_ratio_unit_1"]
)

df_trade_filled


# In[ ]:


df_trade_filled.to_parquet(FILLED_TRADE_PATH)

