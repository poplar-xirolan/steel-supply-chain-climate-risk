#!/usr/bin/env python
# coding: utf-8

"""Spatialize national P1 iron-ore production to S&P mine locations."""

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
parser.add_argument("--year", type=int, default=2023)
parser.add_argument("--boundaries", type=Path, default=data_path('input/spatialization/world_administration_esri.gpkg'))
parser.add_argument("--workbook", type=Path, default=data_path('input/mfa/mfa_inputs.xlsx'))
parser.add_argument("--basic-info", type=Path, default=data_path('input/spatialization/spglobal_iron_mine/basic_info.parquet'))
parser.add_argument("--production", type=Path, default=data_path('input/spatialization/spglobal_iron_mine/production.parquet'))
parser.add_argument("--mfa", type=Path, default=data_path('results/mfa/2023.parquet'))
parser.add_argument("--output", type=Path, default=data_path('results/facilities/mine_scaled.parquet'))
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

# In[1]:


import pandas as pd
import numpy as np
import geopandas as gpd
def display(*_objects):
    return None
from shapely.geometry import Point, Polygon
from tqdm import tqdm

YEAR = args.year


# # 国家代号+行政区划边界数据获取
# 

# In[2]:


# 行政区划数据
GEO_ADMIN_PATH = args.boundaries

# 国家代号数据
COUNTRY_CODE_INFO_PATH = args.workbook
COUNTRY_CODE_INFO_SHEET_NAME = "国家或地区代号"

gdf_admin = gpd.read_file(GEO_ADMIN_PATH)

# 补全ISO_A3代号
gdf_admin["ISO_A3"] = None
df_country_code = pd.read_excel(COUNTRY_CODE_INFO_PATH, sheet_name=COUNTRY_CODE_INFO_SHEET_NAME)
for idx, row in tqdm(df_country_code.iterrows(), total=len(df_country_code), desc="补全ISO_A3代号"):
    iso_a2 = row["Alpha-2 code"]
    iso_a3 = row["Alpha-3 code"]
    gdf_admin.loc[gdf_admin["ISO_CC"] == iso_a2, "ISO_A3"] = iso_a3


# # 点位数据整理+国家代号分配
# 

# In[3]:


GEO_BASIC_INFO_PATH = args.basic_info
geo_basic_info_df = pd.read_parquet(GEO_BASIC_INFO_PATH)
geo_basic_info_df


# In[4]:


geo_basic_info_df["country_a3"] = None
geo_basic_info_df["dist_to_country"] = None  # 新增距离列

for idx, row in tqdm(geo_basic_info_df.iterrows(), total=len(geo_basic_info_df), desc="补全country_a3代号及距离"):
    point = Point(row["Geo_lon"], row["Geo_lat"])

    # 1. 优先尝试直接包含关系 (Contains)
    # 注意：gdf_admin 需要是 GeoDataFrame
    country_contains = gdf_admin[gdf_admin.contains(point)]

    if len(country_contains) > 0:
        # 如果点在国家内部，距离为 0
        geo_basic_info_df.at[idx, "country_a3"] = country_contains.iloc[0]["ISO_A3"]
        geo_basic_info_df.at[idx, "dist_to_country"] = 0.0

        if len(country_contains) > 1:
            print(f"提示：点({row['Geo_lon']}, {row['Geo_lat']})位于多个国家边界重叠处，默认取第一个：{country_contains.iloc[0]['ADMIN']}。")

    else:
        # 2. 如果未找到包含国家，寻找最近的国家 (Nearest)
        # 计算该点到 gdf_admin 中所有几何体的距离
        # 注意：如果坐标系是经纬度(WGS84)，这里的距离单位是“度”
        # ！“度”距离不严谨，仅供参考！经过检验，适用于本项目的分配估算
        distances = gdf_admin.distance(point)

        # 找到最小距离及其对应的索引
        min_dist = distances.min()
        nearest_idx = distances.idxmin()
        nearest_country = gdf_admin.loc[nearest_idx]

        # 赋值
        geo_basic_info_df.at[idx, "country_a3"] = nearest_country["ISO_A3"]
        geo_basic_info_df.at[idx, "dist_to_country"] = min_dist

        print(f"警告：点({row['Geo_lon']}, {row['Geo_lat']})未找到包含国家，已匹配最近国家 {nearest_country['ISO_A3']}，距离 {min_dist:.4f}（度）。")

# 这里的 dropna 只会删掉那些连最近邻都找不到的极端情况（通常不会发生，除非 gdf_admin 为空）
geo_basic_info_df.dropna(subset=["country_a3"], inplace=True)

# 显示结果前几行
print("处理完成。")
geo_basic_info_df


# # 点位数据生产量计算+铁当量换算
# 

# In[5]:


GEO_PRODUCTION_PATH = args.production

geo_basic_info_df["ore_weight_total_kg"] = None

df_production = pd.read_parquet(GEO_PRODUCTION_PATH)
df_production = df_production[df_production["Prod_year"] == YEAR]

for _, row in df_production.iterrows():
    id_internal = row["ID_internal"]
    geo_basic_info_df.loc[geo_basic_info_df["ID_internal"] == id_internal, "ore_weight_total_kg"] = row["Production"] * 1000000  # kt to kg

geo_basic_info_df.dropna(subset=["ore_weight_total_kg"], inplace=True)

geo_basic_info_df


# In[6]:


FE_PROD_USGS_XLSX_PATH = args.workbook
FE_PROD_USGS_SHEET_NAME = "铁矿石产量、品位信息_USGS"

df_fe_prod = pd.read_excel(FE_PROD_USGS_XLSX_PATH, sheet_name=FE_PROD_USGS_SHEET_NAME)


# 基于铁矿石产量表，读取铁当量
def read_fe_eq(country_code_iso_a3, year):
    import re

    # Step 1: Find an exact match for the country and year
    exact_year_col = f"{year} Fe eq（以1计）"
    if exact_year_col in df_fe_prod.columns:
        country_data = df_fe_prod[df_fe_prod["Country_code_iso3166-2_alpha_3"] == country_code_iso_a3]
        if not country_data.empty:
            fe_eq_value = country_data.iloc[0][exact_year_col]
            if pd.notna(fe_eq_value):
                print(f"使用了精确匹配数据: {country_code_iso_a3}, {year} -> {fe_eq_value}")
                return fe_eq_value

    # Step 2: Find the most adjacent year for the given country
    country_data = df_fe_prod[df_fe_prod["Country_code_iso3166-2_alpha_3"] == country_code_iso_a3]
    if not country_data.empty:
        fe_eq_cols = [col for col in df_fe_prod.columns if "Fe eq" in col]

        available_data = country_data.iloc[0][fe_eq_cols].dropna()

        if not available_data.empty:
            available_years = {}
            for col_name, value in available_data.items():
                match = re.search(r"(\d{4}) Fe eq", col_name)
                if match:
                    available_years[int(match.group(1))] = value

            if available_years:
                closest_year = min(available_years.keys(), key=lambda y: abs(y - year))
                print(f"使用了最邻近年份数据: {country_code_iso_a3}, {closest_year} -> {available_years[closest_year]}")
                return available_years[closest_year]

    # Step 3: Fallback to World Average for the exact target year
    if exact_year_col in df_fe_prod.columns:
        total_data = df_fe_prod[df_fe_prod["Country_code_iso3166-2_alpha_3"] == "Total"]
        if not total_data.empty:
            total_fe_eq = total_data.iloc[0][exact_year_col]
            if pd.notna(total_fe_eq):
                print(f"使用了精确年份世界平均数据: Total, {year} -> {total_fe_eq}")
                return total_fe_eq

    # Step 4: Fallback to the most adjacent World Average year
    total_data = df_fe_prod[df_fe_prod["Country_code_iso3166-2_alpha_3"] == "Total"]
    if not total_data.empty:
        fe_eq_cols = [col for col in df_fe_prod.columns if "Fe eq" in col]
        available_data = total_data.iloc[0][fe_eq_cols].dropna()

        if not available_data.empty:
            available_years = {}
            for col_name, value in available_data.items():
                match = re.search(r"(\d{4}) Fe eq", col_name)
                if match:
                    available_years[int(match.group(1))] = value

            if available_years:
                closest_year = min(available_years.keys(), key=lambda y: abs(y - year))
                print(f"使用了最邻近年份的世界平均数据: Total, {closest_year} -> {available_years[closest_year]}")
                return available_years[closest_year]

    raise ValueError(f"read_fe_eq代码逻辑错误：全部边界条件未能捕获到 country_code_iso_a3={country_code_iso_a3}, year={year} 的数据")


geo_basic_info_df["fe_eq_percentage"] = None
geo_basic_info_df["fe_eq_weight_kg"] = None

for idx, row in tqdm(geo_basic_info_df.iterrows(), total=len(geo_basic_info_df), desc="计算铁当量"):
    country_a3 = row["country_a3"]
    ore_weight_total_kg = row["ore_weight_total_kg"]
    fe_eq_percentage = read_fe_eq(country_a3, YEAR)
    fe_eq_weight_kg = ore_weight_total_kg * fe_eq_percentage
    geo_basic_info_df.at[idx, "fe_eq_percentage"] = fe_eq_percentage
    geo_basic_info_df.at[idx, "fe_eq_weight_kg"] = fe_eq_weight_kg

geo_basic_info_df


# # 基于 MFA 流，进行产量放缩
# 

# In[10]:


MFA_SYSTEM_PATH = args.mfa

df_mfa_system = pd.read_parquet(MFA_SYSTEM_PATH)


# 辅助函数，根据entity_from、entity_to、sector_from、sector_to过滤df_mfa数据，并加和mass列
# 输出单位：kg。这已经在前面的数据处理环节保证
# 输入字段如果为None，代表不过滤该字段
def sum_mass(entity_from=None, entity_to=None, sector_from=None, sector_to=None):
    df_filtered = df_mfa_system
    if entity_from is not None:
        df_filtered = df_filtered[df_filtered["entity_from"] == entity_from]
    if entity_to is not None:
        df_filtered = df_filtered[df_filtered["entity_to"] == entity_to]
    if sector_from is not None:
        df_filtered = df_filtered[df_filtered["sector_from"] == sector_from]
    if sector_to is not None:
        df_filtered = df_filtered[df_filtered["sector_to"] == sector_to]
    return df_filtered["mass"].sum()


def get_iron_ore_prod_fe_eq_kg(country_a3):
    return sum_mass(entity_from=country_a3, entity_to=country_a3, sector_from="P1", sector_to="M1")


# 获得点位数据表中所有的国家代号
unique_countries = geo_basic_info_df["country_a3"].unique()
# 分别计算点位数据表和mfa系统中的铁当量产量
country_fe_eq_prod_dict = {}
for country in unique_countries:
    fe_eq_prod_kg_geosum = geo_basic_info_df[geo_basic_info_df["country_a3"] == country]["fe_eq_weight_kg"].sum()
    country_fe_eq_prod_dict[country] = {"fe_eq_prod_kg_geosum": fe_eq_prod_kg_geosum, "fe_eq_prod_kg_mfa": get_iron_ore_prod_fe_eq_kg(country)}
    country_fe_eq_prod_dict[country]["scaling_factor"] = (
        country_fe_eq_prod_dict[country]["fe_eq_prod_kg_mfa"] / country_fe_eq_prod_dict[country]["fe_eq_prod_kg_geosum"]
    )

# 应用放缩系数
geo_basic_info_df["ore_weight_total_kg_scaled"] = None
geo_basic_info_df["fe_eq_weight_kg_scaled"] = None
for idx, row in tqdm(geo_basic_info_df.iterrows(), total=len(geo_basic_info_df), desc="应用放缩系数"):
    country_a3 = row["country_a3"]
    scaling_factor = country_fe_eq_prod_dict[country_a3]["scaling_factor"]
    geo_basic_info_df.at[idx, "ore_weight_total_kg_scaled"] = row["ore_weight_total_kg"] * scaling_factor
    geo_basic_info_df.at[idx, "fe_eq_weight_kg_scaled"] = row["fe_eq_weight_kg"] * scaling_factor

# 删除产量为0的点位
geo_basic_info_df = geo_basic_info_df[geo_basic_info_df["fe_eq_weight_kg_scaled"] > 0]


# 重新检查
check_dict={}
for country in unique_countries:
    fe_eq_prod_kg_geosum_scaled = geo_basic_info_df[geo_basic_info_df["country_a3"] == country]["fe_eq_weight_kg_scaled"].sum()
    fe_eq_prod_kg_mfa = country_fe_eq_prod_dict[country]["fe_eq_prod_kg_mfa"]
    check_dict[country] = {"fe_eq_prod_kg_geosum_scaled": fe_eq_prod_kg_geosum_scaled, "fe_eq_prod_kg_mfa": fe_eq_prod_kg_mfa}
check_dict


# In[11]:


display(geo_basic_info_df)
# 保存结果
OUTPUT_PATH = args.output
geo_basic_info_df.to_parquet(OUTPUT_PATH)

