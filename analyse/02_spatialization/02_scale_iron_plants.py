#!/usr/bin/env python
# coding: utf-8

"""Spatialize national P2 pig-iron production to GEM iron plants."""

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
parser.add_argument("--basic-info", type=Path, default=data_path('input/spatialization/global_energy_monitor_iron_steel/basic_info.parquet'))
parser.add_argument("--capacity", type=Path, default=data_path('input/spatialization/global_energy_monitor_iron_steel/capacity.parquet'))
parser.add_argument("--mfa", type=Path, default=data_path('results/mfa/2023.parquet'))
parser.add_argument("--output", type=Path, default=data_path('results/facilities/iron_factory_scaled.parquet'))
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

# In[17]:


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

# In[18]:


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

# In[19]:


GEO_BASIC_INFO_PATH = args.basic_info
geo_basic_info_df = pd.read_parquet(GEO_BASIC_INFO_PATH)
geo_basic_info_df


# In[20]:


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

# In[21]:


GEO_CAPACITY_PATH = args.capacity

geo_basic_info_df["iron_weight_total_kg"] = None

df_capacity = pd.read_parquet(GEO_CAPACITY_PATH)
df_capacity = df_capacity[(df_capacity["Status"].isin(["operating", "operating pre-retirement"])) & (df_capacity["Type"] == "Iron")]  # 仅保留运营中的钢铁厂数据

display(df_capacity)

# 有total字段的capacity，优先选择total字段
df_capacity_with_total = df_capacity[df_capacity["Sub_type"] == "Total"]
agg_df_capacity_with_total = df_capacity_with_total.groupby("ID_internal", as_index=False)["Capacity"].sum()

# 没有total字段的capacity，加和其他类型的capacity，作为总capacity
ids_with_total = agg_df_capacity_with_total["ID_internal"].unique()
df_capacity_without_total = df_capacity[~df_capacity["ID_internal"].isin(ids_with_total)]
agg_df_capacity_without_total = df_capacity_without_total.groupby("ID_internal", as_index=False)["Capacity"].sum()

# 合并两部分capacity数据
agg_df_capacity = pd.concat([agg_df_capacity_with_total, agg_df_capacity_without_total], ignore_index=True)
display(agg_df_capacity)

for _, row in agg_df_capacity.iterrows():
    id_internal = row["ID_internal"]
    geo_basic_info_df.loc[geo_basic_info_df["ID_internal"] == id_internal, "iron_weight_total_kg"] = row["Capacity"] * 1000000  # kt to kg

geo_basic_info_df.dropna(subset=["iron_weight_total_kg"], inplace=True)

geo_basic_info_df


# In[22]:


IRON_FE_EQ = 0.96  # MFA系统定义

geo_basic_info_df["fe_eq_percentage"] = IRON_FE_EQ
geo_basic_info_df["fe_eq_weight_kg"] = geo_basic_info_df["iron_weight_total_kg"] * IRON_FE_EQ

geo_basic_info_df


# # 基于 MFA 流，进行产量放缩
# 

# In[25]:


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


def get_iron_prod_fe_eq_kg(country_a3):
    return sum_mass(entity_from=country_a3, entity_to=country_a3, sector_from="P2", sector_to="M2")


# 获得点位数据表中所有的国家代号
unique_countries = geo_basic_info_df["country_a3"].unique()
# 分别计算点位数据表和mfa系统中的铁当量产量
country_fe_eq_prod_dict = {}
for country in unique_countries:
    fe_eq_prod_kg_geosum = geo_basic_info_df[geo_basic_info_df["country_a3"] == country]["fe_eq_weight_kg"].sum()
    country_fe_eq_prod_dict[country] = {"fe_eq_prod_kg_geosum": fe_eq_prod_kg_geosum, "fe_eq_prod_kg_mfa": get_iron_prod_fe_eq_kg(country)}
    country_fe_eq_prod_dict[country]["scaling_factor"] = (
        country_fe_eq_prod_dict[country]["fe_eq_prod_kg_mfa"] / country_fe_eq_prod_dict[country]["fe_eq_prod_kg_geosum"]
    )

display(pd.DataFrame(country_fe_eq_prod_dict))

# 应用放缩系数
geo_basic_info_df["iron_weight_total_kg_scaled"] = None
geo_basic_info_df["fe_eq_weight_kg_scaled"] = None
for idx, row in tqdm(geo_basic_info_df.iterrows(), total=len(geo_basic_info_df), desc="应用放缩系数"):
    country_a3 = row["country_a3"]
    scaling_factor = country_fe_eq_prod_dict[country_a3]["scaling_factor"]
    geo_basic_info_df.at[idx, "iron_weight_total_kg_scaled"] = row["iron_weight_total_kg"] * scaling_factor
    geo_basic_info_df.at[idx, "fe_eq_weight_kg_scaled"] = row["fe_eq_weight_kg"] * scaling_factor

# 删除产量为0的点位
geo_basic_info_df = geo_basic_info_df[geo_basic_info_df["fe_eq_weight_kg_scaled"] > 0]


# 重新检查
check_dict = {}
for country in unique_countries:
    fe_eq_prod_kg_geosum_scaled = geo_basic_info_df[geo_basic_info_df["country_a3"] == country]["fe_eq_weight_kg_scaled"].sum()
    fe_eq_prod_kg_mfa = country_fe_eq_prod_dict[country]["fe_eq_prod_kg_mfa"]
    check_dict[country] = {"fe_eq_prod_kg_geosum_scaled": fe_eq_prod_kg_geosum_scaled, "fe_eq_prod_kg_mfa": fe_eq_prod_kg_mfa}
check_dict


# In[26]:


display(geo_basic_info_df)
# 保存结果
OUTPUT_PATH = args.output
geo_basic_info_df.to_parquet(OUTPUT_PATH)

