"""Faithful script conversion of the final MFA construction notebook.

Paths and year were parameterized; computational statements and coefficients are retained.
Source SHA256: 5b86e7365453763481bbebf1323ffe4d03fdc49b2a1832e8723959792795f1a4
"""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


# %% [markdown] cell 0
# # MFA 系统构建脚本
# 
# 主要将原始数据转换成自洽的 MFA 系统所需的格式；为每个年份导出一个 parquet 文件（文件名为年份，4 位小写数字，比如`2025`），包括 6 个字段：
# 
# - `entity_from [string not null]`：ISO 3166-2 alpha-3 国家代号，流出国
# - `entity_to [string not null]`：ISO 3166-2 alpha-3 国家代号，流入国
# - `sector_from [string not null]`：流出部门代号，与 MFA 系统架构图对应
# - `sector_to [string not null]`：流入部门代号，与 MFA 系统架构图对应
# - `mass [float64 not null]`：流量大小。以铁当量计算
# - `unit [string not null]`：流量单位。必须为 kg
# - `extra [string nullable]`：可选字段，存储额外信息。例如，记录 P3 的三种子类别：BOF、EAF、OHF
# 
# 不能保证每个(来源国,流入国,流入部门,流出部门)四元组的行数都唯一；或者说，由于不同工艺、贸易商品类别的存在，至少有一个四元组一定不是唯一的。
# 
# 敲完代码的流记得在 MFA 系统里标绿，方便记录。
# 
# 一些系数什么的，请去组内之前的 MFA 项目以及我记的笔记里去找；找不到的可以再去找找 LQC 的论文

# %% cell 1
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

REPO_ROOT = bundle_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steel_supply.pipeline_receipts import sha256_file, write_receipt

from steel_supply.mfa.mfa_validation import calculate_node_balance, validate_mfa_schema


def display(_object):
    """Notebook display compatibility shim for the script workflow."""
    return None
import json
from loguru import logger
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Build the annual global steel MFA network.")
parser.add_argument("--year", type=int, default=2023)
parser.add_argument(
    "--workbook",
    default=data_path('input/mfa/mfa_inputs.xlsx'),
    help="Curated MFA input workbook.",
)
parser.add_argument(
    "--trade-data",
    default=data_path('work/mfa_trade/4_iron_ore_grade_adjusted/trade_data.parquet'),
    help="Processed UN Comtrade Parquet table from step 04.",
)
parser.add_argument(
    "--output-dir",
    default=data_path('results/mfa'),
)
parser.add_argument(
    "--receipt-path",
    type=Path,
    default=None,
    help="Optional path for a machine-readable build and validation receipt.",
)
parser.add_argument(
    "--disable-m8-s1-asu-fsp-correction",
    action="store_true",
    help=(
        "Disable the WorldSteel ASU-FSP floor used for countries whose raw M8 "
        "final-product market balance is zero or negative."
    ),
)
args = parser.parse_args()

YEAR = args.year
INPUT_WORKBOOK_PATH = args.workbook
TRADE_DATA_PATH_INPUT = args.trade_data
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"MFA system target year: {YEAR}")

# 构建结果所需的dataframe
df_result = pd.DataFrame(
    {
        "entity_from": pd.Series(dtype="str"),
        "entity_to": pd.Series(dtype="str"),
        "sector_from": pd.Series(dtype="str"),
        "sector_to": pd.Series(dtype="str"),
        "mass": pd.Series(dtype="float64"),
        "unit": pd.Series(dtype="str"),
        "extra": pd.Series(dtype="str"),
    }
)

logger.info("初始化结果DataFrame")
display(df_result)

# 读取国际贸易表
TRADE_DATA_PATH = TRADE_DATA_PATH_INPUT
# 如果国际贸易表已存在，不重复读取
if "DF_TRADE" in globals():
    logger.info("国际贸易表已存在，跳过读取步骤")
else:
    DF_TRADE = pd.read_parquet(TRADE_DATA_PATH)
logger.info("已读取国际贸易表 👇")
display(DF_TRADE)
logger.warning(
    "提醒：默认贸易数据为进口流，reporter -> importer，partner -> exporter。望确认"
)

# %% cell 2
# 获取MFA系统的ISO 3166-3 Alpha3国家代号
COUNTRY_CODE_PATH = INPUT_WORKBOOK_PATH
COUNTRY_CODE_SHEET_NAME = "国家或地区代号"
COUNTRY_CODES = []
df_country_code = pd.read_excel(COUNTRY_CODE_PATH, sheet_name=COUNTRY_CODE_SHEET_NAME)
for idx, row in df_country_code.iterrows():
    COUNTRY_CODES.append(row["Alpha-3 code"])
logger.info(f"获取到的国家代号 👇")
logger.info(COUNTRY_CODES)

# %% cell 3
# P1-M1
# 目前使用USGS数据完成
SHEET_P1_M1_PATH = INPUT_WORKBOOK_PATH
SHEET_P1_M1_NAME = "铁矿石产量、品位信息_USGS"

df_p1_m1_original = pd.read_excel(SHEET_P1_M1_PATH, sheet_name=SHEET_P1_M1_NAME)[
    ["Country_code_iso3166-2_alpha_3", f"{YEAR} 铁当量"]
].dropna(subset=[f"{YEAR} 铁当量"])

for idx, row in df_p1_m1_original.iterrows():
    # 跳过总计行
    if row["Country_code_iso3166-2_alpha_3"] == "Total":
        logger.info("跳过总计行")
        continue

    # 检查国家代号是否在已知国家代号列表中
    assert row["Country_code_iso3166-2_alpha_3"] in COUNTRY_CODES

    # 获取生产数据
    unit = "kg"
    entity_from = row["Country_code_iso3166-2_alpha_3"]
    entity_to = entity_from
    sector_from = "P1"
    sector_to = "M1"
    mass = row[f"{YEAR} 铁当量"] * 1000000  # USGS数据单位为百万吨，转换为kg

    # 追加到结果DataFrame
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("P1-M1 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 4
# 读取SITC贸易代码列表
def read_trade_code(commodity_type, commodity_subtype=None):
    TRADE_CODE_SHEET_PATH = INPUT_WORKBOOK_PATH
    TRADE_CODE_SHEET_NAME = "<参考资料>铁SITC 1贸易代码"
    df_trade_code = pd.read_excel(
        TRADE_CODE_SHEET_PATH, sheet_name=TRADE_CODE_SHEET_NAME
    )
    assert commodity_type in df_trade_code["CommodityType"].values

    # 没有提供子类别，仅使用主类别查找
    if commodity_subtype is None:
        trade_codes = df_trade_code[df_trade_code["CommodityType"] == commodity_type][
            "SITCCode"
        ].tolist()
        return trade_codes
    else:
        assert commodity_subtype in df_trade_code["CommoditySubtype"].values
        trade_codes = df_trade_code[
            (df_trade_code["CommodityType"] == commodity_type)
            & (df_trade_code["CommoditySubtype"] == commodity_subtype)
        ]["SITCCode"].tolist()
        return trade_codes


# 读取贸易数据条目
def read_trade_entries(year, trade_codes):
    df_trade_entries = DF_TRADE[
        (DF_TRADE["trade_year"] == year)
        & (DF_TRADE["commodity_code"].isin(trade_codes))
        & (DF_TRADE["reporter_iso_a3"].isin(COUNTRY_CODES))
        & (DF_TRADE["partner_iso_a3"].isin(COUNTRY_CODES))
        & (DF_TRADE["reporter_iso_a3"] != DF_TRADE["partner_iso_a3"])
        & (DF_TRADE["iron_eq_weight_unified_kg"] > 0)
    ]
    return df_trade_entries

# %% cell 5
# M1国际贸易
trade_code_m1 = read_trade_code("Iron Ore / Iron Concentrate")
logger.info(f"铁矿石贸易代码: {trade_code_m1}")
df_trade_m1 = read_trade_entries(YEAR, trade_code_m1)

logger.info(f"M1国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m1)

# 将M1国际贸易数据追加到结果DataFrame
for idx, row in tqdm(df_trade_m1.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M1"
    sector_to = "M1"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [extra],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M1国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 6
# 基于质量守恒，计算市场部门（M开头）向下一部门的输出量
#
# 返回结果：字典，{iso_3166_a3: [market_output_kg, inflow_kg, outflow_kg] ...}；
# - market_output_kg表示市场部门向下一部门的后续输出量，inflow_kg代表流入该市场部门的总量，outflow_kg代表流出该市场部门的总量（除下一部门的后续输出外）
#
# 注意事项：
# - 请注意执行顺序，如果流入某个sector的流还未完全量化就执行，会导致系统性误差
# - 如果向下一部门的输出量为负数（理论上不符合现实，只可能由于数据/系统误差引起），归0处理
def calculate_market_output(df_mfa, sector, need_negative_correction=True):
    net_inflow_dict = {}

    # 获取所有sector_from为相应部门的流
    df_outflow = df_mfa[df_mfa["sector_from"] == sector]
    # 获取所有sector_to为相应部门的流
    df_inflow = df_mfa[df_mfa["sector_to"] == sector]

    # 获取df_outflow中所有entity_from的独特值
    unique_entity_from = list(df_outflow["entity_from"].unique())
    # 获取df_inflow中所有entity_to的独特值
    unique_entity_to = list(df_inflow["entity_to"].unique())
    # 合并两个列表并去重，得到所有涉及该部门的entity
    all_entities = sorted(set(unique_entity_from + unique_entity_to))

    for entity in all_entities:
        # 计算流入量
        inflow = df_inflow[df_inflow["entity_to"] == entity]["mass"].sum()
        # 计算流出量
        outflow = df_outflow[df_outflow["entity_from"] == entity]["mass"].sum()
        # 计算净流入量
        net_inflow = inflow - outflow
        # 在要求负数改正的情况下，如果净流入量为负数，归0处理
        if need_negative_correction:
            net_inflow = max(net_inflow, 0)
            net_inflow_dict[entity] = [net_inflow, inflow, outflow]
        else:
            net_inflow_dict[entity] = [net_inflow, inflow, outflow]

    return net_inflow_dict


def read_worldsteel_asu_fsp_final_demand(year):
    """Read WorldSteel ASU-FSP as a conservative M8-S1 final-demand floor.

    WorldSteel ASU-FSP is reported in thousand tonnes of finished steel products.
    The MFA uses kg Fe eq, so the value is converted with the hot-rolled product
    Fe-equivalent rate used elsewhere in this script.
    """
    sheet_name = "表观钢铁消费量（钢铁最终产品重量）_Worldsteel"
    column_name = f"{year} ASU - FSP"
    try:
        df_asu_fsp = pd.read_excel(INPUT_WORKBOOK_PATH, sheet_name=sheet_name)[
            ["Country_code_iso3166-2_alpha_3", column_name]
        ]
    except (KeyError, ValueError):
        logger.warning(
            f"未找到 {sheet_name} 或 {column_name}，M8-S1 ASU-FSP 修正将被跳过。"
        )
        return {}

    asu_fsp_dict = {}
    for _, row in df_asu_fsp.iterrows():
        iso_code = row["Country_code_iso3166-2_alpha_3"]
        if iso_code == "Total" or iso_code not in COUNTRY_CODES:
            continue
        asu_fsp = row[column_name]
        if pd.isna(asu_fsp) or asu_fsp <= 0:
            continue
        asu_fsp_dict[iso_code] = (
            float(asu_fsp) * 1_000_000 * FE_EQ_RATE_HOT_ROLLED_PROD
        )
    return asu_fsp_dict

# %% cell 7
# M1-P2
# 质量守恒，计算P1-M1和M1国际贸易后的净流入量
net_inflow_dict = calculate_market_output(df_result, "M1")
logger.info("M1-P2 质量守恒计算完成，当前净流入量字典如下 👇")
logger.info(net_inflow_dict)

for key in net_inflow_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M1"
    sector_to = "P2"
    mass = net_inflow_dict[key][0]
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M1-P2 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 8
# P2-M2
# 目前使用USGS数据完成
SHEET_P2_M2_PATH = INPUT_WORKBOOK_PATH
SHEET_P2_M2_NAME = "粗铁产量信息_USGS"

# 粗铁中的铁含量
FE_EQ_RATE_PIG_IRON = 0.96
FE_EQ_RATE_DRI = 0.96
FE_EQ_RATE_TOTAL = (FE_EQ_RATE_PIG_IRON + FE_EQ_RATE_DRI) / 2

df_p2_m2_original = pd.read_excel(SHEET_P2_M2_PATH, sheet_name=SHEET_P2_M2_NAME)[
    [
        "Country_code_iso3166-2_alpha_3",
        f"{YEAR} pig iron",
        f"{YEAR} direct reduced iron",
        f"{YEAR} total",
    ]
]

for idx, row in df_p2_m2_original.iterrows():
    # 跳过总计行
    if row["Country_code_iso3166-2_alpha_3"] == "Total":
        logger.info("跳过总计行")
        continue

    # 检查国家代号是否在已知国家代号列表中
    assert row["Country_code_iso3166-2_alpha_3"] in COUNTRY_CODES

    # 获取生产数据
    pig_iron_prod = (
        row[f"{YEAR} pig iron"] if not pd.isna(row[f"{YEAR} pig iron"]) else None
    )
    dri_prod = (
        row[f"{YEAR} direct reduced iron"]
        if not pd.isna(row[f"{YEAR} direct reduced iron"])
        else None
    )
    total_prod = row[f"{YEAR} total"] if not pd.isna(row[f"{YEAR} total"]) else None

    unit = "kg"
    entity_from = row["Country_code_iso3166-2_alpha_3"]
    entity_to = entity_from
    sector_from = "P2"
    sector_to = "M2"

    # 如果有分项数据，优先使用分项数据，且不记录总数据
    if (pig_iron_prod is not None) or (dri_prod is not None):
        if pig_iron_prod is not None:
            mass = pig_iron_prod * 1000000 * FE_EQ_RATE_PIG_IRON
            extra = "BF"
            df_result = pd.concat(
                [
                    df_result,
                    pd.DataFrame(
                        {
                            "entity_from": [entity_from],
                            "entity_to": [entity_to],
                            "sector_from": [sector_from],
                            "sector_to": [sector_to],
                            "mass": [mass],
                            "unit": [unit],
                            "extra": [extra],
                        }
                    ),
                ],
                ignore_index=True,
            )
        if dri_prod is not None:
            mass = dri_prod * 1000000 * FE_EQ_RATE_DRI
            extra = "DRI"
            df_result = pd.concat(
                [
                    df_result,
                    pd.DataFrame(
                        {
                            "entity_from": [entity_from],
                            "entity_to": [entity_to],
                            "sector_from": [sector_from],
                            "sector_to": [sector_to],
                            "mass": [mass],
                            "unit": [unit],
                            "extra": [extra],
                        }
                    ),
                ],
                ignore_index=True,
            )
    else:
        # 否则使用总量数据
        if total_prod is not None:
            mass = total_prod * 1000000 * FE_EQ_RATE_TOTAL
            extra = "Total"
            df_result = pd.concat(
                [
                    df_result,
                    pd.DataFrame(
                        {
                            "entity_from": [entity_from],
                            "entity_to": [entity_to],
                            "sector_from": [sector_from],
                            "sector_to": [sector_to],
                            "mass": [mass],
                            "unit": [unit],
                            "extra": [extra],
                        }
                    ),
                ],
                ignore_index=True,
            )
        else:
            continue

logger.info("P2-M2 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 9
# P2-Environment，冶铁废弃物
# 基于P2-M2，通过转换率计算
IRON_MAKING_BF_SLAG_RATE = 0.5 / 100
IRON_MAKING_DRI_SLAG_RATE = 0.5 / 100
IRON_MAKING_TOTAL_SLAG_RATE = (IRON_MAKING_BF_SLAG_RATE + IRON_MAKING_DRI_SLAG_RATE) / 2

df_p2_env = df_result[
    (df_result["sector_from"] == "P2") & (df_result["sector_to"] == "M2")
].copy()
df_p2_env["sector_to"] = "Environment"
df_p2_env.loc[df_p2_env["extra"] == "BF", "mass"] *= IRON_MAKING_BF_SLAG_RATE
df_p2_env.loc[df_p2_env["extra"] == "DRI", "mass"] *= IRON_MAKING_DRI_SLAG_RATE
df_p2_env.loc[df_p2_env["extra"] == "Total", "mass"] *= IRON_MAKING_TOTAL_SLAG_RATE

df_result = pd.concat([df_result, df_p2_env], ignore_index=True)

logger.info("P2-Environment 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 10
# M2国际贸易
trade_code_m2 = read_trade_code("Pig Iron")
logger.info(f"粗铁贸易代码: {trade_code_m2}")
df_trade_m2 = read_trade_entries(YEAR, trade_code_m2)

logger.info(f"M2国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m2)

for idx, row in tqdm(df_trade_m2.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M2"
    sector_to = "M2"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [extra],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M2国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 11
# M2-P3 & M2-P4
# 质量守恒，计算M2国际贸易后的净市场供应量
market_supply_dict = calculate_market_output(df_result, "M2")
logger.info("M2-P3 & M2-P4 质量守恒计算完成，当前净市场供应量字典如下 👇")
logger.info(market_supply_dict)

CASTING_BYPASS_RATE = (
    7 / 100
)  # 铸造旁路率，假设7%的粗铁直接作为铸铁，不经过冶钢，而是前往使用环节
for key in market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M2"
    mass = market_supply_dict[key][0]
    unit = "kg"

    # 计算流向P3的量
    sector_to = "P3"
    mass_to_p3 = mass * (1 - CASTING_BYPASS_RATE)
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass_to_p3],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

    # 计算流向P4的量
    sector_to = "P4"
    mass_to_p4 = mass * CASTING_BYPASS_RATE
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass_to_p4],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M2-P3 & M2-P4 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 12
# P3-M3
# 目前使用Worldsteel数据完成

# crude steel中的铁含量
# SRC1：https://worldsteel.org/wp-content/uploads/worldsteel-book-final-2022-1.pdf，page 8
# SRC2：SITC贸易代码官方定义
# 通过上述两者综合而成；btw，一点小的偏差不会引起大问题
FE_EQ_RATE_CRUDE_STEEL = 0.98

SHEET_P3_M3_PATH = INPUT_WORKBOOK_PATH
SHEET_P3_M3_NAME = "钢铁产量信息_Worldsteel"

df_p3_m3_original = pd.read_excel(SHEET_P3_M3_PATH, sheet_name=SHEET_P3_M3_NAME)[
    ["Country_code_iso3166-2_alpha_3", f"{YEAR} BOF", f"{YEAR} EAF", f"{YEAR} Other"]
]
df_p3_m3_original

for idx, row in df_p3_m3_original.iterrows():
    # 跳过总计行
    if row["Country_code_iso3166-2_alpha_3"] == "Total":
        logger.info("跳过总计行")
        continue

    # 检查国家代号是否在已知国家代号列表中
    assert row["Country_code_iso3166-2_alpha_3"] in COUNTRY_CODES

    # 获取生产数据
    bof_prod = row[f"{YEAR} BOF"] if not pd.isna(row[f"{YEAR} BOF"]) else None
    eaf_prod = row[f"{YEAR} EAF"] if not pd.isna(row[f"{YEAR} EAF"]) else None
    other_prod = row[f"{YEAR} Other"] if not pd.isna(row[f"{YEAR} Other"]) else None

    unit = "kg"
    entity_from = row["Country_code_iso3166-2_alpha_3"]
    entity_to = entity_from
    sector_from = "P3"
    sector_to = "M3"

    # BOF
    if bof_prod is not None:
        mass = bof_prod * 1000000 * FE_EQ_RATE_CRUDE_STEEL
        extra = "BOF"
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    {
                        "entity_from": [entity_from],
                        "entity_to": [entity_to],
                        "sector_from": [sector_from],
                        "sector_to": [sector_to],
                        "mass": [mass],
                        "unit": [unit],
                        "extra": [extra],
                    }
                ),
            ],
            ignore_index=True,
        )

    # EAF
    if eaf_prod is not None:
        mass = eaf_prod * 1000000 * FE_EQ_RATE_CRUDE_STEEL
        extra = "EAF"
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    {
                        "entity_from": [entity_from],
                        "entity_to": [entity_to],
                        "sector_from": [sector_from],
                        "sector_to": [sector_to],
                        "mass": [mass],
                        "unit": [unit],
                        "extra": [extra],
                    }
                ),
            ],
            ignore_index=True,
        )

    # OHF
    if other_prod is not None:
        mass = other_prod * 1000000 * FE_EQ_RATE_CRUDE_STEEL
        extra = "OHF"
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    {
                        "entity_from": [entity_from],
                        "entity_to": [entity_to],
                        "sector_from": [sector_from],
                        "sector_to": [sector_to],
                        "mass": [mass],
                        "unit": [unit],
                        "extra": [extra],
                    }
                ),
            ],
            ignore_index=True,
        )

logger.info("P3-M3 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 13
# P3-Environment，炼钢废弃物
# 基于P3-M3，通过转换率计算
STEEL_MAKING_BOF_SLAG_RATE = 4 / 100
STEEL_MAKING_EAF_SLAG_RATE = 4 / 100
STEEL_MAKING_OHF_SLAG_RATE = 4 / 100

df_p3_env = df_result[
    (df_result["sector_from"] == "P3") & (df_result["sector_to"] == "M3")
].copy()
df_p3_env["sector_to"] = "Environment"
df_p3_env.loc[df_p3_env["extra"] == "BOF", "mass"] *= STEEL_MAKING_BOF_SLAG_RATE
df_p3_env.loc[df_p3_env["extra"] == "EAF", "mass"] *= STEEL_MAKING_EAF_SLAG_RATE
df_p3_env.loc[df_p3_env["extra"] == "OHF", "mass"] *= STEEL_MAKING_OHF_SLAG_RATE

df_result = pd.concat([df_result, df_p3_env], ignore_index=True)

logger.info("P3-Environment 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 14
# P4-M4
# 基于P3生产数据，通过转换率计算
CASTING_BYPASS_RATE_P3_TO_P4 = CASTING_BYPASS_RATE / (
    1 - CASTING_BYPASS_RATE
)  # 从P3到P4的旁路率

df_p4_m4 = df_result[
    (df_result["sector_from"] == "P3") & (df_result["sector_to"] == "M3")
].copy()

# 删除extra字段
df_p4_m4["extra"] = None
# 加和mass字段，对于其他字段均一致的行
df_p4_m4 = df_p4_m4.groupby(
    by=["entity_from", "entity_to", "sector_from", "sector_to", "unit", "extra"],
    dropna=False,
    as_index=False,
).agg({"mass": "sum"})

# 计算流向P4的量
df_p4_m4["mass"] *= CASTING_BYPASS_RATE_P3_TO_P4
df_p4_m4["sector_from"] = "P4"
df_p4_m4["sector_to"] = "M4"

df_result = pd.concat([df_result, df_p4_m4], ignore_index=True)

logger.info("P4-M4 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 15
# M3国际贸易
tade_code_m3 = read_trade_code("Cast steel")
logger.info(f"钢材贸易代码: {tade_code_m3}")
df_trade_m3 = read_trade_entries(YEAR, tade_code_m3)

logger.info(f"M3国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m3)

for idx, row in tqdm(df_trade_m3.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M3"
    sector_to = "M3"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [extra],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M3国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 16
# M4国际贸易
trade_code_m4 = read_trade_code("Cast and forged iron")
logger.info(f"铸铁贸易代码: {trade_code_m4}")
df_trade_m4 = read_trade_entries(YEAR, trade_code_m4)

logger.info(f"M4国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m4)

for idx, row in tqdm(df_trade_m4.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M4"
    sector_to = "M4"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [extra],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M4国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 17
# M4-P6
# 质量守恒，计算M4国际贸易后的净市场供应量
market_supply_dict = calculate_market_output(df_result, "M4")
logger.info("M4-P6 质量守恒计算完成，当前净市场供应量字典如下 👇")
logger.info(market_supply_dict)

for key in market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M4"
    sector_to = "P6"
    mass = market_supply_dict[key][0]
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M4-P6 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 18
# M3-P5
# 质量守恒，计算M3国际贸易后的净市场供应量
market_supply_dict = calculate_market_output(df_result, "M3")
logger.info("M3-P5 质量守恒计算完成，当前净市场供应量字典如下 👇")
logger.info(market_supply_dict)

for key in market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M3"
    sector_to = "P5"
    mass = market_supply_dict[key][0]
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M3-P5 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 19
# P5-M5
# 目前使用Worldsteel数据完成

# 热轧产品中的铁含量
# 我们认为，热轧主要改变钢材的形状和内部微观结构，对化学成分没有显著影响
# crude steel后续可能会经过合金过程，生产不锈钢等产品，在这一过程中，铁含量可能会显著减少
# 但是，碳钢这一工序的合金，基本不会产生显著的铁含量变化
# 因此，我们需要知道碳钢、不锈钢等的占比。总体而言，碳钢占比远大于不锈钢
# 上面这一推论需要引用，以下是几个资料来源
# https://www.eurofer.eu/assets/publications/brochures-booklets-and-factsheets/european-steel-in-figures-2024/EUROFER-2024-Version-June14.pdf
# - 第18页，欧洲碳钢vs不锈钢占比
# https://www.emergenresearch.com/industry-report/steel-market/market-share
# - "As a matter of fact, carbon steel accounts for more than 70% of all steel produced in the world."
# https://worldstainless.org/wp-content/uploads/2024/07/ISSF_Stainless_Steel_in_Figures_2020_English_public_version.pdf
# - Figure 1，只给了不锈钢总量。和总钢铁产量比一比
# 比起crude steel的0.98来说，我们这边使用0.96，这个数字是基于EUROFER的那个估计的
FE_EQ_RATE_HOT_ROLLED_PROD = 96 / 100

SHEET_P5_M5_PATH = INPUT_WORKBOOK_PATH
SHEET_P5_M5_NAME = "热轧制产量信息_Worldsteel"

df_p5_m5_original = pd.read_excel(SHEET_P5_M5_PATH, sheet_name=SHEET_P5_M5_NAME)[
    ["Country_code_iso3166-2_alpha_3", f"{YEAR} hot rolled products"]
]
df_p5_m5_original

for idx, row in df_p5_m5_original.iterrows():
    # 跳过总计行
    if row["Country_code_iso3166-2_alpha_3"] == "Total":
        logger.info("跳过总计行")
        continue

    # 检查国家代号是否在已知国家代号列表中
    assert row["Country_code_iso3166-2_alpha_3"] in COUNTRY_CODES

    # 获取生产数据
    hot_rolled_prod = (
        row[f"{YEAR} hot rolled products"]
        if not pd.isna(row[f"{YEAR} hot rolled products"])
        else None
    )

    unit = "kg"
    entity_from = row["Country_code_iso3166-2_alpha_3"]
    entity_to = entity_from
    sector_from = "P5"
    sector_to = "M5"

    if hot_rolled_prod is not None:
        mass = hot_rolled_prod * 1000000 * FE_EQ_RATE_HOT_ROLLED_PROD
        extra = None
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    {
                        "entity_from": [entity_from],
                        "entity_to": [entity_to],
                        "sector_from": [sector_from],
                        "sector_to": [sector_to],
                        "mass": [mass],
                        "unit": [unit],
                        "extra": [extra],
                    }
                ),
            ],
            ignore_index=True,
        )

logger.info("P5-M5 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 20
# P5-M9，Forming（轧制、锻造等）Scrap流
# 基于P5-M5，通过转换率计算
FORMING_SCRAP_RATE = 9 / 100
FORMING_SCRAP_REVERSE = FORMING_SCRAP_RATE / (1 - FORMING_SCRAP_RATE)

df_p5_m9 = df_result[
    (df_result["sector_from"] == "P5") & (df_result["sector_to"] == "M5")
].copy()
df_p5_m9["sector_to"] = "M9"
df_p5_m9["mass"] *= FORMING_SCRAP_REVERSE

df_result = pd.concat([df_result, df_p5_m9], ignore_index=True)

logger.info("P5-M9 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 21
# P5-Environment，Forming的Scale（表层氧化铁）废弃流
# 基于P5-M5，通过转换率计算
SCALE_WASTE_RATE = 1 / 100
SCALE_WASTE_REVERSE = SCALE_WASTE_RATE / (1 - SCALE_WASTE_RATE)

df_p5_env = df_result[
    (df_result["sector_from"] == "P5") & (df_result["sector_to"] == "M5")
].copy()
df_p5_env["sector_to"] = "Environment"
df_p5_env["mass"] *= SCALE_WASTE_REVERSE

df_result = pd.concat([df_result, df_p5_env], ignore_index=True)

logger.info("P5-Environment 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 22
# M5国际贸易
trade_code_m5 = read_trade_code("Finished steel")
logger.info(f"成品钢贸易代码: {trade_code_m5}")
df_trade_m5 = read_trade_entries(YEAR, trade_code_m5)

logger.info(f"M5国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m5)

for idx, row in tqdm(df_trade_m5.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M5"
    sector_to = "M5"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [extra],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M5国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 23
# M5-M6
# 质量守恒，计算M5国际贸易后的净市场供应量
market_supply_dict = calculate_market_output(df_result, "M5")
logger.info("M5-M6 质量守恒计算完成，当前净市场供应量字典如下 👇")
logger.info(market_supply_dict)

for key in market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M5"
    sector_to = "M6"
    mass = market_supply_dict[key][0]
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M5-M6 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 24
# M6国际贸易
trade_code_m6 = read_trade_code("Products of steel")
logger.info(f"钢制品贸易代码: {trade_code_m6}")
df_trade_m6 = read_trade_entries(YEAR, trade_code_m6)

logger.info(f"M6国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m6)

added_entry_dict = {
    "entity_from": [],
    "entity_to": [],
    "sector_from": [],
    "sector_to": [],
    "mass": [],
    "unit": [],
    "extra": [],
}

for idx, row in tqdm(df_trade_m6.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M6"
    sector_to = "M6"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    added_entry_dict["entity_from"].append(entity_from)
    added_entry_dict["entity_to"].append(entity_to)
    added_entry_dict["sector_from"].append(sector_from)
    added_entry_dict["sector_to"].append(sector_to)
    added_entry_dict["mass"].append(mass)
    added_entry_dict["unit"].append(unit)
    added_entry_dict["extra"].append(extra)

df_result = pd.concat(
    [
        df_result,
        pd.DataFrame(added_entry_dict),
    ],
    ignore_index=True,
)

logger.info("M6国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 25
# M6-P6
# 基于Worldsteel数据的代码，数据源不是很准确，搁置
r'''# 基于Worldsteel数据完成
# 下为说明：
# - M6-P6在worldsteel的数据中叫做apparent steel use（ASU），以两种口径衡量：crude steel当量和steel products当量
# - 前者为基于forming loss等信息，将loss加回重量上，同时基于iron fe eq，转换成的对于原始crude steel的消耗量；
# - 后者为基于steel products的重量；
# - 我们选择前者，因为其相对容易把握铁当量信息。steel products数据的铁当量很难准确判断；
# - 转换公式：M6-P6 = ASU_crude_steel_equivalent * FE_EQ_CRUDE_STEEL * FORMING_LOSS
FE_EQ_CRUDE_STEEL_M6_P6 = FE_EQ_RATE_CRUDE_STEEL
FORMING_LOSS_M6_P6 = FORMING_SCRAP_RATE

SHEET_M6_P6_PATH = INPUT_WORKBOOK_PATH
SHEET_M6_P6_NAME = "表观钢铁消费量（钢铁当量）_Worldsteel"

df_m6_p6_original = pd.read_excel(SHEET_M6_P6_PATH, sheet_name=SHEET_M6_P6_NAME)[
    ["Country_code_iso3166-2_alpha_3", f"{YEAR} ASU - CSE"]
]
df_m6_p6_original

for idx, row in df_m6_p6_original.iterrows():
    # 跳过总计行
    if row["Country_code_iso3166-2_alpha_3"] == "Total":
        logger.info("跳过总计行")
        continue

    # 检查国家代号是否在已知国家代号列表中
    assert row["Country_code_iso3166-2_alpha_3"] in COUNTRY_CODES

    # 获取生产数据
    asu_cse = (
        row[f"{YEAR} ASU - CSE"] if not pd.isna(row[f"{YEAR} ASU - CSE"]) else None
    )

    unit = "kg"
    entity_from = row["Country_code_iso3166-2_alpha_3"]
    entity_to = entity_from
    sector_from = "M6"
    sector_to = "P6"

    if asu_cse is not None:
        mass = asu_cse * 1000000 * FE_EQ_CRUDE_STEEL_M6_P6 * (1 - FORMING_LOSS_M6_P6)
        extra = None
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    {
                        "entity_from": [entity_from],
                        "entity_to": [entity_to],
                        "sector_from": [sector_from],
                        "sector_to": [sector_to],
                        "mass": [mass],
                        "unit": [unit],
                        "extra": [extra],
                    }
                ),
            ],
            ignore_index=True,
        )

logger.info("M6-P6 数据处理完成，当前结果DataFrame如下 👇")
df_result'''

# 基于质量守恒的代码
market_supply_dict = calculate_market_output(df_result, "M6")
logger.info("M6-P6 质量守恒计算完成，当前净市场供应量字典如下 👇")
logger.info(market_supply_dict)

for key in market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M6"
    sector_to = "P6"
    mass = market_supply_dict[key][0]
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M6-P6 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 26
# P6-M7 & P6-M9
# 基于M6-P6与M4-P6，通过转换率计算
# 转化率的计算相对复杂，包括两个部分：首先需要将总量分配到四个子部门上，之后为每个子部门单独计算损失率和产率

# 部门比例分配表读取
SHEET_SECTOR_SPLIT_PATH = INPUT_WORKBOOK_PATH
SHEET_SECTOR_SPLIT_NAME = "国家最终产品生产比例"
_df_sector_split = pd.read_excel(
    SHEET_SECTOR_SPLIT_PATH, sheet_name=SHEET_SECTOR_SPLIT_NAME
)
display(_df_sector_split)


# 输入ISO_3166_2_A3代号和部门代号，读取对应的分配比例（单位1）
def get_sector_split(iso_3166_a3_code, sector_name):
    assert sector_name in ["Transportation", "Machinery", "Construction", "Others"]
    assert iso_3166_a3_code in COUNTRY_CODES
    # 检查是否有精准数据
    row_match = _df_sector_split[
        _df_sector_split["Country_code_iso3166-2_alpha_3"] == iso_3166_a3_code
    ]
    if len(row_match) == 1:
        # 有精准数据
        split_value = row_match[sector_name].values[0]
        assert not pd.isna(split_value)
        return split_value
    else:
        # 没有精准条目，使用全球平均
        logger.info(
            f"没有找到国家 {iso_3166_a3_code} 的精准部门分配数据，使用全球平均值代替"
        )
        row_global = _df_sector_split[
            _df_sector_split["Country_code_iso3166-2_alpha_3"] == "Other"
        ]
        assert len(row_global) == 1
        split_value = row_global[sector_name].values[0]
        assert not pd.isna(split_value)
        return split_value


# 各部门损失率和产率
MANUFACTURING_SCRAP_RATE = {
    "Transportation": 0.267,
    "Machinery": 0.171,
    "Construction": 0.074,
    "Others": 0.227,
}

# 1. 基于质量守恒，计算M4和M6的总输入量
market_supply_dict_p6 = calculate_market_output(df_result, "P6")
logger.info("P6 质量守恒计算完成，当前净供应量字典如下 👇")
logger.info(market_supply_dict_p6)

# 2. 计算P6的各输出流
for key in market_supply_dict_p6.keys():
    entity_from = key
    entity_to = key
    sector_from = "P6"
    mass_total = market_supply_dict_p6[key][0]
    unit = "kg"

    for sub_sector in MANUFACTURING_SCRAP_RATE.keys():
        scrap_rate = MANUFACTURING_SCRAP_RATE[sub_sector]
        sector_percentage = get_sector_split(entity_from, sub_sector)
        mass_to_m7 = mass_total * sector_percentage * (1 - scrap_rate)
        mass_to_m9 = mass_total * sector_percentage * scrap_rate
        # P6-M7
        sector_to = "M7"
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    {
                        "entity_from": [entity_from],
                        "entity_to": [entity_to],
                        "sector_from": [sector_from],
                        "sector_to": [sector_to],
                        "mass": [mass_to_m7],
                        "unit": [unit],
                        "extra": ["[End Product] " + sub_sector],
                    }
                ),
            ],
            ignore_index=True,
        )
        # P6-M9
        sector_to = "M9"
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    {
                        "entity_from": [entity_from],
                        "entity_to": [entity_to],
                        "sector_from": [sector_from],
                        "sector_to": [sector_to],
                        "mass": [mass_to_m9],
                        "unit": [unit],
                        "extra": ["[End Product] " + sub_sector],
                    }
                ),
            ],
            ignore_index=True,
        )

logger.info("P6-M7 & P6-M9 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 27
# M7国际贸易
trade_code_m7 = read_trade_code("Final Products_Parts")
logger.info(f"最终产品及零部件贸易代码: {trade_code_m7}")
df_trade_m7 = read_trade_entries(YEAR, trade_code_m7)

logger.info(f"M7国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m7)

added_entry_dict = {
    "entity_from": [],
    "entity_to": [],
    "sector_from": [],
    "sector_to": [],
    "mass": [],
    "unit": [],
    "extra": [],
}

for idx, row in tqdm(df_trade_m7.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M7"
    sector_to = "M7"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    added_entry_dict["entity_from"].append(entity_from)
    added_entry_dict["entity_to"].append(entity_to)
    added_entry_dict["sector_from"].append(sector_from)
    added_entry_dict["sector_to"].append(sector_to)
    added_entry_dict["mass"].append(mass)
    added_entry_dict["unit"].append(unit)
    added_entry_dict["extra"].append(extra)

df_result = pd.concat(
    [
        df_result,
        pd.DataFrame(added_entry_dict),
    ],
    ignore_index=True,
)

logger.info("M7国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 28
# M7-M8
# 质量守恒，计算M7国际贸易后的净市场供应量
market_supply_dict = calculate_market_output(df_result, "M7")
logger.info("M7-M8 质量守恒计算完成，当前净市场供应量字典如下 👇")
logger.info(market_supply_dict)

for key in market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M7"
    sector_to = "M8"
    mass = market_supply_dict[key][0]
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M7-M8 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 29
# M8国际贸易
trade_code_m8 = read_trade_code("Final Products")
logger.info(f"最终产品贸易代码: {trade_code_m8}")
df_trade_m8 = read_trade_entries(YEAR, trade_code_m8)

logger.info(f"M8国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m8)

added_entry_dict = {
    "entity_from": [],
    "entity_to": [],
    "sector_from": [],
    "sector_to": [],
    "mass": [],
    "unit": [],
    "extra": [],
}

for idx, row in tqdm(df_trade_m8.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M8"
    sector_to = "M8"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    added_entry_dict["entity_from"].append(entity_from)
    added_entry_dict["entity_to"].append(entity_to)
    added_entry_dict["sector_from"].append(sector_from)
    added_entry_dict["sector_to"].append(sector_to)
    added_entry_dict["mass"].append(mass)
    added_entry_dict["unit"].append(unit)
    added_entry_dict["extra"].append(extra)

df_result = pd.concat(
    [
        df_result,
        pd.DataFrame(added_entry_dict),
    ],
    ignore_index=True,
)

logger.info("M8国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 30
# M8-S1
# 质量守恒，计算M8国际贸易后的净市场供应量。
# 对日本等最终产品市场被贸易扣成零/负值的国家，使用WorldSteel ASU-FSP作为最终需求下限。
raw_market_supply_dict = calculate_market_output(
    df_result, "M8", need_negative_correction=False
)
asu_fsp_final_demand_dict = (
    {}
    if args.disable_m8_s1_asu_fsp_correction
    else read_worldsteel_asu_fsp_final_demand(YEAR)
)
m8_s1_correction_records = []
logger.info("M8-S1 原始质量守恒计算完成，当前净市场供应量字典如下 👇")
logger.info(raw_market_supply_dict)

for key in raw_market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M8"
    sector_to = "S1"
    raw_market_output = raw_market_supply_dict[key][0]
    mass_balance_output = max(raw_market_output, 0)
    asu_fsp_output = asu_fsp_final_demand_dict.get(key)
    correction_reason = None
    if (
        asu_fsp_output is not None
        and raw_market_output <= 0
        and asu_fsp_output > mass_balance_output
    ):
        mass = asu_fsp_output
        correction_reason = "ASU-FSP final demand floor for non-positive raw M8 balance"
        m8_s1_correction_records.append(
            {
                "entity": key,
                "raw_m8_net_supply_kg": raw_market_output,
                "mass_balance_m8_s1_kg": mass_balance_output,
                "asu_fsp_floor_m8_s1_kg": asu_fsp_output,
                "added_m8_s1_kg": asu_fsp_output - mass_balance_output,
                "raw_m8_inflow_kg": raw_market_supply_dict[key][1],
                "raw_m8_outflow_kg": raw_market_supply_dict[key][2],
                "reason": correction_reason,
            }
        )
        logger.warning(
            f"国家 {key} 的 M8 原始净供应为 {raw_market_output} kg，"
            f"使用 WorldSteel ASU-FSP 将 M8-S1 修正为 {asu_fsp_output} kg。"
        )
    else:
        mass = mass_balance_output
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [correction_reason],
                }
            ),
        ],
        ignore_index=True,
    )

if m8_s1_correction_records:
    df_m8_s1_corrections = pd.DataFrame(m8_s1_correction_records).sort_values(
        "added_m8_s1_kg", ascending=False
    )
    correction_report_path = OUTPUT_DIR / f"{YEAR}_m8_s1_asu_fsp_corrections.csv"
    df_m8_s1_corrections.to_csv(correction_report_path, index=False)
    logger.warning(
        f"M8-S1 ASU-FSP 修正涉及 {len(df_m8_s1_corrections)} 个国家，"
        f"修正清单已导出至 {correction_report_path}"
    )
else:
    logger.info("M8-S1 未触发 ASU-FSP 修正。")

logger.info("M8-S1 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 31
# M9-P3
# 质量守恒；基于P3-M3、M2-P3、P3-Environment等一系列流，计算Scrap的输入量
# 注意禁止负数改正
market_supply_dict = calculate_market_output(
    df_result, "P3", need_negative_correction=False
)
logger.info("P3 质量守恒计算完成，当前净供应量字典如下 👇")
logger.info(market_supply_dict)

for key in market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "M9"
    sector_to = "P3"
    mass = -market_supply_dict[key][0]  # 取负值
    if mass < 0:
        mass = 0
        logger.info(
            f"国家 {key} 的 M9-P3 流为负值，已被修正为0。这是数据误差引起的正常现象。"
        )
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("M9-P3 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 32
# M9国际贸易
trade_code_m9 = read_trade_code("Scrap")
logger.info(f"废钢贸易代码: {trade_code_m9}")
df_trade_m9 = read_trade_entries(YEAR, trade_code_m9)

logger.info(f"M9国际贸易数据筛选完成，当前筛选后数据如下 👇")
display(df_trade_m9)

added_entry_dict = {
    "entity_from": [],
    "entity_to": [],
    "sector_from": [],
    "sector_to": [],
    "mass": [],
    "unit": [],
    "extra": [],
}

for idx, row in tqdm(df_trade_m9.iterrows()):
    entity_from = row["partner_iso_a3"]
    entity_to = row["reporter_iso_a3"]
    sector_from = "M9"
    sector_to = "M9"
    mass = row["iron_eq_weight_unified_kg"]
    unit = "kg"
    extra = f'TRADE_SITC_{row["commodity_code"]}'
    added_entry_dict["entity_from"].append(entity_from)
    added_entry_dict["entity_to"].append(entity_to)
    added_entry_dict["sector_from"].append(sector_from)
    added_entry_dict["sector_to"].append(sector_to)
    added_entry_dict["mass"].append(mass)
    added_entry_dict["unit"].append(unit)
    added_entry_dict["extra"].append(extra)

df_result = pd.concat(
    [
        df_result,
        pd.DataFrame(added_entry_dict),
    ],
    ignore_index=True,
)

logger.info("M9国际贸易数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 33
# P7-M9
# 质量守恒，基于M9的一系列流，计算Old Scrap的输入量
# 注意禁止负数改正
market_supply_dict = calculate_market_output(
    df_result, "M9", need_negative_correction=False
)
logger.info("M9 质量守恒计算完成，当前净供应量字典如下 👇")
logger.info(market_supply_dict)

for key in market_supply_dict.keys():
    entity_from = key
    entity_to = key
    sector_from = "P7"
    sector_to = "M9"
    mass = -market_supply_dict[key][0]  # 取负值
    if mass < 0:
        mass = 0
        logger.info(
            f"国家 {key} 的 P7-M9 流为负值，已被修正为0。这是数据误差引起的正常现象。"
        )
    unit = "kg"
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                {
                    "entity_from": [entity_from],
                    "entity_to": [entity_to],
                    "sector_from": [sector_from],
                    "sector_to": [sector_to],
                    "mass": [mass],
                    "unit": [unit],
                    "extra": [None],
                }
            ),
        ],
        ignore_index=True,
    )

logger.info("P7-M9 数据处理完成，当前结果DataFrame如下 👇")
df_result

# %% cell 34
# Process balance修正
# Process balance（PB）：对于P系列sector的流量输入、输出不平衡问题，令每个P的流量达到最大，将不平衡问题推至M（并结合后续的phantom flow修正）

# 映射字典，包括我们需要处理的P部门、输入M部门、输出M部门
process_balance_map = {
    "P2": {"from": "M1", "to": "M2"},
    "P3": {"from": "M2", "to": "M3"},
    "P4": {"from": "M2", "to": "M4"},
    "P5": {"from": "M3", "to": "M5"},
    "P6": {"from": "M6", "to": "M7"},
}

for p_sector in process_balance_map.keys():
    net_inflow_dict = calculate_market_output(
        df_result, p_sector, need_negative_correction=False
    )
    for key in tqdm(net_inflow_dict.keys()):
        net_inflow = net_inflow_dict[key][0]
        if net_inflow > 0:
            if p_sector == "P2":
                # 如果M1-P2的输入量大于P2的总输出量，令M1-P2的输入量等于P2的总输出量
                # 也即，我们将原本由P2-M2承担的不平衡部分，转移到M1-P2上
                # 主要原因：由于铁矿品位千差万别，M1的质量守恒结果要远逊于其它上中游产品，且iron making的统计数据相对十分可靠
                logger.info(
                    f"国家 {key} 的 P2 部门存在净流入 {net_inflow} kg，已将其转移至 M1-P2 流中以修正 Process Balance 问题。"
                )
                # 查找(国家,部门来源M1,部门去向P2)的所有流
                mask = (
                    (df_result["entity_from"] == key)
                    & (df_result["entity_to"] == key)
                    & (df_result["sector_from"] == "M1")
                    & (df_result["sector_to"] == "P2")
                )
                # 检查行数是否唯一
                assert (
                    mask.sum() == 1
                ), f"多个M1-P2流在国家 {key} 中被找到，无法唯一确定流量进行修正，请检查数据。"
                # 修正
                df_result.loc[mask, "mass"] -= net_inflow
                # 检查修正后是否为负值
                assert (
                    df_result.loc[mask, "mass"].values[0] >= 0
                ), f"国家 {key} 的 M1-P2 流在 Process Balance 修正后出现负值，请检查数据。"
                # 完成修正，直接进入下一迭代
                continue
            # P部门净流入，说明输入大于输出，增加输出
            entity_from = key
            entity_to = key
            sector_from = p_sector
            sector_to = process_balance_map[p_sector]["to"]
            mass = net_inflow
            unit = "kg"
            df_result = pd.concat(
                [
                    df_result,
                    pd.DataFrame(
                        {
                            "entity_from": [entity_from],
                            "entity_to": [entity_to],
                            "sector_from": [sector_from],
                            "sector_to": [sector_to],
                            "mass": [mass],
                            "unit": [unit],
                            "extra": ["Process Balance Adjustment"],
                        }
                    ),
                ],
                ignore_index=True,
            )
        elif net_inflow < 0:
            # P部门净流出，说明输出大于输入，增加输入
            entity_from = key
            entity_to = key
            sector_from = process_balance_map[p_sector]["from"]
            sector_to = p_sector
            mass = -net_inflow
            unit = "kg"
            df_result = pd.concat(
                [
                    df_result,
                    pd.DataFrame(
                        {
                            "entity_from": [entity_from],
                            "entity_to": [entity_to],
                            "sector_from": [sector_from],
                            "sector_to": [sector_to],
                            "mass": [mass],
                            "unit": [unit],
                            "extra": ["Process Balance Adjustment"],
                        }
                    ),
                ],
                ignore_index=True,
            )

logger.info("Process Balance 修正完成，当前结果DataFrame如下 👇")
df_result

# %% cell 35
# Phatom flow改正
# Phantom flow (PF)：由于数据不一致性和误差等问题，引起某一process的输入、输出不相等时，弥补该误差所引入的改正项
all_sectors = sorted(
    set(
        df_result["sector_from"].unique().tolist()
        + df_result["sector_to"].unique().tolist()
    )
)
# 手动删除P1、Environment、S1、P7四个在系统定义图中与未量化流相连的部门
all_sectors.remove("P1")
all_sectors.remove("Environment")
all_sectors.remove("S1")
all_sectors.remove("P7")

# 删除所有P过程：P过程不需要phantom flow，已经在前面改正
all_sectors = [s for s in all_sectors if not s.startswith("P")]

# 遍历所有的sector，计算净流入量，根据结果改正每个国家的PF
for sector in all_sectors:
    net_inflow_dict = calculate_market_output(
        df_result, sector, need_negative_correction=False
    )
    for key in tqdm(net_inflow_dict.keys()):
        net_inflow = net_inflow_dict[key][0]
        if net_inflow > 0:
            # 净流入，说明输入大于输出，增加一个PF输出
            entity_from = key
            entity_to = key
            sector_from = sector
            sector_to = "Phantom Flow"
            mass = net_inflow
            unit = "kg"
            extra = "Phantom Flow Correction"
            df_result = pd.concat(
                [
                    df_result,
                    pd.DataFrame(
                        {
                            "entity_from": [entity_from],
                            "entity_to": [entity_to],
                            "sector_from": [sector_from],
                            "sector_to": [sector_to],
                            "mass": [mass],
                            "unit": [unit],
                            "extra": [extra],
                        }
                    ),
                ],
                ignore_index=True,
            )
        elif net_inflow < 0:
            # 净流出，说明输出大于输入，增加一个PF输入
            entity_from = key
            entity_to = key
            sector_from = "Phantom Flow"
            sector_to = sector
            mass = -net_inflow  # 取负值
            unit = "kg"
            extra = "Phantom Flow Correction"
            df_result = pd.concat(
                [
                    df_result,
                    pd.DataFrame(
                        {
                            "entity_from": [entity_from],
                            "entity_to": [entity_to],
                            "sector_from": [sector_from],
                            "sector_to": [sector_to],
                            "mass": [mass],
                            "unit": [unit],
                            "extra": [extra],
                        }
                    ),
                ],
                ignore_index=True,
            )
        else:
            # 净流量为0，不需要改正
            pass

logger.info("Phantom Flow 改正完成，当前结果DataFrame如下 👇")
df_result

# %% cell 36
# 最终结果检查
# 检查所有sector的质量守恒情况

# 1e-3 kg = 1 g，允许1克的误差
# 请把SMALL_VALUE值设的大一些，本身MFA的数值非常大，这里如果比较小的话，可能会因为浮点数精度问题导致报错
SMALL_VALUE = 1e-3

final_all_sectors = all_sectors
for sector in final_all_sectors:
    net_inflow_dict = calculate_market_output(df_result, sector)
    for key in net_inflow_dict.keys():
        net_inflow = net_inflow_dict[key][0]
        assert (
            abs(net_inflow) < SMALL_VALUE
        ), f"国家 {key} 的 {sector} 部门质量守恒检查未通过，净流量为 {net_inflow} kg"

# %% cell 37
import os

# 导出数据
OUTPUT_PARQUET_PATH = str(OUTPUT_DIR / f"{YEAR}.parquet")
if os.path.exists(OUTPUT_PARQUET_PATH):
    backup_dir = OUTPUT_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / (
        f"{YEAR}_before_overwrite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    shutil.copy2(OUTPUT_PARQUET_PATH, backup_path)
    logger.info(f"已有输出已备份至 {backup_path}")
    os.remove(OUTPUT_PARQUET_PATH)
df_result.to_parquet(OUTPUT_PARQUET_PATH, index=False)
logger.success(f"数据已导出至 {OUTPUT_PARQUET_PATH} 🎉")

if args.receipt_path is not None:
    schema_validation = validate_mfa_schema(df_result)
    node_balance = calculate_node_balance(df_result)
    required_node_balance = node_balance.loc[~node_balance["is_boundary"]]
    failed_node_count = int((~required_node_balance["is_balanced"]).sum())
    max_node_imbalance_kg = float(
        required_node_balance["abs_imbalance_kg"].max()
        if not required_node_balance.empty
        else 0.0
    )
    # Exported sources are identified by content, not a private Git history.
    code_commit = "not-applicable-exported-source"
    builder_source_path = Path(__file__).resolve()
    builder_source_bytes = builder_source_path.read_bytes()
    builder_source_lf_sha256 = hashlib.sha256(
        builder_source_bytes.replace(b"\r\n", b"\n")
    ).hexdigest()
    checks = {
        **schema_validation["checks"],
        "country_sector_node_balance": failed_node_count == 0,
        "builder_source_present_and_hashed": bool(builder_source_bytes),
    }
    output_path = Path(OUTPUT_PARQUET_PATH).resolve()
    receipt_payload = {
        "stage": "03_mfa_candidate_build",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "code_commit": code_commit,
        "input_paths": [
            str(builder_source_path),
            str(Path(INPUT_WORKBOOK_PATH).resolve()),
            str(Path(TRADE_DATA_PATH).resolve()),
        ],
        "input_hashes": {
            "builder_source_sha256": sha256_file(builder_source_path),
            "builder_source_lf_sha256": builder_source_lf_sha256,
            "workbook_sha256": sha256_file(Path(INPUT_WORKBOOK_PATH)),
            "trade_data_sha256": sha256_file(Path(TRADE_DATA_PATH)),
        },
        "parameters": {
            "year": YEAR,
            "m8_s1_asu_fsp_correction_enabled": not args.disable_m8_s1_asu_fsp_correction,
            "pooled_inventory_interpretation": "unchanged",
            "node_balance_tolerance_kg": 1e-3,
        },
        "row_counts": {
            "trade_input_rows": len(DF_TRADE),
            "mfa_output_rows": len(df_result),
        },
        "coverage": {
            "bilateral_material_count": len(schema_validation["bilateral_balance"]),
            "country_sector_node_count": len(node_balance),
            "required_balanced_node_count": len(required_node_balance),
            "failed_required_node_count": failed_node_count,
            "max_required_node_imbalance_kg": max_node_imbalance_kg,
        },
        "output_paths": [str(output_path)],
        "output_hashes": {"mfa_sha256": sha256_file(output_path)},
        "checks": checks,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_receipt(args.receipt_path, receipt_payload)
    logger.info(f"构建验证 receipt 已写入 {args.receipt_path}")
    if receipt_payload["status"] != "PASS":
        raise RuntimeError(f"MFA validation failed; see receipt {args.receipt_path}")
