"""Run one complete paper ABM scenario.

Original source SHA256: ea5cd84d16bc81b3d857a3b89b44052c69dd3a5e2a63cbab00b7c49bc351b765

The matching and agent-state logic is retained from the final server simulation script.
"""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
import json
import math
from pathlib import Path
import sys

ROOT = bundle_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steel_supply.hazard_dynamic.hazard_calendar import (
    align_2023_baseline,
    date_from_day_of_year,
    iter_dates,
)
from steel_supply.hazard_dynamic.simulation_runtime import (
    DeterministicIdFactory,
    SimulationStep,
    build_random_blacklists,
    build_step_schedule,
)
from steel_supply.hazard_dynamic.abm_step_progress import (
    atomic_write_parquet,
    publish_completed_step,
)

parser = argparse.ArgumentParser(description="Run one global steel supply-chain stress-test scenario.")
parser.add_argument("--hazard-type", "--HAZARD_TYPE", dest="HAZARD_TYPE", required=True)
parser.add_argument("--climate-scenario", "--CLIMATE_SCENARIO", dest="CLIMATE_SCENARIO", required=True)
parser.add_argument("--trade-scenario", "--TRADE_SCENARIO", dest="TRADE_SCENARIO", required=True)
parser.add_argument("--random-disruption-ratio", "--RANDOM_DISRUPTION_RATIO", dest="RANDOM_DISRUPTION_RATIO", type=float)
parser.add_argument("--inventory-to-capacity-ratio", "--INVENTORY_TO_CAPACITY_RATIO", dest="INVENTORY_TO_CAPACITY_RATIO", type=float, required=True)
parser.add_argument("--capacity-to-initial-flow", "--CAPACITY_TO_INITIAL_FLOW", dest="CAPACITY_TO_INITIAL_FLOW", type=float, required=True)
parser.add_argument("--mfa-path", default=data_path('results/mfa/2023.parquet'))
parser.add_argument(
    "--hazard-data-dir",
    default=data_path('results/hazard_country_level_loss'),
)
parser.add_argument(
    "--output-dir",
    default=data_path('results/simulation'),
)
parser.add_argument("--start-year", type=int, default=2023)
parser.add_argument("--end-year", type=int, default=2100)
parser.add_argument("--step-days", type=int, default=7)
parser.add_argument("--fair-match-chunks", type=int, default=100)
parser.add_argument("--random-seed", type=int)
parser.add_argument(
    "--terminal-step-policy",
    choices=("complete", "legacy-floor"),
    default="complete",
)
args = parser.parse_args()

if args.TRADE_SCENARIO == "random-disruption" and args.RANDOM_DISRUPTION_RATIO is None:
    parser.error("--random-disruption-ratio is required for random-disruption")
if args.TRADE_SCENARIO == "random-disruption" and (
    not math.isfinite(args.RANDOM_DISRUPTION_RATIO)
    or not 0.0 <= args.RANDOM_DISRUPTION_RATIO <= 1.0
):
    parser.error("--random-disruption-ratio must be finite and within [0, 1]")
if args.TRADE_SCENARIO != "random-disruption" and args.RANDOM_DISRUPTION_RATIO is not None:
    parser.error("--random-disruption-ratio is only valid for random-disruption")
if args.TRADE_SCENARIO == "random-disruption" and args.random_seed is None:
    parser.error("--random-seed is required for random-disruption")
if args.TRADE_SCENARIO != "random-disruption" and args.random_seed is not None:
    parser.error("--random-seed is only valid for random-disruption")
if args.start_year > args.end_year:
    parser.error("--start-year must not exceed --end-year")
if not 0 < args.step_days <= 366:
    parser.error("--step-days must be in [1, 366]")

PARAMS = {
    "HAZARD_TYPE": args.HAZARD_TYPE,
    "CLIMATE_SCENARIO": args.CLIMATE_SCENARIO,
    "TRADE_SCENARIO": args.TRADE_SCENARIO,
    "RANDOM_DISRUPTION_RATIO": args.RANDOM_DISRUPTION_RATIO,
    "INVENTORY_TO_CAPACITY_RATIO": args.INVENTORY_TO_CAPACITY_RATIO,
    "CAPACITY_TO_INITIAL_FLOW": args.CAPACITY_TO_INITIAL_FLOW,
    "MFA_PATH": str(Path(args.mfa_path).resolve()),
    "HAZARD_DATA_DIR": str(Path(args.hazard_data_dir).resolve()),
    "OUTPUT_DIR": str(Path(args.output_dir).resolve()),
    "START_YEAR": args.start_year,
    "END_YEAR": args.end_year,
    "STEP_DAYS": args.step_days,
    "FAIR_MATCH_CHUNKS": args.fair_match_chunks,
    "RANDOM_SEED": args.random_seed,
    "TERMINAL_STEP_POLICY": args.terminal_step_policy,
}

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: mfa_paper_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 勘误表
#
# 全部 Inherit 应为 inherent
#

# %% [markdown]
# ## 新版模拟算法
#
# ### 约束条件
#
# #### 总流量、幻影流与真实流量
#
# - ![image-2.png](attachment:image-2.png)
#
# 附注
#
# - 之前的幻影流随真实流倍增倍减，但可能产生过多虚空产能（尤其是在不同国家的 phantom flow 比例差别极大时，可能通过贸易关系倍增数十倍），因此改为恒定值
# - 改进方向：Phantom Flow 可以与世界总生产量呈比例关系，随世界总体生产情况增减（暂未实现）
# - 幻影流 Phantom Flow 随真实主产品的 Primary Market 一起对单，不必也不能区别对待
#
# #### 具有多种产品输出的 P 过程
#
# - P 过程只能有一种主产品，进入 Primary Market，遵循供需匹配原则
# - 其他产品均为副产品，进入 Subsidiary Market，不受需求影响
#
# #### 具有多种产品输入的 P 过程
#
# - 产品替代性
#   - 一般情况下，假设所有输入品完全替代
#   - 部分特殊情况下，产品不可互相替代
#     - 在需求单创建时，按照初始库存比例，生成两个或多个产品的订单
# - 优先使用 Subsidiary Market 的产品
#   - 注意 Subsidiary Market 和 P 的执行顺序：任意具有主副产品替代关系的产品对，副产品必须全部生产完毕后，主产品才可对单
#
# ### 算法步骤
#
# - 参数初始化
#   - 清空上一轮的订单
#   - 恢复 P 的初始状态
#     - 恢复 P 的剩余产能
# - 按照顺序，从后往前遍历产业链
#   - P 生成主产品最大供给单，加入市场
#   - P 检查库存，生成需求单，加入市场
#   - M 幻影流修正：幻影流订单创建
#   - 国内对单、生产与库存管理（包括幻影流）
#     - 副产品优先
#     - 主产品其次
#   - 国际对单、生产与库存管理（包括幻影流）
#     - 副产品优先
#     - 主产品其次
#

# %% [markdown]
# ## 初始化
#

# %%
from typing import cast
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass
from uuid import uuid4
from copy import deepcopy
from ortools.graph.python import min_cost_flow
from ortools.graph.python import max_flow
import os
from datetime import date, timedelta
import glob
import shutil
import gc
import random

# %%
# 基础运算参数
SMALL_VALUE: float = 1e-3  # 可以等效视作0的小量
MFA_YEAR: int = 2023  # MFA系统年份

# 模拟系统参数
STEP_DAYS = PARAMS["STEP_DAYS"]  # 每个模拟步长的天数

CAPACITY_TO_INITIAL_FLOW: float = PARAMS["CAPACITY_TO_INITIAL_FLOW"]
P7_CAPACITY_TO_INITIAL_FLOW: float = 1  # P7部门产能与初始流量的比值（P7是存量废料生产部门，产能无法扩增）
INVENTORY_TO_CAPACITY_RATIO: float = PARAMS["INVENTORY_TO_CAPACITY_RATIO"]

FAIR_MATCH_CHUNKS = PARAMS["FAIR_MATCH_CHUNKS"]  # 公平匹配时的分块数，数值越大，匹配越精细，但计算时间越长。chunk=100时，大约10分钟完成一年的模拟

START_YEAR: int = PARAMS["START_YEAR"]  # 模拟起始年份
END_YEAR: int = PARAMS["END_YEAR"]  # 模拟结束年份
START_DATE: date = date(START_YEAR, 1, 1)
END_DATE: date = date(END_YEAR, 12, 31)
total_days: int = (END_DATE - START_DATE).days + 1
TERMINAL_STEP_POLICY: str = PARAMS["TERMINAL_STEP_POLICY"]
SIMULATION_SCHEDULE: list[SimulationStep] = build_step_schedule(
    START_DATE,
    END_DATE,
    STEP_DAYS,
    complete_terminal_step=TERMINAL_STEP_POLICY == "complete",
)
TOTAL_STEPS: int = len(SIMULATION_SCHEDULE)
logger.info(f"Simulating from {START_DATE} to {END_DATE}")
logger.info(f"Total Duration: {total_days} days | Total Steps: {TOTAL_STEPS}")

INDUSTRIAL_CHAIN_ORDER: list[list[str]] = [
    ["S1"],
    ["VP2"],
    ["P6"],
    ["VP1"],
    ["P5"],
    ["P4"],
    ["P3"],
    ["P7", "P2"],
    ["P1"],
]  # 生产链部门顺序。从前向后，同一列表内的部门同时生产
PHANTOM_PRODUCE_RELATION: dict[str, list[str]] = {  # Phantom Flow流和P部门的生产关系；哪些幻影流必须在P部门调用后、对单前生产
    # 思路：确保生成真实流的供给/需求单后，在下次对单前，供给/需求幻影流也同样生成
    "VP2": ["M8"],
    "P6": ["M7"],
    "VP1": ["M6"],
    "P5": ["M5", "M9"],
    "P3": ["M3"],
    "P4": ["M4"],
    "P2": ["M2"],
    "P1": ["M1"],
}

# 优先对单部门；解决恒定流被可变流挤压的问题。详见Demand和Supply类描述
PRIORITIZED_DEMAND_SECTORS: list[str] = ["Phantom Flow"]
PRIORITIZED_SUPPLY_SECTORS: list[str] = ["Phantom Flow"]

DF_MFA_PATH: str = PARAMS["MFA_PATH"]


# %%
class MfaSystemSchema(pa.DataFrameModel):
    entity_from: Series[str]
    entity_to: Series[str]
    sector_from: Series[str]
    sector_to: Series[str]
    mass: Series[float]
    unit: Series[str]


@pa.check_types
def load_mfa_system() -> pd.DataFrame:
    return pd.read_parquet(DF_MFA_PATH)  # type: ignore


DF_MFA_SYSTEM: pd.DataFrame = load_mfa_system()

# %% [markdown]
# ## 基于 MFA 系统，计算参数
#

# %%
# ======================================================================
# 国家代号列表读取
# ======================================================================
COUNTRY_ALL: list[str] = sorted(
    set(DF_MFA_SYSTEM["entity_from"].unique())
    | set(DF_MFA_SYSTEM["entity_to"].unique())
)


# ======================================================================
# P部门参数构建
# ======================================================================


# 单个P对象
@dataclass
class P:
    Sector: str
    Country: str
    InheritCapacity: float  # 内在产能，不受灾害等胁迫影响下的最大生产量，以总流量计
    Capacity: float  # 产能，以总流量计
    CapacityLeft: float  # 剩余产能，以总流量计
    Inventory: float  # 库存量，以总流量计
    InventoryTargetToCapacityRatio: float  # 库存目标与产能比，以1为单位.例如，300%库存目标，则为3
    RawMaterialInputRatio: dict[str, float]  # 原材料输入量占比，以1为单位。与初始总流量比值（例如，初始输入如果小于输出，输入产品占比之和小于1）
    ProductOutputRatio: dict[str, float]  # 产品产量占比，以1为单位
    MainProduct: str  # 主产品
    IsMaterialInputSubstitutable: bool  # 原材料输入是否可互相完全替代（一般可以，P6不行）


# P部门参数总表
P_SECTOR_ALL: dict[
    str,  # 部门
    dict[
        str,  # 国家
        P,  # P对象
    ],
] = {}

# 构建P部门参数总表
## 真实P部门
REAL_P_SECTORS: list[str] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
MAIN_PRODUCTS: dict[str, str] = {"P1": "M1", "P2": "M2", "P3": "M3", "P4": "M4", "P5": "M5", "P6": "M7", "P7": "M9"}
for p_sector in REAL_P_SECTORS:
    if p_sector not in P_SECTOR_ALL:
        P_SECTOR_ALL[p_sector] = {}
    for country in tqdm(COUNTRY_ALL):
        # 计算P对象字段，添加至字典
        inflow_total: float = DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_to"] == country) & (DF_MFA_SYSTEM["sector_to"] == p_sector)]["mass"].sum()
        inflow_total = 0 if pd.isna(inflow_total) else inflow_total
        outflow_total: float = DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_from"] == country) & (DF_MFA_SYSTEM["sector_from"] == p_sector)]["mass"].sum()
        outflow_total = 0 if pd.isna(outflow_total) else outflow_total
        if (inflow_total < SMALL_VALUE) and (outflow_total < SMALL_VALUE):
            continue
        total_flow: float = max(inflow_total, outflow_total)
        if p_sector == "P7":
            capacity_initial: float = P7_CAPACITY_TO_INITIAL_FLOW * total_flow
        else:
            capacity_initial: float = CAPACITY_TO_INITIAL_FLOW * total_flow
        capacity_left_initial: float = capacity_initial
        inventory_initial: float = INVENTORY_TO_CAPACITY_RATIO * capacity_initial
        inventory_target_to_capacity_ratio: float = INVENTORY_TO_CAPACITY_RATIO
        raw_material_input_ratio: dict[str, float] = {}
        for commodity in sorted(DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_to"] == country) & (DF_MFA_SYSTEM["sector_to"] == p_sector)]["sector_from"].unique()):
            inflow_commodity: float = cast(
                float,
                DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_to"] == country) & (DF_MFA_SYSTEM["sector_to"] == p_sector) & (DF_MFA_SYSTEM["sector_from"] == commodity)][
                    "mass"
                ].sum(),  # type: ignore
            )
            inflow_commodity = 0 if pd.isna(inflow_commodity) else inflow_commodity
            if inflow_commodity > SMALL_VALUE:
                raw_material_input_ratio[commodity] = inflow_commodity / total_flow
        product_output_ratio: dict[str, float] = {}
        for commodity in sorted(DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_from"] == country) & (DF_MFA_SYSTEM["sector_from"] == p_sector)]["sector_to"].unique()):
            outflow_commodity: float = cast(
                float,
                DF_MFA_SYSTEM[
                    (DF_MFA_SYSTEM["entity_from"] == country) & (DF_MFA_SYSTEM["sector_from"] == p_sector) & (DF_MFA_SYSTEM["sector_to"] == commodity)
                ][
                    "mass"
                ].sum(),  # type: ignore
            )
            outflow_commodity = 0 if pd.isna(outflow_commodity) else outflow_commodity
            if outflow_commodity > SMALL_VALUE:
                product_output_ratio[commodity] = outflow_commodity / total_flow
        main_product: str = MAIN_PRODUCTS[p_sector]
        P_SECTOR_ALL[p_sector][country] = P(
            Sector=p_sector,
            Country=country,
            InheritCapacity=capacity_initial,
            Capacity=capacity_initial,
            CapacityLeft=capacity_left_initial,
            Inventory=inventory_initial,
            InventoryTargetToCapacityRatio=inventory_target_to_capacity_ratio,
            RawMaterialInputRatio=raw_material_input_ratio,
            ProductOutputRatio=product_output_ratio,
            MainProduct=main_product,
            IsMaterialInputSubstitutable=(p_sector != "P6"),
        )
## 虚拟P部门
VIRTUAL_P_SECTORS: list[str] = ["VP1", "VP2"]
VIRTUAL_P_SECTORS_FROM_TO: dict[str, tuple[str, str]] = {  # [部门ID,[来源商品部门ID,去向商品部门ID]]
    "VP1": ("M5", "M6"),
    "VP2": ("M7", "M8"),
}
for p_sector in VIRTUAL_P_SECTORS:
    if p_sector not in P_SECTOR_ALL:
        P_SECTOR_ALL[p_sector] = {}
    for country in tqdm(COUNTRY_ALL):
        # 计算P对象字段，添加至字典
        from_commodity: str = VIRTUAL_P_SECTORS_FROM_TO[p_sector][0]
        to_commodity: str = VIRTUAL_P_SECTORS_FROM_TO[p_sector][1]
        total_flow: float = DF_MFA_SYSTEM[
            (DF_MFA_SYSTEM["entity_from"] == country)
            & (DF_MFA_SYSTEM["entity_to"] == country)
            & (DF_MFA_SYSTEM["sector_from"] == from_commodity)
            & (DF_MFA_SYSTEM["sector_to"] == to_commodity)
        ]["mass"].sum()
        total_flow = 0 if pd.isna(total_flow) else total_flow
        if total_flow < SMALL_VALUE:
            continue
        capacity_initial: float = CAPACITY_TO_INITIAL_FLOW * total_flow
        capacity_left_initial: float = capacity_initial
        inventory_initial: float = INVENTORY_TO_CAPACITY_RATIO * capacity_initial
        inventory_target_to_capacity_ratio: float = INVENTORY_TO_CAPACITY_RATIO
        raw_material_input_ratio: dict[str, float] = {from_commodity: 1.0}
        product_output_ratio: dict[str, float] = {to_commodity: 1.0}
        main_product: str = to_commodity
        P_SECTOR_ALL[p_sector][country] = P(
            Sector=p_sector,
            Country=country,
            InheritCapacity=capacity_initial,
            Capacity=capacity_initial,
            CapacityLeft=capacity_left_initial,
            Inventory=inventory_initial,
            InventoryTargetToCapacityRatio=inventory_target_to_capacity_ratio,
            RawMaterialInputRatio=raw_material_input_ratio,
            ProductOutputRatio=product_output_ratio,
            MainProduct=main_product,
            IsMaterialInputSubstitutable=True,
        )

# ======================================================================
# M部门参数构建
# ======================================================================


# 单个M部门对象
@dataclass
class M:
    Commodity: str
    Country: str
    InflowRatio: dict[str, float]  # M部门的输入占比，以1为单位。包括Phantom Flow部门。输入和输出不相等的情况下，可能相加小于1
    OutflowRatio: dict[str, float]  # M部门的输出占比，以1为单位。包括Phantom Flow部门。输入和输出不相等的情况下，可能相加小于1
    InflowMasses: dict[str, float]  # M部门的输入重量
    OutflowMasses: dict[str, float]  # M部门的输出重量


M_SECTOR_ALL: dict[
    str,  # 商品部门
    dict[
        str,  # 国家
        M,  # M对象
    ],
] = {}

# 构建M部门参数总表
M_SECTORS: list[str] = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
M_SECTOR_VP_FIX: dict[str, dict[str, str]] = {
    "M5": {"from": "M6", "to": "VP1"},  # M5到M6的国内流，改成到VP1的流
    "M6": {"from": "M5", "to": "VP1"},
    "M7": {"from": "M8", "to": "VP2"},  # M7到M8的国内流，改成到VP2的流
    "M8": {"from": "M7", "to": "VP2"},
}
for m_sector in M_SECTORS:
    if m_sector not in M_SECTOR_ALL:
        M_SECTOR_ALL[m_sector] = {}
    for country in tqdm(COUNTRY_ALL):
        # 计算M对象字段，添加至字典
        inflow_total: float = DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_to"] == country) & (DF_MFA_SYSTEM["sector_to"] == m_sector)]["mass"].sum()
        inflow_total = 0 if pd.isna(inflow_total) else inflow_total
        outflow_total: float = DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_from"] == country) & (DF_MFA_SYSTEM["sector_from"] == m_sector)]["mass"].sum()
        outflow_total = 0 if pd.isna(outflow_total) else outflow_total
        if (inflow_total < SMALL_VALUE) and (outflow_total < SMALL_VALUE):
            continue
        total_flow: float = max(inflow_total, outflow_total)
        inflow_ratio: dict[str, float] = {}
        inflow_masses: dict[str, float] = {}
        for sector_from in sorted(DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_to"] == country) & (DF_MFA_SYSTEM["sector_to"] == m_sector)]["sector_from"].unique()):
            inflow_commodity: float = cast(
                float,
                DF_MFA_SYSTEM[
                    (DF_MFA_SYSTEM["entity_to"] == country) & (DF_MFA_SYSTEM["sector_to"] == m_sector) & (DF_MFA_SYSTEM["sector_from"] == sector_from)
                ][
                    "mass"
                ].sum(),  # type: ignore
            )
            inflow_commodity = 0 if pd.isna(inflow_commodity) else inflow_commodity
            if inflow_commodity > SMALL_VALUE:
                # VP修正
                if (m_sector in M_SECTOR_VP_FIX) and (sector_from == M_SECTOR_VP_FIX[m_sector]["from"]):
                    sector_from: str = M_SECTOR_VP_FIX[m_sector]["to"]
                inflow_ratio[sector_from] = inflow_commodity / total_flow
                inflow_masses[sector_from] = inflow_commodity
        outflow_ratio: dict[str, float] = {}
        outflow_masses: dict[str, float] = {}
        for sector_to in sorted(DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_from"] == country) & (DF_MFA_SYSTEM["sector_from"] == m_sector)]["sector_to"].unique()):
            outflow_commodity: float = cast(
                float,
                DF_MFA_SYSTEM[
                    (DF_MFA_SYSTEM["entity_from"] == country) & (DF_MFA_SYSTEM["sector_from"] == m_sector) & (DF_MFA_SYSTEM["sector_to"] == sector_to)
                ][
                    "mass"
                ].sum(),  # type: ignore
            )
            outflow_commodity = 0 if pd.isna(outflow_commodity) else outflow_commodity
            if outflow_commodity > SMALL_VALUE:
                # VP修正
                if (m_sector in M_SECTOR_VP_FIX) and (sector_to == M_SECTOR_VP_FIX[m_sector]["from"]):
                    sector_to: str = M_SECTOR_VP_FIX[m_sector]["to"]
                outflow_ratio[sector_to] = outflow_commodity / total_flow
                outflow_masses[sector_to] = outflow_commodity
        M_SECTOR_ALL[m_sector][country] = M(
            Commodity=m_sector,
            Country=country,
            InflowRatio=inflow_ratio,
            OutflowRatio=outflow_ratio,
            InflowMasses=inflow_masses,
            OutflowMasses=outflow_masses,
        )

# ======================================================================
# S部门参数构建
# ======================================================================


# 单个S对象
@dataclass
class S:
    Sector: str
    Country: str
    Demand: float  # 每轮需求量


S1_DICT: dict[str, S] = {}  # dict[国家代号,S]
for country in tqdm(COUNTRY_ALL):
    demand: float = DF_MFA_SYSTEM[(DF_MFA_SYSTEM["entity_to"] == country) & (DF_MFA_SYSTEM["sector_to"] == "S1")]["mass"].sum()
    demand = 0 if pd.isna(demand) else demand
    if demand < SMALL_VALUE:
        continue
    S1_DICT[country] = S(Sector="S1", Country=country, Demand=demand)


# %% [markdown]
# ## 遥测记录组件
#

# %%
@dataclass
class MfaTelemetryRecord:
    EntityFrom: str
    EntityTo: str
    SectorFrom: str
    SectorTo: str
    Mass: float
    Unit: str
    Extra: str | None


MFA_TELEMETRY_RECORDS: dict[int, list[MfaTelemetryRecord]] = {}  # {步数step: list[记录]}
MFA_TELEMETRY_SINGLE_STEP: list[MfaTelemetryRecord] = []  # 当前步数step的记录。每个step结束后，统一打包到MFA_TELEMETRY_RECORDS中，并清空


# ======================================================================
# Telemetry Extension
# ======================================================================


@dataclass
class AgentTelemetryRecord:
    Step: int
    Year: int
    Sector: str
    Country: str
    Capacity: float  # Capacity AFTER hazard modification
    CapacityLeft: float  # Remaining capacity
    Inventory: float  # Inventory level
    Stage: str  # "START" (begin of step) or "END" (end of step)


# Buffer containers
FLOW_RECORDS_BUFFER: list[MfaTelemetryRecord] = []  # Replaces MFA_TELEMETRY_SINGLE_STEP for clarity, or aliases it
AGENT_RECORDS_BUFFER: list[AgentTelemetryRecord] = []

# Alias for compatibility with your existing match function
MFA_TELEMETRY_SINGLE_STEP = FLOW_RECORDS_BUFFER


def capture_agent_states(
    step: int,
    year: int,
    stage: str,
    time_scale_ratio: float,
):
    """
    Snapshot the state (Capacity, Inventory) of all P-sectors.
    """
    for sector_name in sorted(P_SECTOR_ALL):
        country_dict = P_SECTOR_ALL[sector_name]
        for country_code in sorted(country_dict):
            p_obj = country_dict[country_code]
            AGENT_RECORDS_BUFFER.append(
                AgentTelemetryRecord(
                    Step=step,
                    Year=year,
                    Sector=sector_name,
                    Country=country_code,
                    Capacity=p_obj.Capacity * time_scale_ratio,
                    CapacityLeft=p_obj.CapacityLeft * time_scale_ratio,
                    Inventory=p_obj.Inventory * time_scale_ratio,
                    Stage=stage,
                )
            )


def add_step_context(df: pd.DataFrame, current_step: SimulationStep) -> None:
    df["step_start_date"] = current_step.start.isoformat()
    df["step_end_date_exclusive"] = current_step.end_exclusive.isoformat()
    df["step_days"] = current_step.days
    df["time_scale_ratio"] = current_step.time_scale_ratio
    df["random_seed"] = PARAMS["RANDOM_SEED"]
    df["terminal_step_policy"] = PARAMS["TERMINAL_STEP_POLICY"]


def flush_step_telemetry(current_step: SimulationStep, year: int, export_dir: str):
    """
    Save current buffers to parquet files and clear memory.
    """
    flow_metadata = None
    agent_metadata = None

    # 1. Save Trade Flows (MFA)
    if FLOW_RECORDS_BUFFER:
        df_flows = pd.DataFrame([r.__dict__ for r in FLOW_RECORDS_BUFFER])
        # Standardize columns
        df_flows.rename(
            columns={
                "EntityFrom": "entity_from",
                "EntityTo": "entity_to",
                "SectorFrom": "sector_from",
                "SectorTo": "sector_to",
                "Mass": "mass",
                "Unit": "unit",
                "Extra": "extra",
            },
            inplace=True,
        )

        # Add Time Context
        df_flows["step"] = current_step.index
        df_flows["year"] = year
        add_step_context(df_flows, current_step)

        # Save
        flow_dir = os.path.join(export_dir, "flows")
        os.makedirs(flow_dir, exist_ok=True)
        flow_metadata = atomic_write_parquet(
            df_flows,
            Path(flow_dir) / f"flows_step_{current_step.index}.parquet",
        )

        # Clear Memory
        FLOW_RECORDS_BUFFER.clear()

    # 2. Save Agent States (Inventory/Capacity)
    if AGENT_RECORDS_BUFFER:
        df_agents = pd.DataFrame([r.__dict__ for r in AGENT_RECORDS_BUFFER])
        df_agents.rename(
            columns={
                "Step": "step",
                "Year": "year",
                "Sector": "sector",
                "Country": "country",
                "Capacity": "capacity",
                "CapacityLeft": "capacity_left",
                "Inventory": "inventory",
                "Stage": "stage",
            },
            inplace=True,
        )
        add_step_context(df_agents, current_step)

        agent_dir = os.path.join(export_dir, "agents")
        os.makedirs(agent_dir, exist_ok=True)
        agent_metadata = atomic_write_parquet(
            df_agents,
            Path(agent_dir) / f"agents_step_{current_step.index}.parquet",
        )

        # Clear Memory
        AGENT_RECORDS_BUFFER.clear()

    if agent_metadata is not None:
        publish_completed_step(
            Path(export_dir),
            current_step,
            flow_metadata,
            agent_metadata,
        )

# %% [markdown]
# ## 模拟算法
#

# %% [markdown]
# ### 算法准备
#

# %%
# ======================================================================
# 需求&供给单对象 + 订单列表
# ======================================================================


# 需求单
@dataclass
class Demand:
    CountryInDemand: str  # 需求国家
    SectorInDemand: str  # 需求部门
    CommodityInDemand: list[str]  # 需求商品列表（完全替代关系）（例如['M7','M8']）
    MassInDemand: float  # 需求量
    MassFulfilled: float  # 已满足需求量
    IsPhantomFlowOrStockSector: bool  # 是否为Phantom Flow流或者S部门。Phantom Flow流没有对应部门对象，S部门的对象不是P对象，不需要进行库存修正与实际生产操作
    IsPhantom: bool  # 是否为幻影流
    BlacklistedCountries: list[str]  # 黑名单国家列表（不考虑这些国家的供给）
    UniqueID: str  # 唯一ID（UUID）
    IsPrioritized: bool = False  # 是否为优先需求
    # IsPrioritized 效果：在对单时，优先需求的订单不会被分配成本递增的阶梯序列，从而比其它订单更早被完全分配
    # IsPrioritized 使用建议：对于Phantom Flow等恒定大小的订单流，建议设置为True，以防止在capacity ratio参数调整时，这些订单因为无法递增而被过度挤占


# 供给单
@dataclass
class Supply:
    CountryInSupply: str  # 供给国家
    SectorInSupply: str  # 供给部门
    CommodityInSupply: str  # 供给商品（单一商品）
    IsSubsidiary: bool  # 是否为副产品供给
    MassInSupply: float  # 供给量
    MassSupplied: float  # 已供给量
    IsPhantomFlowOrStockSector: bool  # 是否为Phantom Flow流或者S部门。Phantom Flow流没有对应部门对象，S部门的对象不是P对象，不需要进行库存修正与实际生产操作
    IsPhantom: bool  # 是否为幻影流
    BlacklistedCountries: list[str]  # 黑名单国家列表（不考虑这些国家的需求）
    UniqueID: str  # 唯一ID（UUID）
    IsPrioritized: bool = False  # 是否为优先供给
    # IsPrioritized 效果：在对单时，优先供给的订单不会被分配成本递增的阶梯序列，从而比其它订单更早被完全分配
    # IsPrioritized 使用建议：对于Phantom Flow、P7等恒定大小的订单流，建议设置为True，以防止在capacity ratio参数调整时，这些订单因为无法递增而被过度挤占


DEMAND_TICKETS: list[Demand] = []  # 需求单列表
SUPPLY_TICKETS: list[Supply] = []  # 供给单列表
TICKET_ID_FACTORY = DeterministicIdFactory()


# =====================================================================
# 幻影流修正函数
# =====================================================================


def phant_supply_generate(commodity: str) -> list[Supply]:
    phant_supply_tickets: list[Supply] = []
    for country in sorted(M_SECTOR_ALL[commodity]):
        m_sector: M = M_SECTOR_ALL[commodity][country]
        if "Phantom Flow" in m_sector.InflowMasses:
            phantom_inflow_mass: float = m_sector.InflowMasses["Phantom Flow"]
            if phantom_inflow_mass > SMALL_VALUE:
                phant_supply_tickets.append(
                    Supply(
                        CountryInSupply=country,
                        SectorInSupply="Phantom Flow",
                        CommodityInSupply=commodity,
                        IsSubsidiary=False,
                        MassInSupply=phantom_inflow_mass,
                        MassSupplied=0.0,
                        IsPhantomFlowOrStockSector=True,
                        IsPhantom=True,
                        BlacklistedCountries=[],
                        UniqueID=TICKET_ID_FACTORY.next_id("supply"),
                        IsPrioritized="Phantom Flow" in PRIORITIZED_SUPPLY_SECTORS,
                    )
                )
    return phant_supply_tickets


def phant_demand_generate(commodity: str) -> list[Demand]:
    phant_demand_tickets: list[Demand] = []
    for country in sorted(M_SECTOR_ALL[commodity]):
        m_sector: M = M_SECTOR_ALL[commodity][country]
        if "Phantom Flow" in m_sector.OutflowMasses:
            phantom_outflow_mass: float = m_sector.OutflowMasses["Phantom Flow"]
            if phantom_outflow_mass > SMALL_VALUE:
                phant_demand_tickets.append(
                    Demand(
                        CountryInDemand=country,
                        SectorInDemand="Phantom Flow",
                        CommodityInDemand=[commodity],
                        MassInDemand=phantom_outflow_mass,
                        MassFulfilled=0.0,
                        IsPhantomFlowOrStockSector=True,
                        IsPhantom=True,
                        BlacklistedCountries=[],
                        UniqueID=TICKET_ID_FACTORY.next_id("demand"),
                        IsPrioritized="Phantom Flow" in PRIORITIZED_SUPPLY_SECTORS,
                    )
                )
    return phant_demand_tickets


# =====================================================================
# 市场操作函数
# =====================================================================


# 部门供给单生成
def supply_tickets_generate(sector: str) -> list[Supply]:
    supply_tickets: list[Supply] = []
    for country in sorted(P_SECTOR_ALL[sector]):
        p_sector: P = P_SECTOR_ALL[sector][country]
        # 计算最大产量
        ## 剩余产能限制
        max_suppliable_total_flow: float = p_sector.CapacityLeft
        ## 库存限制（只针对需要原材料输入的部门）
        if len(P_SECTOR_ALL[sector][country].RawMaterialInputRatio) > 0:
            max_suppliable_total_flow = min(max_suppliable_total_flow, p_sector.Inventory)
        if max_suppliable_total_flow < SMALL_VALUE:
            continue
        # 生成供给单
        max_suppliable_main_product: float = max_suppliable_total_flow * p_sector.ProductOutputRatio[p_sector.MainProduct]
        supply_tickets.append(
            Supply(
                CountryInSupply=country,
                SectorInSupply=sector,
                CommodityInSupply=p_sector.MainProduct,
                IsSubsidiary=False,
                MassInSupply=max_suppliable_main_product,
                MassSupplied=0.0,
                IsPhantomFlowOrStockSector=False,
                IsPhantom=False,
                BlacklistedCountries=[],
                UniqueID=TICKET_ID_FACTORY.next_id("supply"),
                IsPrioritized=(sector in PRIORITIZED_SUPPLY_SECTORS),
            )
        )
    return supply_tickets


# 部门需求单生成
def demand_tickets_generate(sector: str) -> list[Demand]:
    demand_tickets: list[Demand] = []
    if sector == "S1":
        for country in sorted(S1_DICT):
            s1_sector: S = S1_DICT[country]
            demand_tickets.append(
                Demand(
                    CountryInDemand=country,
                    SectorInDemand=sector,
                    CommodityInDemand=["M8"],
                    MassInDemand=s1_sector.Demand,
                    MassFulfilled=0.0,
                    IsPhantomFlowOrStockSector=True,
                    IsPhantom=False,
                    BlacklistedCountries=[],
                    UniqueID=TICKET_ID_FACTORY.next_id("demand"),
                    IsPrioritized=(sector in PRIORITIZED_DEMAND_SECTORS),
                )
            )
        return demand_tickets
    else:
        for country in sorted(P_SECTOR_ALL[sector]):
            p_sector: P = P_SECTOR_ALL[sector][country]
            # 纯输出部门跳过
            if len(p_sector.RawMaterialInputRatio) == 0:
                continue
            # 计算需求量
            demand_total_flow: float = max(
                0, p_sector.InheritCapacity * p_sector.InventoryTargetToCapacityRatio - p_sector.Inventory
            )  # 总需求量=目标库存-当前库存
            if demand_total_flow < SMALL_VALUE:
                continue
            ## 情况1：输入商品可完全替代，合并商品订单
            if p_sector.IsMaterialInputSubstitutable:
                demand_commodities: list[str] = sorted(p_sector.RawMaterialInputRatio)
                demand_tickets.append(
                    Demand(
                        CountryInDemand=country,
                        SectorInDemand=sector,
                        CommodityInDemand=demand_commodities,
                        MassInDemand=demand_total_flow,
                        MassFulfilled=0.0,
                        IsPhantomFlowOrStockSector=False,
                        IsPhantom=False,
                        BlacklistedCountries=[],
                        UniqueID=TICKET_ID_FACTORY.next_id("demand"),
                        IsPrioritized=(sector in PRIORITIZED_DEMAND_SECTORS),
                    )
                )
            ## 情况2：输入商品不可完全替代，为每个商品单独创建比例订单
            else:
                for commodity in sorted(p_sector.RawMaterialInputRatio):
                    demand_commodity_flow: float = demand_total_flow * p_sector.RawMaterialInputRatio[commodity]
                    demand_tickets.append(
                        Demand(
                            CountryInDemand=country,
                            SectorInDemand=sector,
                            CommodityInDemand=[commodity],
                            MassInDemand=demand_commodity_flow,
                            MassFulfilled=0.0,
                            IsPhantomFlowOrStockSector=False,
                            IsPhantom=False,
                            BlacklistedCountries=[],
                            UniqueID=TICKET_ID_FACTORY.next_id("demand"),
                            IsPrioritized=(sector in PRIORITIZED_DEMAND_SECTORS),
                        )
                    )
        return demand_tickets


# 单个订单对对单函数
def demand_supply_individual_match(
    demand_ticket: Demand,
    supply_ticket: Supply,
    mass: float,
    time_scale_ratio: float,
) -> None:
    """
    进行单个订单对的对单操作

    输入参数：
    - demand_ticket: 需求单对象
    - supply_ticket: 供给单对象
    - mass: 对单量
    """
    # -------------------------------------------------------
    # [FIX] Floating Point Drift Correction (Clamping)
    # -------------------------------------------------------
    # Calculate exact remaining amounts
    supply_remaining = supply_ticket.MassInSupply - supply_ticket.MassSupplied
    demand_remaining = demand_ticket.MassInDemand - demand_ticket.MassFulfilled

    # 1. Check Supply Limit
    if mass > supply_remaining:
        # If the overshoot is within tolerance (e.g., floating point noise), clamp it.
        # We use a slightly larger tolerance (10x SMALL_VALUE) for safety during the check,
        # assuming the solver intends to use up the remaining supply.
        if mass - supply_remaining < SMALL_VALUE * 10:
            mass = supply_remaining
        else:
            # If overshoot is significant, it's a real logic error.
            raise AssertionError(
                f"供给单剩余供给量不足 (Supply Insufficient)\n"
                f"ID: {supply_ticket.UniqueID}\n"
                f"Requested: {mass}\n"
                f"Remaining: {supply_remaining}\n"
                f"Diff: {mass - supply_remaining}"
            )

    # 2. Check Demand Limit
    if mass > demand_remaining:
        if mass - demand_remaining < SMALL_VALUE * 10:
            mass = demand_remaining
        else:
            raise AssertionError(
                f"需求单剩余需求量不足 (Demand Insufficient)\n"
                f"ID: {demand_ticket.UniqueID}\n"
                f"Requested: {mass}\n"
                f"Remaining: {demand_remaining}\n"
                f"Diff: {mass - demand_remaining}"
            )

    # 订单本体操作
    ## 前验证
    assert mass >= SMALL_VALUE, "对单量必须大于0"
    assert demand_ticket.MassInDemand - demand_ticket.MassFulfilled >= mass - SMALL_VALUE, "需求单剩余需求量不足"
    assert supply_ticket.MassInSupply - supply_ticket.MassSupplied >= mass - SMALL_VALUE, "供给单剩余供给量不足"
    assert supply_ticket.CommodityInSupply in demand_ticket.CommodityInDemand, "供给商品不在需求商品列表中"
    assert supply_ticket.CountryInSupply not in demand_ticket.BlacklistedCountries, "供给国家在需求单黑名单中"
    assert demand_ticket.CountryInDemand not in supply_ticket.BlacklistedCountries, "需求国家在供给单黑名单中"
    ## 订单更新
    demand_ticket.MassFulfilled += mass
    supply_ticket.MassSupplied += mass

    # =======
    # 遥测记录
    # =======
    # 我们模拟算法里都是P->P，但是MFA系统是P->M->P。因此进行一些中间转接
    # 具体方案如下：
    ## 原始过程：
    ### P -> P
    ###  \-> P
    ## 改正过程：
    # P -> M -> P
    #      ↓
    #      M -> P
    # P->M，进入本国市场
    MFA_TELEMETRY_SINGLE_STEP.append(
        MfaTelemetryRecord(
            EntityFrom=supply_ticket.CountryInSupply,
            EntityTo=supply_ticket.CountryInSupply,
            SectorFrom=supply_ticket.SectorInSupply,
            SectorTo=supply_ticket.CommodityInSupply,
            Mass=mass * time_scale_ratio,
            Unit="kg",
            Extra=None,
        )
    )
    # M->M，本国市场和外国市场的交互
    if supply_ticket.CountryInSupply != demand_ticket.CountryInDemand:
        MFA_TELEMETRY_SINGLE_STEP.append(
            MfaTelemetryRecord(
                EntityFrom=supply_ticket.CountryInSupply,
                EntityTo=demand_ticket.CountryInDemand,
                SectorFrom=supply_ticket.CommodityInSupply,
                SectorTo=supply_ticket.CommodityInSupply,
                Mass=mass * time_scale_ratio,
                Unit="kg",
                Extra=None,
            )
        )
    # M->P，最终的市场到后续P部门
    MFA_TELEMETRY_SINGLE_STEP.append(
        MfaTelemetryRecord(
            EntityFrom=demand_ticket.CountryInDemand,
            EntityTo=demand_ticket.CountryInDemand,
            SectorFrom=supply_ticket.CommodityInSupply,
            SectorTo=demand_ticket.SectorInDemand,
            Mass=mass * time_scale_ratio,
            Unit="kg",
            Extra=None,
        )
    )
    # =======
    # 遥测结束
    # =======

    # 买方国家操作
    if not demand_ticket.IsPhantomFlowOrStockSector:  # 非幻影流或S部门
        p_sector_demand: P = P_SECTOR_ALL[demand_ticket.SectorInDemand][demand_ticket.CountryInDemand]
        # 库存更新
        p_sector_demand.Inventory += mass

    # 卖方国家操作
    if (not supply_ticket.IsPhantomFlowOrStockSector) and (not supply_ticket.IsSubsidiary):  # 非幻影流或S部门，且非副产品
        # 主产品生产与库存、产能余量减少
        p_sector_supply: P = P_SECTOR_ALL[supply_ticket.SectorInSupply][supply_ticket.CountryInSupply]
        product_ratio: float = p_sector_supply.ProductOutputRatio[supply_ticket.CommodityInSupply]
        total_flow_consumed: float = mass / product_ratio
        ## 库存更新
        if len(p_sector_supply.RawMaterialInputRatio) > 0:  ## 有原材料输入的部门，才更新库存
            p_sector_supply.Inventory -= total_flow_consumed
            # --- 修正 1: 库存也加上归零逻辑 ---
            if p_sector_supply.Inventory < 0 and p_sector_supply.Inventory >= -SMALL_VALUE * 10:
                p_sector_supply.Inventory = 0.0
            assert p_sector_supply.Inventory >= -SMALL_VALUE, f"库存更新后为负值，部门{p_sector_supply.Sector}，国家{p_sector_supply.Country}"
        ## 剩余产能更新
        p_sector_supply.CapacityLeft -= total_flow_consumed
        # --- 修正 2: 增加归零逻辑 (关键修改) ---
        # 如果剩余产能略微小于0（在10倍SMALL_VALUE容差范围内），直接强制归零
        # 这里的 * 10 是为了容忍 float 运算在多次累加中产生的漂移
        if p_sector_supply.CapacityLeft < 0 and p_sector_supply.CapacityLeft >= -SMALL_VALUE * 10:
            p_sector_supply.CapacityLeft = 0.0
        assert p_sector_supply.CapacityLeft >= -SMALL_VALUE, f"剩余产能更新后为负值，部门{p_sector_supply.Sector}，国家{p_sector_supply.Country}"
        # 副产品订单创建
        for commodity in sorted(p_sector_supply.ProductOutputRatio):
            ratio = p_sector_supply.ProductOutputRatio[commodity]
            if (commodity != p_sector_supply.MainProduct) and (ratio > SMALL_VALUE):
                subsidiary_mass: float = total_flow_consumed * ratio
                if subsidiary_mass > SMALL_VALUE:
                    SUPPLY_TICKETS.append(
                        Supply(
                            CountryInSupply=supply_ticket.CountryInSupply,
                            SectorInSupply=supply_ticket.SectorInSupply,
                            CommodityInSupply=commodity,
                            IsSubsidiary=True,
                            MassInSupply=subsidiary_mass,
                            MassSupplied=0.0,
                            IsPhantomFlowOrStockSector=False,
                            IsPhantom=False,
                            BlacklistedCountries=deepcopy(supply_ticket.BlacklistedCountries),
                            UniqueID=TICKET_ID_FACTORY.next_id("supply"),
                        )
                    )
    else:
        pass


## （系列对单算法使用的）供给单数据结构（与上面的Supply对象不同）
@dataclass
class SupplyForAlgo:
    ID: str  # 供给单ID
    Commodity: str  # 供给商品
    Quantity: float  # 供给量
    Country: str  # [NEW] Add Country
    BlacklistedCountries: set[str]  # [NEW] Use Set for O(1) lookup
    IsPrioritized: bool = False  # 是否为优先供给（详见Supply类）


## （系列对单算法使用的）需求单数据结构（与上面的Demand对象不同）
@dataclass
class DemandForAlgo:
    ID: str  # 需求单ID
    Commodities: list[str]  # 需求商品列表（完全替代关系）（例如['M7','M8']）
    Quantity: float  # 需求量
    Country: str  # [NEW] Add Country
    BlacklistedCountries: set[str]  # [NEW] Use Set for O(1) lookup
    IsPrioritized: bool = False  # 是否为优先需求（详见Demand类）


def supply_algo_sort_key(ticket: SupplyForAlgo) -> tuple:
    return (
        ticket.Country,
        ticket.Commodity,
        ticket.IsPrioritized,
        ticket.Quantity,
        tuple(sorted(ticket.BlacklistedCountries)),
        ticket.ID,
    )


def demand_algo_sort_key(ticket: DemandForAlgo) -> tuple:
    return (
        ticket.Country,
        tuple(ticket.Commodities),
        ticket.IsPrioritized,
        ticket.Quantity,
        tuple(sorted(ticket.BlacklistedCountries)),
        ticket.ID,
    )


## 算法匹配结果
@dataclass
class MatchResult:
    SupplierID: str
    DemanderID: str
    Volume: float
    Commodity: str


# 基于google or tools包，使用
# 最小成本网络算法的
# 近似公平订单匹配函数
def solve_match_or_tools_fair_optimized(
    supply_tickets: list[SupplyForAlgo],
    demand_tickets: list[DemandForAlgo],
    chunk_size: int = 100,
    precision_factor: int = max(int(0.5 / SMALL_VALUE), 1),
    is_domestic_mode: bool = False,  # [NEW] Add mode flag
) -> tuple[float, dict[tuple[str, str], MatchResult]]:
    """
    Optimized version of the fair matching solver.
    Replaces NetworkX with OR-Tools MaxFlow and optimizes graph construction loops.
    """
    supply_tickets = sorted(supply_tickets, key=supply_algo_sort_key)
    demand_tickets = sorted(demand_tickets, key=demand_algo_sort_key)
    PRECISION_FACTOR = precision_factor
    assert PRECISION_FACTOR <= int(1 / SMALL_VALUE), "precision_factor不能大于1/SMALL_VALUE，否则会出现各种奇怪报错"

    # ==========================================
    # 0. PRE-PROCESSING & SAFETY CHECKS
    # ==========================================

    # 1. Calculate totals first to define capacity and check safety
    total_supply_qty = sum(s.Quantity for s in supply_tickets)
    total_demand_qty = sum(d.Quantity for d in demand_tickets)

    # 2. Scale and check for INT64 overflow
    # OR-Tools uses int64, max value is (2^63 - 1)
    INT64_MAX = 9_223_372_036_854_775_807

    # Calculate the max possible flow in the system (total supply is the upper bound)
    # We add a small buffer (+1) just to be strictly greater
    max_potential_flow_scaled = int(total_supply_qty * PRECISION_FACTOR) + 1

    # Check if we are safe
    if max_potential_flow_scaled >= INT64_MAX:
        raise OverflowError(
            f"Precision factor {PRECISION_FACTOR} is too high for the dataset volume. " f"Scaled flow {max_potential_flow_scaled} exceeds int64 limit."
        )

    # 3. Define Dynamic HUGE_CAPACITY
    # This guarantees the edge never bottlenecks the flow, as it's > total possible supply
    HUGE_CAPACITY = max_potential_flow_scaled

    # Map IDs to integers once.
    # Structure: [Source, Sink, Supply_Nodes..., Demand_Nodes...]

    # Static Node Indices
    SOURCE_IDX = 0
    SINK_IDX = 1

    # Counters
    current_idx = 2

    # Mappings
    s_id_to_idx = {}
    d_id_to_idx = {}

    # To reconstruct results later
    idx_to_s_id = {}
    idx_to_d_id = {}

    # Supply Nodes
    for s in supply_tickets:
        s_id_to_idx[s.ID] = current_idx
        idx_to_s_id[current_idx] = s.ID
        current_idx += 1

    # Demand Nodes
    for d in demand_tickets:
        d_id_to_idx[d.ID] = current_idx
        idx_to_d_id[current_idx] = d.ID
        current_idx += 1

    # Optimization: Group demands by commodity to avoid O(N*M) loop
    demands_by_comm = {}
    for d in demand_tickets:
        for c in d.Commodities:
            if c not in demands_by_comm:
                demands_by_comm[c] = []
            demands_by_comm[c].append(d)

    # ==========================================
    # 1. CALCULATE MAX FLOW (OR-Tools)
    # ==========================================
    # Use SimpleMaxFlow instead of NetworkX for speed

    mf = max_flow.SimpleMaxFlow()

    # A. Source -> Supply
    for s in supply_tickets:
        cap = int(s.Quantity * PRECISION_FACTOR)
        mf.add_arc_with_capacity(SOURCE_IDX, s_id_to_idx[s.ID], cap)

    # B. Demand -> Sink
    for d in demand_tickets:
        cap = int(d.Quantity * PRECISION_FACTOR)
        mf.add_arc_with_capacity(d_id_to_idx[d.ID], SINK_IDX, cap)

    # C. Supply -> Demand (The Matching Layer)
    # We use a huge capacity here since limits are on nodes
    for s in supply_tickets:
        candidate_demands = demands_by_comm.get(s.Commodity, [])
        for d in candidate_demands:
            if is_domestic_mode:
                if s.Country != d.Country:
                    continue
            
            # 2. Blacklist Check (Set Lookup O(1))
            # "If Supplier's Country is in Demander's Blacklist" -> BLOCK
            if s.Country in d.BlacklistedCountries:
                continue
            # "If Demander's Country is in Supplier's Blacklist" -> BLOCK
            if d.Country in s.BlacklistedCountries:
                continue

            # [OPTIMIZED CHECK END]

            # If passed, add arc
            mf.add_arc_with_capacity(s_id_to_idx[s.ID], d_id_to_idx[d.ID], HUGE_CAPACITY)

    # Solve Max Flow
    status = mf.solve(SOURCE_IDX, SINK_IDX)

    if status != mf.OPTIMAL:
        logger.warning("Max flow solver could not find an optimal solution.")
        return 0, {}

    max_flow_value = mf.optimal_flow()

    if max_flow_value < SMALL_VALUE * PRECISION_FACTOR:
        logger.info("订单匹配算法未检查到可用订单 (Max Flow = 0)")
        return 0, {}

    # ==========================================
    # 2. MIN COST MAX FLOW (OR-Tools)
    # ==========================================

    mcf = min_cost_flow.SimpleMinCostFlow()

    # We continue using `current_idx` for the new "Tier" nodes
    # No need to store string names for tier nodes, just use the integer index

    # A. Source -> Tier Nodes -> Supply Nodes
    total_supply_scaled = 0

    for s in supply_tickets:
        all_quantity = int(s.Quantity * PRECISION_FACTOR)
        if all_quantity == 0:
            continue

        total_supply_scaled += all_quantity
        s_node_idx = s_id_to_idx[s.ID]

        # Calculate pieces
        base_piece = all_quantity // chunk_size
        remainder = all_quantity % chunk_size
        cost = 1

        for tier in range(1, chunk_size + 1):
            single_piece = base_piece + (1 if tier <= remainder else 0)
            if single_piece == 0:
                continue

            # Tier Logic: Prioritized gets lower cost (flat cost), others scale
            if not s.IsPrioritized:
                cost += 1
            current_cost = cost

            # Allocate a new transient node index for this tier
            tier_node_idx = current_idx
            current_idx += 1

            # Arc: Source -> Tier
            mcf.add_arc_with_capacity_and_unit_cost(SOURCE_IDX, tier_node_idx, single_piece, current_cost)
            # Arc: Tier -> Supply
            mcf.add_arc_with_capacity_and_unit_cost(tier_node_idx, s_node_idx, single_piece, 0)

    # B. Demand Nodes -> Tier Nodes -> Sink
    for d in demand_tickets:
        all_quantity = int(d.Quantity * PRECISION_FACTOR)
        if all_quantity == 0:
            continue

        d_node_idx = d_id_to_idx[d.ID]

        base_piece = all_quantity // chunk_size
        remainder = all_quantity % chunk_size
        cost = 1

        for tier in range(1, chunk_size + 1):
            single_piece = base_piece + (1 if tier <= remainder else 0)
            if single_piece == 0:
                continue

            if not d.IsPrioritized:
                cost += 1
            current_cost = cost

            tier_node_idx = current_idx
            current_idx += 1

            # Arc: Demand -> Tier
            mcf.add_arc_with_capacity_and_unit_cost(d_node_idx, tier_node_idx, single_piece, 0)
            # Arc: Tier -> Sink
            mcf.add_arc_with_capacity_and_unit_cost(tier_node_idx, SINK_IDX, single_piece, current_cost)

    # C. Supply -> Demand (Market Matching)
    # Re-use the optimized loop from Max Flow step
    # Note: We use total_supply_scaled as capacity to be safe

    for s in supply_tickets:
        candidate_demands = demands_by_comm.get(s.Commodity, [])
        for d in candidate_demands:
            if is_domestic_mode:
                if s.Country != d.Country:
                    continue
            
            # 2. Blacklist Check (Set Lookup O(1))
            # "If Supplier's Country is in Demander's Blacklist" -> BLOCK
            if s.Country in d.BlacklistedCountries:
                continue
            # "If Demander's Country is in Supplier's Blacklist" -> BLOCK
            if d.Country in s.BlacklistedCountries:
                continue

            # ---------------------------------------------------
            # [OPTIMIZED CHECK END]
            # ---------------------------------------------------

            mcf.add_arc_with_capacity_and_unit_cost(s_id_to_idx[s.ID], d_id_to_idx[d.ID], total_supply_scaled, 0)

    # D. Set Supplies
    mcf.set_node_supply(SOURCE_IDX, max_flow_value)
    mcf.set_node_supply(SINK_IDX, -max_flow_value)

    # E. Solve
    status = mcf.solve()

    # ==========================================
    # 3. EXTRACT RESULTS
    # ==========================================
    matches: dict[tuple[str, str], MatchResult] = {}
    total_flow = 0.0

    if status == mcf.OPTIMAL:
        for i in range(mcf.num_arcs()):
            flow_vol = mcf.flow(i)
            if flow_vol > 0:
                tail = mcf.tail(i)
                head = mcf.head(i)

                # Check if this arc is strictly between a Supply Node and a Demand Node
                # We use our lookup dicts to verify. Tier nodes are not in these dicts.
                if tail in idx_to_s_id and head in idx_to_d_id:
                    s_id = idx_to_s_id[tail]
                    d_id = idx_to_d_id[head]

                    unscaled_vol = flow_vol / PRECISION_FACTOR
                    found_comm = next(t.Commodity for t in supply_tickets if t.ID == s_id)

                    matches[(s_id, d_id)] = MatchResult(SupplierID=s_id, DemanderID=d_id, Volume=unscaled_vol, Commodity=found_comm)
                    total_flow += unscaled_vol
    else:
        logger.error(f"OR-Tools MinCostFlow solver failed. Status: {status}")
        raise RuntimeError(f"Solver failed with status {status}")

    return total_flow, matches


# 订单匹配函数
def final_match(
    is_domestic_match: bool,
    is_global_match: bool,
    time_scale_ratio: float,
) -> None:
    """
    进行订单匹配的外层函数

    目前使用google or tools的最小成本最大流算法

    输入参数：
    - is_domestic_match: bool, 是否是国内匹配环节
    - is_global_match: bool, 是否是全球匹配环节

    附注：
    - 订单匹配函数不限定部门，部门顺序需要通过生产、需求单生成函数来控制
    - domestic_match -> global_match，每个阶段优先进行副产品对单
    """
    assert (is_domestic_match + is_global_match) == 1, "只能选择一种匹配模式"

    def main_logic(is_subsidiary_match: bool) -> None:
        demand_tickets_to_process: list[DemandForAlgo] = []
        supply_tickets_to_process: list[SupplyForAlgo] = []

        # 需求单生成
        for demand in sorted(
            DEMAND_TICKETS,
            key=lambda ticket: (
                ticket.CountryInDemand,
                tuple(ticket.CommodityInDemand),
                ticket.SectorInDemand,
                ticket.IsPrioritized,
                ticket.MassInDemand,
                ticket.UniqueID,
            ),
        ):
            if demand.MassInDemand - demand.MassFulfilled <= SMALL_VALUE:
                continue
            ## 加入订单自身的对单列表限制
            blacklisted_set = set(demand.BlacklistedCountries)
            ## 生成订单
            demand_tickets_to_process.append(
                DemandForAlgo(
                    ID=demand.UniqueID,
                    Commodities=demand.CommodityInDemand,
                    Quantity=demand.MassInDemand - demand.MassFulfilled,
                    Country=demand.CountryInDemand,  # Pass Country
                    BlacklistedCountries=blacklisted_set,  # Pass Set
                    IsPrioritized=demand.IsPrioritized,
                )
            )

        # 供给单生成
        for supply in sorted(
            SUPPLY_TICKETS,
            key=lambda ticket: (
                ticket.CountryInSupply,
                ticket.CommodityInSupply,
                ticket.SectorInSupply,
                ticket.IsSubsidiary,
                ticket.IsPrioritized,
                ticket.MassInSupply,
                ticket.UniqueID,
            ),
        ):
            if supply.MassInSupply - supply.MassSupplied <= SMALL_VALUE:
                continue
            if is_subsidiary_match != supply.IsSubsidiary:
                continue
            blacklisted_set = set(supply.BlacklistedCountries)
            ## 生成订单
            supply_tickets_to_process.append(
                SupplyForAlgo(
                    ID=supply.UniqueID,
                    Commodity=supply.CommodityInSupply,
                    Quantity=supply.MassInSupply - supply.MassSupplied,
                    Country=supply.CountryInSupply,  # Pass Country
                    BlacklistedCountries=blacklisted_set,  # Pass Set
                    IsPrioritized=supply.IsPrioritized,
                )
            )

        # 对单+修正
        _, match_results = solve_match_or_tools_fair_optimized(
            supply_tickets_to_process, demand_tickets_to_process, chunk_size=FAIR_MATCH_CHUNKS, is_domestic_mode=is_domestic_match  # Pass this flag to solver
        )
        for sup_id, dem_id in sorted(match_results):
            match = match_results[(sup_id, dem_id)]
            demand_ticket: Demand = next(d for d in DEMAND_TICKETS if d.UniqueID == dem_id)
            supply_ticket: Supply = next(s for s in SUPPLY_TICKETS if s.UniqueID == sup_id)
            demand_supply_individual_match(
                demand_ticket,
                supply_ticket,
                match.Volume,
                time_scale_ratio,
            )

    # 副产品对单
    main_logic(is_subsidiary_match=True)
    # 主产品对单
    main_logic(is_subsidiary_match=False)


# %% [markdown]
# ### 模拟
#

# %% [markdown]
# ##### [在用] ISIMIP3b 情境编排
#

# %%
SECTOR_FILENAME_MAP = {"iron-mine": "P1", "iron-factory": "P2", "steel-factory": "P3"}

# Hazard & Scenario Configuration
HAZARD_TYPE: str = PARAMS["HAZARD_TYPE"]
CLIMATE_SCENARIO: str = PARAMS["CLIMATE_SCENARIO"]
HAZARD_DATA_DIR: str = PARAMS["HAZARD_DATA_DIR"]


# 风险数据加载模块
class HazardDataLoader:
    def __init__(self, data_dir: str, hazard_type: str, scenario: str):
        self.data_cache = {}  # {year: dataframe}
        self.baseline_2023_cache = None  # heatwave only
        self.data_dir = data_dir
        self.hazard_type = hazard_type
        self.scenario = scenario
        self.is_blank_reference = self.hazard_type.lower() == "none"

    def _load_raw_data(self, year: int) -> pd.DataFrame:
        """
        Internal helper: Loads raw parquet files for a specific year without applying any logic.
        """
        if self.is_blank_reference:
            return pd.DataFrame()

        # Construct search pattern
        file_pattern = os.path.join(self.data_dir, f"*_{self.hazard_type}_{year}_{self.scenario}.parquet")
        files = glob.glob(file_pattern)

        if not files:
            raise FileNotFoundError(
                f"Missing required hazard data for year {year}, "
                f"hazard {self.hazard_type}, scenario {self.scenario}"
            )

        dfs = []
        for f in files:
            filename = os.path.basename(f)
            try:
                raw_sector_name = filename.split(f"_{self.hazard_type}")[0]
            except IndexError:
                logger.warning(f"Skipping file with unexpected name format: {filename}")
                continue

            if raw_sector_name in SECTOR_FILENAME_MAP:
                sector_code = SECTOR_FILENAME_MAP[raw_sector_name]
                temp_df = pd.read_parquet(f)
                temp_df["sector"] = sector_code
                dfs.append(temp_df)
            else:
                logger.warning(f"Ignored file with unmapped sector: {raw_sector_name}")

        if not dfs:
            raise ValueError(
                f"Required hazard files for year {year} contained no mapped sectors"
            )

        full_df = pd.concat(dfs, ignore_index=True)
        if not full_df.empty:
            full_df["day_of_year"] = full_df["day_of_year"].astype(int)
            full_df = full_df.set_index(["sector", "country_a3", "day_of_year"]).sort_index()
            for day_of_year in sorted(
                full_df.index.get_level_values("day_of_year").unique()
            ):
                date_from_day_of_year(year, day_of_year)

        return full_df

    def load_year_data(self, year: int):
        """
        Lazy load and merge all sector data for a specific year.
        If heatwave, applies relative loss logic (Year X - Year 2023).
        """
        if self.is_blank_reference:
            return pd.DataFrame()

        if year in self.data_cache:
            return self.data_cache[year]

        # 1. Load Raw Data for the requested year
        df_current = self._load_raw_data(year)

        # 2. Apply Heatwave Logic: Subtract 2023 Baseline
        if self.hazard_type == "heatwave" and not df_current.empty:
            # Ensure 2023 baseline is loaded (raw)
            if self.baseline_2023_cache is None:
                logger.info("Loading 2023 baseline for heatwave relative calculation...")
                self.baseline_2023_cache = self._load_raw_data(2023)

            if self.baseline_2023_cache.empty:
                raise FileNotFoundError(
                    "Missing required heatwave baseline data for 2023 "
                    f"and scenario {self.scenario}"
                )

            baseline_loss = align_2023_baseline(
                self.baseline_2023_cache["production_loss_fraction"],
                df_current.index,
                target_year=year,
            )

            # Perform subtraction
            df_current["production_loss_fraction"] = df_current["production_loss_fraction"] - baseline_loss

            # Low cap at 0 (No negative loss allowed)
            df_current["production_loss_fraction"] = df_current["production_loss_fraction"].clip(lower=0.0)

        # 3. Cache and Return
        self.data_cache[year] = df_current
        return self.data_cache[year]

    def get_loss_fraction(self, start_date: date, end_date: date, sector: str, country: str) -> float:
        """
        sector: sector code, e.g., "P1"
        country: country code, iso 3166 a3, e.g., "USA"
        start_date, end_date: date objects defining the range (end_date exclusive)

        [CHANGE] Calculate weighted average loss for a date range that might span two years.
        """
        # [CHANGE] Immediate return for blank reference
        if self.is_blank_reference:
            return 0.0

        total_days = (end_date - start_date).days
        if total_days <= 0:
            return 0.0

        total_loss_sum = 0.0

        # Iterate day by day to handle year boundaries safely
        for current_chk_date in iter_dates(start_date, end_date):
            year = current_chk_date.year
            day_of_year = current_chk_date.timetuple().tm_yday
            date_from_day_of_year(year, day_of_year)

            # Lazy load happens here automatically if year changes
            df = self.load_year_data(year)

            daily_loss = 0.0
            if not df.empty:
                try:
                    # Fast lookup
                    daily_loss = df.loc[(sector, country, day_of_year)]["production_loss_fraction"]
                except KeyError:
                    daily_loss = 0.0
            else:
                logger.warning(f"No data loaded for year {year}")

            total_loss_sum += daily_loss

        return total_loss_sum / total_days

    def clear_old_cache(self, active_year: int):
        if self.is_blank_reference:
            return

        keys_to_delete = [y for y in self.data_cache if y < active_year]
        for k in keys_to_delete:
            del self.data_cache[k]


# Initialize Loader (Update path as needed)
hazard_loader = HazardDataLoader(data_dir=HAZARD_DATA_DIR, hazard_type=HAZARD_TYPE, scenario=CLIMATE_SCENARIO)

# %% [markdown]
# ##### [在用] 贸易情境编排
#

# %%
# ======================================================================
# Geopolitical & Trade Conflict Configuration
# ======================================================================

# Scenario Options:
# 1. "free-trade"           - No restrictions
# 2. "us-cn-decoupling"     - Direct decoupling between US and CN only
# 3. "bloc-confrontation"   - US Allies vs CN Partners (Iron Curtain style)
# 4. "global-decoupling"    - No international trade (Domestic only)
# 5. "random-disruption"    - [NEW] Random % of countries blacklisted per step
TRADE_SCENARIO: str = PARAMS["TRADE_SCENARIO"]

# [NEW] Configuration for Random Disruption
# Percentage (0.0 - 1.0) of countries that will be blacklisted by every country
# e.g., 0.1 means Country A will randomly block 10% of all other countries in the world.
RANDOM_DISRUPTION_RATIO: float = PARAMS["RANDOM_DISRUPTION_RATIO"]
RANDOM_SEED: int | None = PARAMS["RANDOM_SEED"]
RANDOM_GENERATOR: random.Random | None = (
    random.Random(RANDOM_SEED) if TRADE_SCENARIO == "random-disruption" else None
)

# [NEW] Cache to store the random blacklist map for the current simulation step
# Structure: { "USA": ["CHN", "RUS"...], "CHN": ["USA", "JPN"...] }
STEP_RANDOM_BLACKLIST_CACHE: dict[str, list[str]] = {}

# Define Country Groups (You should expand these lists based on your specific MFA country codes)
GROUP_USA = {"USA"}
GROUP_CHN = {"CHN", "HKG", "MAC", "TWN"}  # Greater China region
GROUP_US_ALLIES = {
    "USA",
    "GBR",
    "CAN",
    "AUS",
    "NZL",  # Five Eyes
    "JPN",
    "KOR",
    "PHL",
    "DEU",
    "FRA",
    "ITA",
    "ESP",
    "NLD",
    "BEL",
    "SWE",
    "POL",  # NATO/EU
}
GROUP_CN_PARTNERS = {"CHN", "RUS", "PRK", "IRN", "BLR", "VEN", "SYR"}  # Paper-defined bloc

def update_step_random_blacklist():
    """
    [NEW HELPER] Regenerates the random blacklist for the current step.
    Must be called at the beginning of the simulation loop.
    """
    STEP_RANDOM_BLACKLIST_CACHE.clear()
    
    if TRADE_SCENARIO == "random-disruption":
        assert RANDOM_GENERATOR is not None
        STEP_RANDOM_BLACKLIST_CACHE.update(
            build_random_blacklists(
                COUNTRY_ALL,
                RANDOM_DISRUPTION_RATIO,
                RANDOM_GENERATOR,
            )
        )

def get_trade_blacklist(country: str) -> list[str]:
    """
    Returns a list of country codes that the given 'country' CANNOT trade with.
    """
    if TRADE_SCENARIO == "free-trade":
        return []

    elif TRADE_SCENARIO == "us-cn-decoupling":
        # USA blocks China Group
        if country == "USA":
            return sorted(GROUP_CHN)
        # China Group blocks USA
        if country in GROUP_CHN:
            return ["USA"]

    elif TRADE_SCENARIO == "bloc-confrontation":
        # US Allies block CN Partners
        if country in GROUP_US_ALLIES:
            return sorted(GROUP_CN_PARTNERS)
        # CN Partners block US Allies
        if country in GROUP_CN_PARTNERS:
            return sorted(GROUP_US_ALLIES)

    elif TRADE_SCENARIO == "global-decoupling":
        # Block everyone except itself
        # Note: COUNTRY_ALL must be available in global scope
        return [c for c in COUNTRY_ALL if c != country]
    
    # [NEW] Random Disruption Logic
    elif TRADE_SCENARIO == "random-disruption":
        # Return the pre-calculated list for this step from cache
        return STEP_RANDOM_BLACKLIST_CACHE[country]
    
    else:
        logger.warning(f"Unknown trade scenario: {TRADE_SCENARIO}. No trade restrictions applied.")
    return []


# %% [markdown]
# #### 主循环
#

# %%
# ======================================================================
# 3. Main Simulation Loop [CHANGE: Complete Logic Replacement]
# ======================================================================

# State Tracking
current_year_loaded = -1

# Containers
DEMAND_TICKETS = []
SUPPLY_TICKETS = []

# Prepare Output Directory
random_run_identifier = (
    f"{RANDOM_DISRUPTION_RATIO}_seed{RANDOM_SEED}"
    if TRADE_SCENARIO == "random-disruption"
    else ""
)
uid = f"{HAZARD_TYPE}_{CLIMATE_SCENARIO}_abun-cap{CAPACITY_TO_INITIAL_FLOW}_{TRADE_SCENARIO}{random_run_identifier}_inv{INVENTORY_TO_CAPACITY_RATIO}_{START_YEAR}-{END_YEAR}_sd{STEP_DAYS}_sensitivity-id-{str(uuid4())[:8]}"
SIMULATION_EXPORT_DIR = str(Path(PARAMS["OUTPUT_DIR"]) / uid)

# A new run must never delete an existing trajectory, even on an ID collision.
os.makedirs(SIMULATION_EXPORT_DIR, exist_ok=False)

run_metadata = {
    "random_seed": RANDOM_SEED,
    "terminal_step_policy": TERMINAL_STEP_POLICY,
    "steps": [
        {
            "index": current_step.index,
            "start_date": current_step.start.isoformat(),
            "end_date_exclusive": current_step.end_exclusive.isoformat(),
            "step_days": current_step.days,
            "time_scale_ratio": current_step.time_scale_ratio,
        }
        for current_step in SIMULATION_SCHEDULE
    ],
}
with open(
    os.path.join(SIMULATION_EXPORT_DIR, "run_metadata.json"),
    "w",
    encoding="utf-8",
) as metadata_file:
    json.dump(run_metadata, metadata_file, ensure_ascii=False, indent=2)


for current_step in tqdm(SIMULATION_SCHEDULE, desc="Simulation Steps"):
    step = current_step.index
    gc.collect()
    # -------------------------------------------------------
    # [NEW] 0. Update Dynamic Scenarios
    # -------------------------------------------------------
    # This ensures the random blacklist changes every step
    update_step_random_blacklist()

    # -------------------------------------------------------
    # 1. Time Calculation [CHANGE]
    # -------------------------------------------------------
    step_start_date = current_step.start
    step_end_date = current_step.end_exclusive
    current_year = step_start_date.year

    # -------------------------------------------------------
    # 2. Annual Maintenance [CHANGE: Cache & Export]
    # -------------------------------------------------------
    if current_year != current_year_loaded:
        logger.info(f"=== Entering Year {current_year} ===")

        # Clear old hazard data to save RAM
        hazard_loader.clear_old_cache(current_year)

        current_year_loaded = current_year

    # -------------------------------------------------------
    # 3. Initialization & Modifiers
    # -------------------------------------------------------
    DEMAND_TICKETS.clear()
    SUPPLY_TICKETS.clear()
    TICKET_ID_FACTORY.start_step(step)
    FLOW_RECORDS_BUFFER.clear()  # Safety clear
    AGENT_RECORDS_BUFFER.clear()  # Safety clear

    # [A] Inherit Capacity and Capacity Left Restoration
    # Reset Capacity to full InheritCapacity before applying new modifiers
    for p_sector in sorted(P_SECTOR_ALL):
        for country in sorted(P_SECTOR_ALL[p_sector]):
            p_instance = P_SECTOR_ALL[p_sector][country]
            p_instance.Capacity = p_instance.InheritCapacity
            p_instance.CapacityLeft = p_instance.Capacity

    # [B] Dynamic Hazard Modifier [CHANGE: On-the-fly Query]
    # We check P1, P2, P3 (iron-mine, iron-factory, steel-factory)
    affected_sectors = ["P1", "P2", "P3"]

    for sector_code in affected_sectors:
        if sector_code not in P_SECTOR_ALL:
            logger.warning(f"Sector {sector_code} not found in P_SECTOR_ALL, skipping hazard application.")
            continue

        for country in sorted(P_SECTOR_ALL[sector_code]):
            # Query the loader with date range (handles year split automatically)
            loss_fraction = hazard_loader.get_loss_fraction(start_date=step_start_date, end_date=step_end_date, sector=sector_code, country=country)

            if loss_fraction > SMALL_VALUE:
                p_instance = P_SECTOR_ALL[sector_code][country]

                # Apply loss: Capacity = Inherit * (1 - Loss)
                p_instance.Capacity = p_instance.InheritCapacity * (1.0 - loss_fraction)
                p_instance.Capacity = max(0.0, p_instance.Capacity)

                # Sync CapacityLeft
                p_instance.CapacityLeft = min(p_instance.CapacityLeft, p_instance.Capacity)

    # =======================================================
    # [TELEMETRY] CAPTURE START STATE
    # Records: Inventory (Beginning), Capacity (Modified), CapacityLeft (Full)
    # =======================================================
    capture_agent_states(
        step,
        current_year,
        stage="START",
        time_scale_ratio=current_step.time_scale_ratio,
    )

    # -------------------------------------------------------
    # 4. Supply Chain Simulation (Unchanged Logic)
    # -------------------------------------------------------
    for cur_sector_list in INDUSTRIAL_CHAIN_ORDER:
        # Generate Phantom Demand
        for cur_sector in cur_sector_list:
            if cur_sector in PHANTOM_PRODUCE_RELATION:
                for commodity in PHANTOM_PRODUCE_RELATION[cur_sector]:
                    DEMAND_TICKETS.extend(phant_demand_generate(commodity))

        # Generate Supply
        for cur_sector in cur_sector_list:
            if cur_sector != "S1":
                SUPPLY_TICKETS.extend(supply_tickets_generate(cur_sector))

        # Generate Phantom Supply
        for cur_sector in cur_sector_list:
            if cur_sector in PHANTOM_PRODUCE_RELATION:
                for commodity in PHANTOM_PRODUCE_RELATION[cur_sector]:
                    SUPPLY_TICKETS.extend(phant_supply_generate(commodity))

        # =======================================================
        # [C] Geopolitical Modifier [CHANGE: Trade Conflict]
        # Apply blacklist to all active tickets before matching
        # =======================================================
        if TRADE_SCENARIO != "free-trade":
            # Update Supply Tickets
            for s_ticket in SUPPLY_TICKETS:
                # Only process if not fully supplied (optimization)
                if s_ticket.MassInSupply - s_ticket.MassSupplied > SMALL_VALUE:
                    s_ticket.BlacklistedCountries = get_trade_blacklist(s_ticket.CountryInSupply)

            # Update Demand Tickets
            for d_ticket in DEMAND_TICKETS:
                # Only process if not fully satisfied (optimization)
                if d_ticket.MassInDemand - d_ticket.MassFulfilled > SMALL_VALUE:
                    d_ticket.BlacklistedCountries = get_trade_blacklist(d_ticket.CountryInDemand)
        # =======================================================

        # Match Orders
        if "S1" in cur_sector_list:
            assert len(cur_sector_list) == 1, "S1部门必须单独成组"
            pass
        else:
            final_match(
                is_domestic_match=True,
                is_global_match=False,
                time_scale_ratio=current_step.time_scale_ratio,
            )
            final_match(
                is_domestic_match=False,
                is_global_match=True,
                time_scale_ratio=current_step.time_scale_ratio,
            )

        # Generate Demand
        for cur_sector in cur_sector_list:
            DEMAND_TICKETS.extend(demand_tickets_generate(cur_sector))

    # =======================================================
    # [TELEMETRY] CAPTURE END STATE
    # Records: Inventory (End), Capacity (Unchanged), CapacityLeft (Used)
    # =======================================================
    capture_agent_states(
        step,
        current_year,
        stage="END",
        time_scale_ratio=current_step.time_scale_ratio,
    )

    # =======================================================
    # [TELEMETRY] FLUSH TO DISK
    # Saves Flows and Agent States to Parquet
    # =======================================================
    flush_step_telemetry(current_step, current_year, SIMULATION_EXPORT_DIR)

logger.success("Multi-year Simulation Completed")
