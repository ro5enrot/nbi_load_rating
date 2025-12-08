#!/usr/bin/env python3
"""
Inventory Rating 数据清洗 Pipeline 的第 1 阶段：逻辑异常清洗（Logical Anomaly Cleaning）。

本脚本用于在原始 NBI 文本数据基础上，对 Inventory Rating（仅方法 3 和 4）
进行基于时间逻辑的异常值剔除，输出：

1. `rating_increase_records.csv`：
   同一桥、同一方法下，相邻年份 Inventory Rating 反常升高的记录明细。
2. `reconstructed_after_first_rating_records.csv`：
   重建年份晚于首次承载力年份的记录明细。
3. `clean_stage1_inventory_rating.csv`：
   剔除上述两类逻辑异常记录后的“逻辑一致的基础样本集”，
   作为后续缺失值处理、数值异常检测和特征工程的输入。
4. `logical_anomaly_summary_stage1.yaml`：
   对两类逻辑异常在不同方法（3 / 4）下的记录数和桥梁数量进行汇总。

说明：
- 输入为目录 `../all_States_in_a_single_file_raw/` 中的原始 NBI 文本；
- 本脚本只负责“逻辑异常清洗”，不做缺失值填补、不做数值极端值检测、
  不做特征工程，这些将在后续 pipeline 脚本中完成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


# 原始 NBI 文本所在目录名与文件模式
DATA_DIR_NAME = "all_States_in_a_single_file_raw"
CSV_GLOB = "NBI_*_Delimited_AllStates.txt"

# 只处理 Inventory Rating 方法 3 和 4
TARGET_METHODS = {"3", "4"}

# 关键字段名称常量，便于统一管理
STRUCTURE_COL = "STRUCTURE_NUMBER_008"
METHOD_COL = "INV_RATING_METH_065"
RATING_COL = "INVENTORY_RATING_066"
RECONSTRUCT_COL = "YEAR_RECONSTRUCTED_106"
RATING_YEAR_COL = "rating_year"
SOURCE_FILE_COL = "source_file"
ROW_ID_COL = "_row_id"


def load_inventory_rating_data(data_dir: Path) -> pd.DataFrame:
    """
    读取原始 NBI 文本数据，筛选 Inventory Rating 方法为 3/4 的记录，并按行拼接为一个 DataFrame。

    处理要点：
    - 遍历目录下所有 `NBI_*_Delimited_AllStates.txt` 文件；
    - 从文件名中解析年份，作为本条记录的承载力年份字段 `rating_year`；
    - 仅保留 `INV_RATING_METH_065` 为 3 或 4 的记录；
    - 保留原始字段结构，并额外增加 `rating_year`、`source_file` 两个辅助字段；
    - 不做缺失值填补、不做数值异常检测。
    """
    files = sorted(data_dir.glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(
            f"未在目录 {data_dir} 中找到匹配模式 {CSV_GLOB!r} 的 NBI 原始文本文件"
        )

    frames: List[pd.DataFrame] = []

    for path in files:
        name = path.name
        try:
            # 约定文件名形如 NBI_2019_Delimited_AllStates.txt，从中解析承载力年份
            rating_year = int(name.split("_")[1])
        except (IndexError, ValueError):
            # 文件名不符合预期时，跳过该文件，避免引入不明年份的数据
            continue

        # 以字符串读取所有列，保持原始信息，后续按需转换
        df = pd.read_csv(path, dtype=str, encoding="latin-1")

        # 若关键字段缺失，说明原始文件不符合 NBI 标准，直接报错提示
        for col in (STRUCTURE_COL, METHOD_COL, RATING_COL, RECONSTRUCT_COL):
            if col not in df.columns:
                raise ValueError(f"文件 {name} 缺少必要字段 {col}")

        # 只保留 Inventory Rating 方法为 3 或 4 的记录
        method_series = df[METHOD_COL].astype(str).str.strip()
        df = df[method_series.isin(TARGET_METHODS)].copy()
        if df.empty:
            continue

        df[RATING_YEAR_COL] = rating_year
        df[SOURCE_FILE_COL] = name
        frames.append(df)

    if not frames:
        raise ValueError("在原始文件中未找到任何方法为 3 或 4 的 Inventory Rating 记录")

    all_df = pd.concat(frames, ignore_index=True)
    # 增加内部用行标识，便于后续从全集中剔除逻辑异常记录
    all_df[ROW_ID_COL] = all_df.index
    return all_df


def _parse_float(value: object) -> Optional[float]:
    """将单元格值解析为浮点数；空值或非数值返回 None（不做缺失值填补）。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: object) -> Optional[int]:
    """将单元格值解析为整数；空值或非数值返回 None（不做缺失值填补）。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def detect_rating_increase_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    检测“相邻年份承载力升高”的逻辑异常记录（规则 1）。

    对每一个 (STRUCTURE_NUMBER_008, INV_RATING_METH_065) 组合：
    - 按 `rating_year` 升序排序；
    - 第一条记录只用于初始化 `prev_rating`，不做异常判定；
    - 之后每条记录若当前 Inventory Rating 为数值，
      且上一条记录的 Inventory Rating 也是数值且严格更小，
      则当前记录视为“相邻年份承载力异常升高”。

    返回：
    - 包含所有满足条件记录的 DataFrame，保留原始字段结构，
      并包含 `rating_year`、`source_file` 等辅助字段。
    """
    if df.empty:
        return df.copy()

    anomaly_rows: List[int] = []

    # 按桥梁编号 + 方法分组，在组内进行时间序列扫描
    grouped = df.groupby([STRUCTURE_COL, METHOD_COL], dropna=False)
    for (_, _), group in grouped:
        # 按承载力年份升序排列，使用稳定排序保持同年内原始顺序
        group_sorted = group.sort_values(RATING_YEAR_COL, kind="mergesort")

        prev_rating: Optional[float] = None
        first_row = True

        for idx, row in group_sorted.iterrows():
            current_rating = _parse_float(row.get(RATING_COL))

            if first_row:
                # 第一条记录只用于初始化 prev_rating，不做“升高异常”判定
                first_row = False
                if current_rating is not None:
                    prev_rating = current_rating
                continue

            # 若当前值与前一条均为数值且严格升高，则视为逻辑异常
            if (
                current_rating is not None
                and prev_rating is not None
                and current_rating > prev_rating
            ):
                anomaly_rows.append(idx)

            # 若当前值为数值，则更新 prev_rating，用于后续相邻年份比较
            if current_rating is not None:
                prev_rating = current_rating

    if not anomaly_rows:
        return df.iloc[0:0].copy()

    return df.loc[sorted(set(anomaly_rows))].copy()


def detect_reconstruction_after_first_rating_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    检测“重建年份晚于首次承载力年份”的逻辑异常记录（规则 2）。

    对每一个 (STRUCTURE_NUMBER_008, INV_RATING_METH_065) 组合：
    - 仍按 `rating_year` 升序排序；
    - 取排序后第一条记录的 `rating_year` 作为首次承载力年份 first_rating_year；
    - 对该桥该方法下所有记录：
      若 `YEAR_RECONSTRUCTED_106` 为数值且大于 first_rating_year，
      则该记录视为“重建年份晚于首次承载力年份”的逻辑异常。

    返回：
    - 所有满足条件记录的 DataFrame，保留原始字段结构，
      并包含 `rating_year`、`source_file` 等辅助字段。
    """
    if df.empty:
        return df.copy()

    anomaly_rows: List[int] = []

    grouped = df.groupby([STRUCTURE_COL, METHOD_COL], dropna=False)
    for (_, _), group in grouped:
        group_sorted = group.sort_values(RATING_YEAR_COL, kind="mergesort")
        if group_sorted.empty:
            continue

        # 首次承载力年份来自排序后第一条记录的 rating_year
        first_rating_year = _parse_int(group_sorted.iloc[0][RATING_YEAR_COL])
        if first_rating_year is None:
            # 正常情况下 rating_year 应该总是可解析为整数，如遇异常直接跳过该组
            continue

        for idx, row in group_sorted.iterrows():
            recon_year = _parse_int(row.get(RECONSTRUCT_COL))
            if recon_year is None:
                continue
            if recon_year > first_rating_year:
                anomaly_rows.append(idx)

    if not anomaly_rows:
        return df.iloc[0:0].copy()

    return df.loc[sorted(set(anomaly_rows))].copy()


def build_clean_dataset(
    all_df: pd.DataFrame,
    rating_increase_df: pd.DataFrame,
    reconstructed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建“逻辑清洗后的基础数据集”（仅剔除两类逻辑异常记录）。

    处理要点：
    - 从全集数据中剔除所有在 `rating_increase_df` 和 `reconstructed_df` 中出现过的记录；
    - 剩余记录视为在时间逻辑上自洽的 Inventory Rating 观测；
    - 不进行缺失值填补、不进行数值异常检测和特征工程。
    """
    if all_df.empty:
        return all_df.copy()

    bad_ids: set = set()
    for subset in (rating_increase_df, reconstructed_df):
        if subset is not None and not subset.empty and ROW_ID_COL in subset.columns:
            bad_ids.update(subset[ROW_ID_COL].tolist())

    if not bad_ids or ROW_ID_COL not in all_df.columns:
        return all_df.copy()

    mask = ~all_df[ROW_ID_COL].isin(bad_ids)
    clean_df = all_df.loc[mask].copy()
    return clean_df


def _summarize_anomalies_by_method(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    按 Inventory Rating 方法统计某一类逻辑异常的记录数和涉及的桥梁数量。

    返回结构：
    {
        "3": {"num_records": int, "num_structures": int},
        "4": {"num_records": int, "num_structures": int},
    }
    """
    summary: Dict[str, Dict[str, int]] = {
        method: {"num_records": 0, "num_structures": 0} for method in TARGET_METHODS
    }
    if df is None or df.empty:
        return summary

    for method in TARGET_METHODS:
        method_series = df[METHOD_COL].astype(str).str.strip()
        subset = df[method_series == method]
        if subset.empty:
            continue
        summary[method]["num_records"] = int(len(subset))
        summary[method]["num_structures"] = int(subset[STRUCTURE_COL].nunique())
    return summary


def write_dataframe_csv(path: Path, df: pd.DataFrame, drop_internal_cols: bool = True) -> None:
    """
    将 DataFrame 写出为 CSV 文件。

    - 默认会剔除仅用于内部处理的列（如 `_row_id`）；
    - 其他列保持不变，便于后续 pipeline 脚本继续使用原始结构和辅助字段。
    """
    if drop_internal_cols and ROW_ID_COL in df.columns:
        df = df.drop(columns=[ROW_ID_COL])
    # 使用 UTF-8 编码写出，保留列名与数据
    df.to_csv(path, index=False, encoding="utf-8")


def write_summary_yaml(
    path: Path,
    rating_increase_summary: Dict[str, Dict[str, int]],
    reconstructed_summary: Dict[str, Dict[str, int]],
) -> None:
    """
    生成 Stage 1 的逻辑异常汇总 YAML 文件，按方法维度汇总两类逻辑异常的记录数与桥梁数。

    YAML 结构示意：
    - method_3:
        rating_increase:
            num_records: ...
            num_structures: ...
        reconstructed_after_first_rating:
            num_records: ...
            num_structures: ...
    - method_4:
        ...

    并在文件顶部以注释形式说明两类逻辑异常指标的定义。
    """
    summary = {
        "method_3": {
            "rating_increase": rating_increase_summary.get("3", {}),
            "reconstructed_after_first_rating": reconstructed_summary.get("3", {}),
        },
        "method_4": {
            "rating_increase": rating_increase_summary.get("4", {}),
            "reconstructed_after_first_rating": reconstructed_summary.get("4", {}),
        },
    }

    yaml_body = yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    comment = (
        "# rating_increase：同一桥、同一方法下，当前年份的 INVENTORY_RATING_066\n"
        "#   严格大于上一条记录（相邻年份）中的 INVENTORY_RATING_066。\n"
        "# reconstructed_after_first_rating：YEAR_RECONSTRUCTED_106 大于该桥该方法\n"
        "#   首次出现的承载力年份（rating_year）的记录。\n"
    )
    path.write_text(comment + yaml_body, encoding="utf-8")


def main() -> None:
    """
    脚本入口：执行 Inventory Rating 数据清洗 pipeline 的第 1 阶段（逻辑异常清洗）。

    步骤概览：
    1. 读取 ../all_States_in_a_single_file_raw/ 目录下的原始 NBI 文本；
    2. 仅保留 Inventory Rating 方法 3 和 4 的记录，并构建总表 DataFrame；
    3. 按规则 1、规则 2 检测两类时间逻辑异常，并输出对应明细 CSV；
    4. 从全集中剔除这两类逻辑异常记录，得到 clean 基础样本集并输出 CSV；
    5. 根据异常明细生成 `logical_anomaly_summary_stage1.yaml`，
       为后续分析与审计提供统计信息。
    """
    script_dir = Path(__file__).resolve().parent
    # 默认读取目录为脚本上级的 ../all_States_in_a_single_file_raw/
    data_dir = script_dir.parent.parent / DATA_DIR_NAME

    all_df = load_inventory_rating_data(data_dir)

    rating_increase_df = detect_rating_increase_anomalies(all_df)
    reconstructed_df = detect_reconstruction_after_first_rating_anomalies(all_df)

    clean_df = build_clean_dataset(all_df, rating_increase_df, reconstructed_df)

    # 写出三类 CSV：两类异常明细 + clean 基础数据
    write_dataframe_csv(script_dir / "rating_increase_records.csv", rating_increase_df)
    write_dataframe_csv(
        script_dir / "reconstructed_after_first_rating_records.csv",
        reconstructed_df,
    )
    write_dataframe_csv(script_dir / "clean_stage1_inventory_rating.csv", clean_df)

    # 生成 Stage 1 逻辑异常 summary YAML 统计
    rating_increase_summary = _summarize_anomalies_by_method(rating_increase_df)
    reconstructed_summary = _summarize_anomalies_by_method(reconstructed_df)
    write_summary_yaml(
        script_dir / "logical_anomaly_summary_stage1.yaml",
        rating_increase_summary,
        reconstructed_summary,
    )


if __name__ == "__main__":
    main()
