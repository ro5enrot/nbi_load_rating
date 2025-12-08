#!/usr/bin/env python3
"""
Inventory Rating 数据清洗 Pipeline 的第 3 阶段：数值异常检测（Outlier Cleaning）。

职责：
1. 基于 Stage 2 输出的 `clean_stage2_inventory_rating.csv`，
   用 IQR 统计规则与工程规则对 `INVENTORY_RATING_066` 进行数值异常检测；
2. 输出：
   - `outlier_records_stage3.csv`：所有异常记录（含 `outlier_type` 字段）；
   - `clean_stage3_inventory_rating.csv`：剔除异常后的 clean 数据；
   - `outlier_summary_stage3.yaml`：结构化统计结果；
3. 不执行缺失值处理、逻辑异常判断、特征工程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# Stage 2 输出文件名（默认与本脚本同目录）
STAGE2_CLEAN_FILENAME = "clean_stage2_inventory_rating.csv"

# Stage 3 输出文件名
OUTLIER_RECORDS_FILENAME = "outlier_records_stage3.csv"
CLEAN_STAGE3_FILENAME = "clean_stage3_inventory_rating.csv"
OUTLIER_SUMMARY_FILENAME = "outlier_summary_stage3.yaml"

# 关键字段
STRUCTURE_COL = "STRUCTURE_NUMBER_008"
METHOD_COL = "INV_RATING_METH_065"
RATING_COL = "INVENTORY_RATING_066"
RATING_YEAR_COL = "rating_year"

# 支持的 Inventory Rating 方法
TARGET_METHODS = {"3", "4"}


def load_stage2_clean_data(script_dir: Path) -> pd.DataFrame:
    """
    读取 Stage 2 输出的 `clean_stage2_inventory_rating.csv`，并校验关键字段是否存在。
    """
    path = script_dir / STAGE2_CLEAN_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 Stage 2 输出文件：{path}，请先完成缺失值清洗阶段。"
        )

    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    required_cols = [
        STRUCTURE_COL,
        METHOD_COL,
        RATING_COL,
        RATING_YEAR_COL,
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Stage 2 输出文件缺少必要字段：{', '.join(missing_cols)}")
    return df


def parse_numeric_rating(series: pd.Series) -> pd.Series:
    """
    将 `INVENTORY_RATING_066` 解析为浮点数，无法解析的值置为 NaN。
    """
    def _parse(value: object) -> float | np.nan:
        if value is None:
            return np.nan
        text = str(value).strip()
        if not text or text.upper() == "N":
            return np.nan
        try:
            return float(text)
        except ValueError:
            return np.nan

    return series.map(_parse).astype(float)


def detect_iqr_outliers(rating_series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """
    使用 IQR（四分位距）规则检测全局数值异常：
    - Q1, Q3 分别为 25%、75% 分位数；
    - IQR = Q3 - Q1；
    - 低阈值 = Q1 - 1.5 * IQR；
    - 高阈值 = Q3 + 1.5 * IQR；
    - 落在阈值之外的记录视为 IQR 异常。
    返回：bool 序列（与输入索引对齐）、阈值字典。
    """
    valid = rating_series.dropna()
    if valid.empty:
        raise ValueError("INVENTORY_RATING_066 全部缺失，无法计算 IQR 阈值。")

    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (rating_series < lower_bound) | (rating_series > upper_bound)
    mask = mask.fillna(False)

    thresholds = {
        "Q1": q1,
        "Q3": q3,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }
    return mask, thresholds


def detect_engineering_rule_outliers(rating_series: pd.Series) -> Dict[str, pd.Series]:
    """
    基于工程经验规则检测异常：
    - rating <= 0 → engineer_low_outlier（需剔除）
    - rating > 200 → engineer_high_outlier（需剔除）
    - 0 < rating < 1 → engineer_suspicious_low（仅记录，不剔除）
    返回包含三种掩码的字典。
    """
    mask_low = rating_series <= 0
    mask_high = rating_series > 200
    mask_suspicious = (rating_series > 0) & (rating_series < 1)

    result = {
        "engineer_low_outlier": mask_low.fillna(False),
        "engineer_high_outlier": mask_high.fillna(False),
        "engineer_suspicious_low": mask_suspicious.fillna(False),
    }
    return result


def combine_outliers(
    df: pd.DataFrame,
    iqr_mask: pd.Series,
    engineer_masks: Dict[str, pd.Series],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    汇总 IQR 和工程规则异常，生成 outlier DataFrame、clean DataFrame 与统计信息。
    - 多个异常可同时出现，用逗号拼接 outlier_type；
    - 需要剔除的类型：IQR_outlier、engineer_low_outlier、engineer_high_outlier；
    - engineer_suspicious_low 仅记录，不剔除。
    """
    type_masks = {
        "IQR_outlier": iqr_mask.astype(bool),
        "engineer_low_outlier": engineer_masks["engineer_low_outlier"].astype(bool),
        "engineer_high_outlier": engineer_masks["engineer_high_outlier"].astype(bool),
        "engineer_suspicious_low": engineer_masks["engineer_suspicious_low"].astype(bool),
    }
    type_flags = pd.DataFrame(type_masks, index=df.index)

    # 构造 outlier_type 字段
    outlier_mask = type_flags.any(axis=1)
    outlier_df = df[outlier_mask].copy()
    if not outlier_df.empty:
        outlier_types = type_flags.loc[outlier_df.index].apply(
            lambda row: ",".join(row.index[row.values.astype(bool)]),
            axis=1,
        )
        outlier_df["outlier_type"] = outlier_types.values
    else:
        outlier_df["outlier_type"] = []

    # 需要剔除的异常类型
    removal_mask = (
        type_flags["IQR_outlier"]
        | type_flags["engineer_low_outlier"]
        | type_flags["engineer_high_outlier"]
    )
    clean_df = df[~removal_mask].copy()

    # 汇总统计
    type_counts = {name: int(mask.sum()) for name, mask in type_masks.items()}
    total_records = int(len(df))
    total_removed = int(removal_mask.sum())

    method_breakdown = {}
    method_series = df[METHOD_COL].astype(str).str.strip()
    for method in sorted(TARGET_METHODS):
        method_mask = method_series == method
        method_total = int(method_mask.sum())
        method_removed_mask = removal_mask & method_mask
        method_removed = int(method_removed_mask.sum())

        removed_by_type = {}
        for name in ("IQR_outlier", "engineer_low_outlier", "engineer_high_outlier"):
            removed_by_type[name] = int((type_flags[name] & method_mask).sum())

        method_breakdown[f"method_{method}"] = {
            "total": method_total,
            "removed_outliers": method_removed,
            "removed_by_type": removed_by_type,
        }

    stats = {
        "total_records": total_records,
        "total_removed": total_removed,
        "type_counts": type_counts,
        "method_breakdown": method_breakdown,
    }
    return outlier_df, clean_df, stats


def write_outlier_csv(path: Path, df: pd.DataFrame) -> None:
    """
    将异常记录写出为 CSV，包含字段 outlier_type。
    """
    df.to_csv(path, index=False, encoding="utf-8")


def write_clean_csv(path: Path, df: pd.DataFrame) -> None:
    """
    将 Stage 3 清洗后的数据写出为 CSV。
    """
    df.to_csv(path, index=False, encoding="utf-8")


def write_summary_yaml(
    path: Path,
    thresholds: Dict[str, float],
    stats: Dict,
) -> None:
    """
    生成 Stage 3 数值异常检测的 summary YAML。

    YAML 包含：
    - IQR 阈值信息；
    - 各类型异常数量、总剔除数量；
    - 每个方法的异常概览。
    """
    summary = {
        "iqr_thresholds": thresholds,
        "counts": {
            "total_records": stats["total_records"],
            "total_outliers_removed": stats["total_removed"],
            "outliers_by_type": stats["type_counts"],
        },
        "method_breakdown": stats["method_breakdown"],
    }
    yaml_body = yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    comment = (
        "# iqr_thresholds：根据全局 INVENTORY_RATING_066 计算的 IQR 阈值；\n"
        "# counts：各异常类型的记录数量以及被剔除的总数；\n"
        "# method_breakdown：按方法（3/4）划分的异常与剔除情况。\n"
    )
    path.write_text(comment + yaml_body, encoding="utf-8")


def main() -> None:
    """
    执行 Inventory Rating 数据清洗 Pipeline 的第 3 阶段：数值异常检测。
    处理顺序：
    1. 读取 Stage 2 clean 数据；
    2. 解析 Inventory Rating 数值；
    3. 检测 IQR 异常与工程规则异常；
    4. 合并异常记录、剔除需删除的异常；
    5. 写出异常记录 CSV、clean CSV、summary YAML。
    """
    script_dir = Path(__file__).resolve().parent

    # 读取 Stage 2 数据
    df_stage2 = load_stage2_clean_data(script_dir)
    rating_numeric = parse_numeric_rating(df_stage2[RATING_COL])
    df_stage2["_rating_numeric"] = rating_numeric

    # 检测 IQR 异常
    iqr_mask, thresholds = detect_iqr_outliers(rating_numeric)

    # 检测工程规则异常
    engineer_masks = detect_engineering_rule_outliers(rating_numeric)

    # 合并异常并剔除
    outlier_df, clean_df, stats = combine_outliers(df_stage2, iqr_mask, engineer_masks)

    # 写出 CSV
    outlier_export = outlier_df.drop(columns=["_rating_numeric"], errors="ignore")
    write_outlier_csv(script_dir / OUTLIER_RECORDS_FILENAME, outlier_export)
    # 清理内部临时列
    clean_export = clean_df.drop(columns=["_rating_numeric"])
    write_clean_csv(script_dir / CLEAN_STAGE3_FILENAME, clean_export)

    # 写出 summary
    write_summary_yaml(script_dir / OUTLIER_SUMMARY_FILENAME, thresholds, stats)


if __name__ == "__main__":
    main()
