#!/usr/bin/env python3
"""
Inventory Rating 数据清洗 Pipeline 的第 2 阶段：缺失值清洗。

本阶段仅处理一件事：在 Stage 1 输出的 `clean_stage1_inventory_rating.csv` 基础上，
识别并删除所有 `INVENTORY_RATING_066`（标签）缺失的记录，输出：

1. `clean_stage2_inventory_rating.csv`：删除标签缺失记录后的 clean 数据；
2. `missing_value_records_stage2.csv`：汇总被删除的标签缺失记录；
3. `missing_value_summary_stage2.yaml`：带中文注释的统计信息。

本脚本不做逻辑异常检查、不做缺失值填补、不做特征工程，确保 Stage 2
的职责清晰聚焦在“删除标签缺失记录”上。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml


# 输入 / 输出文件名
STAGE1_CLEAN_FILENAME = "clean_stage1_inventory_rating.csv"
STAGE2_CLEAN_FILENAME = "clean_stage2_inventory_rating.csv"
MISSING_RECORDS_FILENAME = "missing_value_records_stage2.csv"
STAGE2_SUMMARY_FILENAME = "missing_value_summary_stage2.yaml"

# 关键字段
METHOD_COL = "INV_RATING_METH_065"
RATING_COL = "INVENTORY_RATING_066"

# 仅统计方法 3 和 4
TARGET_METHODS = {"3", "4"}


def load_stage1_clean_data(script_dir: Path) -> pd.DataFrame:
    """
    读取 Stage 1 输出的 `clean_stage1_inventory_rating.csv`，确保最少字段存在。
    """
    path = script_dir / STAGE1_CLEAN_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 Stage 1 输出文件：{path}，请先运行逻辑异常清洗脚本。"
        )
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    if RATING_COL not in df.columns or METHOD_COL not in df.columns:
        raise ValueError("Stage 1 输出缺少必需字段，无法执行 Stage 2 缺失值清洗。")
    return df


def _is_missing_numeric_like(series: pd.Series) -> pd.Series:
    """
    判断数值型字符串列是否缺失：NaN、空字符串、只有空白、'N'/'N/A'/'-' 等非数值。
    """
    text = series.astype(str).str.strip()
    missing_mask = series.isna() | text.eq("") | text.str.upper().isin({"N", "N/A", "-"})
    return missing_mask


def analyze_missing_stats(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    统计标签缺失情况，包括总体记录数、缺失数，以及按方法 3/4 的缺失统计。
    """
    total_records = int(len(df))
    missing_mask = _is_missing_numeric_like(df[RATING_COL])
    missing_count = int(missing_mask.sum())

    per_method: Dict[str, Dict[str, int]] = {}
    method_series = df[METHOD_COL].astype(str).str.strip()
    for method in sorted(TARGET_METHODS):
        subset_mask = method_series == method
        subset_total = int(subset_mask.sum())
        subset_missing = int((missing_mask & subset_mask).sum())
        per_method[f"method_{method}"] = {
            "total_records": subset_total,
            "missing_inventory_rating_066": subset_missing,
        }

    return {
        "total_records": total_records,
        "missing_inventory_rating_066": missing_count,
        "per_method": per_method,
    }


def drop_missing_label_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    删除所有 `INVENTORY_RATING_066` 缺失的记录，并返回 (clean_df, dropped_df)。
    """
    missing_mask = _is_missing_numeric_like(df[RATING_COL])
    dropped_df = df[missing_mask].copy()
    clean_df = df[~missing_mask].copy()
    return clean_df, dropped_df


def write_clean_csv(path: Path, df: pd.DataFrame) -> None:
    """
    将删除标签缺失后的 clean 数据写出为 CSV。
    """
    df.to_csv(path, index=False, encoding="utf-8")


def write_missing_records_csv(path: Path, df: pd.DataFrame) -> None:
    """
    将所有标签缺失且被删除的记录写出为 CSV，方便后续溯源。
    """
    df.to_csv(path, index=False, encoding="utf-8")


def write_summary_yaml(
    path: Path,
    before_stats: Dict[str, Dict[str, int]],
    after_stats: Dict[str, Dict[str, int]],
    dropped_count: int,
) -> None:
    """
    生成 Stage 2 的汇总 YAML：仅关注标签缺失的删除情况，并附中文注释。
    """
    per_method_section = {}
    for method in sorted(TARGET_METHODS):
        key = f"method_{method}"
        per_method_section[key] = {
            "before": {
                "total_records": before_stats["per_method"][key]["total_records"],
                "missing_inventory_rating_066": before_stats["per_method"][key][
                    "missing_inventory_rating_066"
                ],
            },
            "after": {
                "total_records": after_stats["per_method"][key]["total_records"],
                "missing_inventory_rating_066": after_stats["per_method"][key][
                    "missing_inventory_rating_066"
                ],
            },
        }

    summary = {
        "summary": {
            "total_records_before": before_stats["total_records"],
            "total_records_after": after_stats["total_records"],
        },
        "fields_missing": {
            "before": {"INVENTORY_RATING_066": before_stats["missing_inventory_rating_066"]},
            "after": {"INVENTORY_RATING_066": after_stats["missing_inventory_rating_066"]},
        },
        "dropped_records": {
            "missing_inventory_rating_066": dropped_count,
        },
        "per_method_missing": per_method_section,
    }

    yaml_body = yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    comment = (
        "# summary：记录 Stage 2 删除标签缺失前后的样本数量；\n"
        "# fields_missing：INVENTORY_RATING_066 在清洗前后的缺失数量（清洗后应为 0）；\n"
        "# dropped_records：因标签缺失被删除的记录数量；\n"
        "# per_method_missing：按方法 3/4 统计标签缺失与样本数量的变化。\n"
    )
    path.write_text(comment + yaml_body, encoding="utf-8")


def main() -> None:
    """
    执行 Stage 2：删除所有 Inventory Rating 标签缺失的记录，并输出相应统计。
    """
    script_dir = Path(__file__).resolve().parent

    # 读取 Stage 1 clean 数据
    df_stage1 = load_stage1_clean_data(script_dir)

    # 处理前的标签缺失统计
    before_stats = analyze_missing_stats(df_stage1)

    # 删除标签缺失记录，并导出缺失样本
    clean_df, dropped_df = drop_missing_label_records(df_stage1)
    dropped_count = int(len(dropped_df))
    write_missing_records_csv(script_dir / MISSING_RECORDS_FILENAME, dropped_df)

    # 处理后的标签缺失统计（应无缺失）
    after_stats = analyze_missing_stats(clean_df)

    # 写出清洗后的 CSV
    write_clean_csv(script_dir / STAGE2_CLEAN_FILENAME, clean_df)

    # 写出 YAML 汇总
    write_summary_yaml(
        script_dir / STAGE2_SUMMARY_FILENAME,
        before_stats=before_stats,
        after_stats=after_stats,
        dropped_count=dropped_count,
    )


if __name__ == "__main__":
    main()

