#!/usr/bin/env python3
"""
Inventory Rating 数据清洗 Pipeline 的第 2 阶段：缺失值清洗。

本脚本在第 1 阶段输出的 `clean_inventory_rating_records.csv` 基础上，
针对关键字段进行缺失值检测与处理，输出：

1. `clean_stage2_inventory_rating.csv`：
   - 在剔除无法建模的记录后，
   - 对 INVENTORY_RATING_066 进行“同桥 + 同方法”维度的均值填补，
   - 新增字段 `imputed_flag` 标记是否为填补值（0：原始值；1：被填补）。

2. `missing_value_summary_stage2.yaml`：
   - 各关键字段缺失值数量（处理前 vs 处理后）；
   - 均值填补的记录数量；
   - 被剔除记录数量（按原因分类：缺结构号、缺方法、缺 rating_year、无可用均值）；
   - 按方法（3、4）拆分的缺失情况统计。

本脚本仅负责“缺失值清洗”：
- 不再进行逻辑异常判断（已由 Pipeline 第 1 阶段处理）；
- 不进行数值极端值检测；
- 不进行特征工程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


# 第 1 阶段输出的 clean 数据文件名（默认与本脚本同目录）
STAGE1_CLEAN_FILENAME = "clean_inventory_rating_records.csv"

# 第 2 阶段输出文件名
STAGE2_CLEAN_FILENAME = "clean_stage2_inventory_rating.csv"
STAGE2_SUMMARY_FILENAME = "missing_value_summary_stage2.yaml"


# 关键字段名称常量
STRUCTURE_COL = "STRUCTURE_NUMBER_008"
METHOD_COL = "INV_RATING_METH_065"
RATING_COL = "INVENTORY_RATING_066"
RECONSTRUCT_COL = "YEAR_RECONSTRUCTED_106"
RATING_YEAR_COL = "rating_year"
IMPUTED_FLAG_COL = "imputed_flag"

# 仅统计方法 3 和 4 的缺失情况
TARGET_METHODS = {"3", "4"}


def _is_missing_str(series: pd.Series) -> pd.Series:
    """
    对字符串/对象列定义缺失：
    - NaN；
    - 空字符串；
    - 仅包含空白字符。
    """
    return series.isna() | series.astype(str).str.strip().eq("")


def _is_missing_numeric_like(series: pd.Series) -> pd.Series:
    """
    对数值型字符串列定义缺失：
    - NaN；
    - 空字符串或仅空白；
    - 特殊占位符 'N'（在 NBI 数据中常表示缺失）。
    """
    s = series.astype(str)
    return series.isna() | s.str.strip().eq("") | s.str.strip().str.upper().eq("N")


def _parse_float(value) -> float | None:
    """
    将单元格解析为浮点数。
    - 空值、'N' 或非法数值返回 None；
    - 本函数仅用于计算组内均值，不进行缺失值填补。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_stage1_clean_data(script_dir: Path) -> pd.DataFrame:
    """
    读取第 1 阶段生成的 clean_inventory_rating_records.csv。

    要求：
    - 文件路径默认为与本脚本同目录的 `clean_inventory_rating_records.csv`；
    - 至少包含以下字段：
      - STRUCTURE_NUMBER_008
      - INV_RATING_METH_065
      - INVENTORY_RATING_066
      - YEAR_RECONSTRUCTED_106
      - rating_year
    """
    path = script_dir / STAGE1_CLEAN_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 Stage 1 输出文件：{path}，请先运行第 1 阶段逻辑异常清洗脚本。"
        )

    df = pd.read_csv(path, dtype=str, encoding="utf-8")

    required_cols = [
        STRUCTURE_COL,
        METHOD_COL,
        RATING_COL,
        RECONSTRUCT_COL,
        RATING_YEAR_COL,
    ]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Stage 1 输出文件缺少必要字段：{', '.join(missing_required)}"
        )

    return df


def analyze_missing(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    对关键字段进行缺失值统计（不做任何修改）。

    统计内容：
    - 按字段汇总缺失数量；
    - 按方法（3、4）拆分缺失情况。

    返回字典结构示例：
    {
        "by_field": {
            "STRUCTURE_NUMBER_008": 10,
            "INV_RATING_METH_065": 5,
            "rating_year": 0,
            "INVENTORY_RATING_066": 100,
            "YEAR_RECONSTRUCTED_106": 50,
        },
        "by_method": {
            "3": {
                "total_records": ...,
                "missing": {
                    "STRUCTURE_NUMBER_008": ...,
                    ...
                },
            },
            "4": { ... },
        },
    }
    """
    fields = [
        STRUCTURE_COL,
        METHOD_COL,
        RATING_YEAR_COL,
        RATING_COL,
        RECONSTRUCT_COL,
    ]

    # 字段整体缺失统计
    by_field: Dict[str, int] = {}
    for col in fields:
        if col in (RATING_COL, RECONSTRUCT_COL):
            missing_mask = _is_missing_numeric_like(df[col])
        else:
            missing_mask = _is_missing_str(df[col])
        by_field[col] = int(missing_mask.sum())

    # 按方法拆分的缺失统计
    by_method: Dict[str, Dict] = {}
    method_series = df[METHOD_COL].astype(str).str.strip()
    for method in TARGET_METHODS:
        subset = df[method_series == method]
        if subset.empty:
            by_method[method] = {
                "total_records": 0,
                "missing": {col: 0 for col in fields},
            }
            continue

        method_missing: Dict[str, int] = {}
        for col in fields:
            if col in (RATING_COL, RECONSTRUCT_COL):
                mask = _is_missing_numeric_like(subset[col])
            else:
                mask = _is_missing_str(subset[col])
            method_missing[col] = int(mask.sum())

        by_method[method] = {
            "total_records": int(len(subset)),
            "missing": method_missing,
        }

    return {"by_field": by_field, "by_method": by_method}


def drop_invalid_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    根据规则 1 和规则 2（部分）剔除无法用于建模的记录。

    规则：
    1. STRUCTURE_NUMBER_008 缺失 → 剔除；
    2. INV_RATING_METH_065 缺失 → 剔除；
    3. rating_year 缺失 → 剔除；
       （以上按优先级赋予唯一剔除原因，避免重复计数）
    4. 对于某一 (STRUCTURE_NUMBER_008, INV_RATING_METH_065) 组合，
       若所有年份的 INVENTORY_RATING_066 均缺失（无法计算组内均值），
       则整个组合内的记录全部剔除，原因记为“no_available_mean”。

    返回：
    - 剩余可用于缺失值填补的数据 DataFrame；
    - 剔除记录数量的统计字典：
      {
          "missing_structure_number": ...,
          "missing_method": ...,
          "missing_rating_year": ...,
          "no_available_mean": ...,
      }
    """
    df = df.copy()

    # 三类必需字段缺失（按优先级赋予唯一原因）
    missing_structure = _is_missing_str(df[STRUCTURE_COL])
    missing_method = _is_missing_str(df[METHOD_COL])
    missing_rating_year = _is_missing_str(df[RATING_YEAR_COL])

    drop_reason = pd.Series("", index=df.index, dtype="object")
    drop_reason[missing_structure] = "missing_structure_number"
    drop_reason[(drop_reason == "") & missing_method] = "missing_method"
    drop_reason[(drop_reason == "") & missing_rating_year] = "missing_rating_year"

    stats = {
        "missing_structure_number": int((drop_reason == "missing_structure_number").sum()),
        "missing_method": int((drop_reason == "missing_method").sum()),
        "missing_rating_year": int((drop_reason == "missing_rating_year").sum()),
        "no_available_mean": 0,
    }

    df_valid = df[drop_reason == ""].copy()

    if df_valid.empty:
        return df_valid, stats

    # 规则 2 中“该桥 + 方法全为缺失 → 全部剔除”
    # 先解析数值，便于判断组内是否存在任何有效值
    rating_numeric = df_valid[RATING_COL].map(_parse_float)
    df_valid["_rating_numeric_tmp"] = rating_numeric

    # 按桥 + 方法分组，判断是否“全部为缺失”
    group = df_valid.groupby([STRUCTURE_COL, METHOD_COL], dropna=False)["_rating_numeric_tmp"]
    has_any_value = group.transform(lambda s: s.notna().any())

    # 所有值均缺失的组合会得到 has_any_value == False
    group_all_missing_mask = ~has_any_value
    stats["no_available_mean"] = int(group_all_missing_mask.sum())

    df_valid = df_valid[~group_all_missing_mask].copy()
    df_valid = df_valid.drop(columns=["_rating_numeric_tmp"])

    return df_valid, stats


def impute_inventory_rating(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    对剩余数据中的 INVENTORY_RATING_066 进行组内均值填补。

    规则：
    - 对每一个 (STRUCTURE_NUMBER_008, INV_RATING_METH_065) 组合，
      使用该组合内已有的非缺失数值计算均值；
    - 对该组合中缺失的 INVENTORY_RATING_066，用组内均值进行填补；
    - 若某组合内全部为缺失，已在 drop_invalid_records 阶段整体剔除；
    - 新增列 `imputed_flag`，原始值为 0，被填补的记录为 1。
    """
    if df.empty:
        df_empty = df.copy()
        df_empty[IMPUTED_FLAG_COL] = 0
        return df_empty, 0

    df = df.copy()

    # 解析为数值以计算均值，但不在此阶段改变原始字符串列，直接写回字符串格式
    rating_numeric = df[RATING_COL].map(_parse_float)

    # 按桥 + 方法计算组内均值
    group_means = (
        rating_numeric.groupby([df[STRUCTURE_COL], df[METHOD_COL]], dropna=False)
        .transform("mean")
    )

    # 标记需要填补的记录（当前缺失但组内均值存在）
    missing_mask = rating_numeric.isna()
    has_mean_mask = group_means.notna()
    impute_mask = missing_mask & has_mean_mask

    # 写回填补后的值（以字符串形式保存，方便与原始文本保持一致风格）
    df.loc[impute_mask, RATING_COL] = group_means[impute_mask].map(
        lambda x: f"{x:.6g}"
    )

    # 构建 imputed_flag 标记列
    df[IMPUTED_FLAG_COL] = 0
    df.loc[impute_mask, IMPUTED_FLAG_COL] = 1

    imputed_count = int(impute_mask.sum())
    return df, imputed_count


def write_clean_csv(script_dir: Path, df: pd.DataFrame) -> Path:
    """
    将缺失值处理完成后的数据写出为 CSV。

    输出文件：
    - `clean_stage2_inventory_rating.csv`
    - 保留所有原始字段 + rating_year + source_file 等辅助字段 + imputed_flag。
    """
    path = script_dir / STAGE2_CLEAN_FILENAME
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def write_summary_yaml(
    script_dir: Path,
    missing_before: Dict[str, Dict],
    missing_after: Dict[str, Dict],
    drop_stats: Dict[str, int],
    imputed_count: int,
) -> Path:
    """
    生成第 2 阶段缺失值处理的 summary YAML。

    内容包括：
    - 各字段缺失数量（处理前 vs 处理后）；
    - 均值填补的记录数量；
    - 被剔除记录数量（按原因分类）；
    - 每个方法（3、4）分别的缺失情况统计（处理前 vs 处理后）。
    """
    summary = {
        "fields_missing": {
            "before": missing_before["by_field"],
            "after": missing_after["by_field"],
        },
        "imputation": {
            "inventory_rating_mean_imputed_records": imputed_count,
        },
        "dropped_records": {
            "missing_structure_number": drop_stats.get("missing_structure_number", 0),
            "missing_method": drop_stats.get("missing_method", 0),
            "missing_rating_year": drop_stats.get("missing_rating_year", 0),
            "no_available_mean_for_bridge_method": drop_stats.get("no_available_mean", 0),
        },
        "per_method_missing": {
            f"method_{m}": {
                "before": missing_before["by_method"].get(m, {}),
                "after": missing_after["by_method"].get(m, {}),
            }
            for m in sorted(TARGET_METHODS)
        },
    }

    yaml_body = yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    path = script_dir / STAGE2_SUMMARY_FILENAME
    path.write_text(yaml_body, encoding="utf-8")
    return path


def main() -> None:
    """
    脚本入口：执行 Inventory Rating 数据清洗 Pipeline 的第 2 阶段（缺失值清洗）。

    处理流程：
    1. 读取 Stage 1 输出的 `clean_inventory_rating_records.csv`；
    2. 调用 analyze_missing()，统计处理前的缺失情况；
    3. 调用 drop_invalid_records()，根据规则 1 和规则 2（组内全缺失）剔除记录；
    4. 调用 impute_inventory_rating()，对 INVENTORY_RATING_066 进行组内均值填补；
    5. 再次调用 analyze_missing()，统计处理后的缺失情况；
    6. 写出 `clean_stage2_inventory_rating.csv` 和 `missing_value_summary_stage2.yaml`。
    """
    script_dir = Path(__file__).resolve().parent

    # 1. 读取 Stage 1 clean 数据
    df_stage1 = load_stage1_clean_data(script_dir)

    # 2. 处理前缺失情况统计
    missing_before = analyze_missing(df_stage1)

    # 3. 按规则剔除无法用于建模的记录
    df_valid, drop_stats = drop_invalid_records(df_stage1)

    # 4. 对 INVENTORY_RATING_066 进行组内均值填补
    df_stage2, imputed_count = impute_inventory_rating(df_valid)

    # 5. 处理后缺失情况统计
    missing_after = analyze_missing(df_stage2)

    # 6. 写出清洗后的 CSV 和 summary YAML
    write_clean_csv(script_dir, df_stage2)
    write_summary_yaml(
        script_dir,
        missing_before=missing_before,
        missing_after=missing_after,
        drop_stats=drop_stats,
        imputed_count=imputed_count,
    )


if __name__ == "__main__":
    main()

