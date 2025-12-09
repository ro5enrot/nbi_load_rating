#!/usr/bin/env python3
"""生成 method3（INV_RATING_METH_065=3）的 FS 数据集：筛选字段并执行最少缺失处理。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def load_config() -> Dict:
    """读取配置文件，获取路径与列定义。"""
    config_path = Path(__file__).resolve().parent / "config_fs_ftt_method3.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_categorical(series: pd.Series) -> pd.Series:
    """将类别列中的空白或异常值统一替换为 'Unknown'。"""
    text = series.astype(str).str.strip()
    mask = series.isna() | text.eq("") | text.str.upper().isin({"N", "N/A", "-"})
    result = series.astype(str)
    result[mask] = "Unknown"
    return result


def main() -> None:
    """读取 Stage3 clean CSV，仅保留 method3 样本并写出 FS 数据集。"""
    config = load_config()
    script_dir = Path(__file__).resolve().parent
    paths = config["paths"]
    cols = config["columns"]

    input_csv = (script_dir / paths["input_clean_csv"]).resolve()
    output_csv = script_dir / paths["fs_dataset_csv"]

    required_cols: List[str] = [cols["id_col"], cols["target_col"], "INV_RATING_METH_065"]
    required_cols += cols["numerical_cols"] + cols["categorical_cols"]

    df = pd.read_csv(input_csv, dtype=str, encoding="utf-8")
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集中缺少必要字段：{', '.join(missing_cols)}")

    df = df[required_cols].copy()
    df["INV_RATING_METH_065"] = pd.to_numeric(df["INV_RATING_METH_065"], errors="coerce")
    # 只保留 INV_RATING_METH_065 == 3 的记录（method 3）
    df = df[df["INV_RATING_METH_065"] == 3]
    if df.empty:
        raise ValueError("method3 样本为空，请确认输入数据。")

    target_col = cols["target_col"]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    for col in cols["numerical_cols"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in cols["categorical_cols"]:
        df[col] = sanitize_categorical(df[col])

    df = df.drop(columns=["INV_RATING_METH_065"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"method3 FS 数据集已生成：{output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"生成 method3 FS 数据集失败：{exc}", file=sys.stderr)
        raise
