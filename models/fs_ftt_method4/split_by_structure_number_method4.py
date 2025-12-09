#!/usr/bin/env python3
"""method4：按结构号划分 Train/Val/Test，确保不同集合之间无桥梁重叠。"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml


def load_config() -> Dict:
    """读取配置文件并返回字典。"""
    config_path = Path(__file__).resolve().parent / "config_fs_ftt_method4.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """主流程：读取 FS 数据集、按结构号打乱并划分 Train/Val/Test。"""
    config = load_config()
    script_dir = Path(__file__).resolve().parent
    paths = config["paths"]
    cols = config["columns"]
    split_cfg = config["split"]

    dataset_path = script_dir / paths["fs_dataset_csv"]
    output_yaml = script_dir / paths["split_indices_yaml"]

    df = pd.read_csv(dataset_path, dtype=str, encoding="utf-8")
    id_col = cols["id_col"]
    if id_col not in df.columns:
        raise ValueError(f"FS 数据集中缺少结构号列：{id_col}")

    unique_ids = df[id_col].astype(str).unique().tolist()
    rng = random.Random(split_cfg["random_state"])
    rng.shuffle(unique_ids)

    total = len(unique_ids)
    n_train = int(total * split_cfg["train_ratio"])
    n_val = int(total * split_cfg["val_ratio"])
    n_test = total - n_train - n_val

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train : n_train + n_val]
    test_ids = unique_ids[n_train + n_val :]

    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    yaml_body = yaml.safe_dump(splits, sort_keys=False, allow_unicode=True)
    comment = "# method4：按结构号划分数据，避免桥梁跨集合导致的信息泄漏。\n"
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as f:
        f.write(comment + yaml_body)

    print(f"method4 结构号总数：{total}")
    print(f"训练集结构号：{len(train_ids)}")
    print(f"验证集结构号：{len(val_ids)}")
    print(f"测试集结构号：{len(test_ids)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"method4 结构号划分失败：{exc}", file=sys.stderr)
        raise
