# Inventory Rating Method 065 分析

该目录包含用于分析所有可用 NBI 文件中，`INV_RATING_METH_065` 等于 `3` 或 `4` 时相关记录的代码和输出结果。

## 内容

- `inv_rating_method_analysis.py` – 处理脚本，用于读取
  `../all_States_in_a_single_file_raw` 目录下的每个 `NBI_*_Delimited_AllStates.txt` 文件，并生成下述产物。
- `rating_increase_records.csv` – 包含同一桥梁在相同方法下，后期年份中 `INVENTORY_RATING_066` 高于早期年份的行。
- `reconstructed_after_first_rating_records.csv` – 包含
  `YEAR_RECONSTRUCTED_106` 大于该桥梁/方法组合首次观测年份的行。
- `summary.yaml` – 以 YAML 形式存储的聚合统计结果，给出两类分析中受影响记录和结构的数量，并按方法进行划分，便于后续 pipeline 使用。

## 运行分析

```bash
python inv_rating_method_analysis.py
```

该脚本是幂等的；再次运行时会基于当前源数据刷新生成的 CSV 和 YAML 输出。
