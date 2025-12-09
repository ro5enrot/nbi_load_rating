# Inventory Rating Method 065 清洗与统计 Pipeline

该目录实现了一个面向 `INV_RATING_METH_065 ∈ {3,4}` 的三阶段清洗流程，
对所有 `all_States_in_a_single_file_raw/NBI_*_Delimited_AllStates.txt` 文件进行处理，
并在每个阶段输出可复现的 CSV / YAML 结果，供后续建模与审计使用。

## 目录结构

- `inv_rating_logical_anomaly_cleaning.py`：Stage 1，处理时间逻辑异常；
- `inv_rating_missing_value_cleaning.py`：Stage 2，删除标签缺失记录；
- `inv_rating_outlier_cleaning.py`：Stage 3，数值异常检测；
- `rating_increase_records.csv` / `reconstructed_after_first_rating_records.csv`：
  Stage 1 导出的两类异常详情；
- `logical_anomaly_summary_stage1.yaml`：Stage 1 统计结果；
- `missing_value_records_stage2.csv` / `missing_value_summary_stage2.yaml`：Stage 2 输出；
- `outlier_records_stage3.csv` / `outlier_summary_stage3.yaml`：Stage 3 输出；
- `clean_stage[1-3]_inventory_rating.csv`：各阶段的 clean 样本，作为下一阶段输入。

## Pipeline 阶段

### Stage 1 – 逻辑异常清洗
1. 读取所有原始文本，提取方法 3/4 记录并附加 `rating_year` 与来源文件。
2. 识别相邻年份 `INVENTORY_RATING_066` 反常升高以及
   `YEAR_RECONSTRUCTED_106` 晚于首次承载力年份的记录。
3. 输出：
   - `rating_increase_records.csv`
   - `reconstructed_after_first_rating_records.csv`
   - `clean_stage1_inventory_rating.csv`
   - `logical_anomaly_summary_stage1.yaml`

### Stage 2 – 标签缺失清洗
1. 读取 Stage 1 clean 数据。
2. 删除 `INVENTORY_RATING_066` 缺失或不可解析的记录。
3. 输出：
   - `missing_value_records_stage2.csv`
   - `clean_stage2_inventory_rating.csv`
   - `missing_value_summary_stage2.yaml`

### Stage 3 – 数值异常清洗
1. 读取 Stage 2 clean 数据并解析 `INVENTORY_RATING_066` 为数值。
2. 应用 IQR 统计规则 + 工程经验规则（<=0、>200、0~1）识别异常。
3. 需要剔除的异常会被移出 clean 数据，其余记录保留。
4. 输出：
   - `outlier_records_stage3.csv`（附 `outlier_type`）
   - `clean_stage3_inventory_rating.csv`
   - `outlier_summary_stage3.yaml`

## 运行方式

依次运行三个脚本即可复现全部产物；每个脚本都可重复执行，会根据最新源数据刷新对应输出。

```bash
python inv_rating_logical_anomaly_cleaning.py
python inv_rating_missing_value_cleaning.py
python inv_rating_outlier_cleaning.py
```

所有脚本对 pandas / numpy / pyyaml 有依赖，运行前请确保虚拟环境安装了这些包。
