# Project Overview

## 整体目标与技术路线
- **项目目标**：围绕美国国家桥梁清单（NBI）中 Inventory Rating（字段 `INVENTORY_RATING_066`）的方法 3/4 数据，建立一条可复现的数据清洗 → 特征筛选 → 模型训练流水线，用于评估桥梁承载力（Load Rating）并导出统计与模型结果。
- **技术路线**：  
  1. 以 `all_States_in_a_single_file_raw/NBI_*_Delimited_AllStates.txt` 为原始输入，使用 `pandas` 按阶段执行逻辑异常、缺失值、数值异常清洗，阶段间以 CSV/YAML 结果衔接。  
  2. 基于 Stage 3 的干净样本，按方法 3/4 分别构建 FS（feature selection）数据集，做基础填补与类别规范化。  
  3. 利用 PyTorch 和自定义的简化版 FT-Transformer 架构（数值线性投射 + 类别 embedding + TransformerEncoder）在 GPU 上训练回归模型，按结构号划分 Train/Val/Test，输出指标与预测值。  
  4. 全流程通过结构化 YAML 汇总与注释，保证审计与 AI 进一步代码生成时的可追溯性。

## 目录结构
- `README.md`：说明数据处理、建模、结果、可视化等目录的职责与存在意义。
- `project_overview.md`：本文件，供 AI 了解项目结构与数据流。
- `all_States_in_a_single_file_raw/`：存放原始的 NBI 文本样本（逐州汇总）。清洗脚本使用 `Path.glob("NBI_*_Delimited_AllStates.txt")` 批量读取。
- `data_pipelines/065_value_3_4_lr_rise_106_statistics/`：三阶段 Inventory Rating 清洗脚本、对应 YAML 汇总与 CSV 产物。
- `models/fs_ftt_method3/`：方法 3（`INV_RATING_METH_065=3`）的数据准备、结构号划分、FT-Transformer 训练及指标输出。
- `models/fs_ftt_method4/`：方法 4（`INV_RATING_METH_065=4`）与 method3 相同的建模流程。
- 其余 README 中提到的 `results/`, `visualizations/`, `notebooks/`, `docs/` 目前未在仓库根目录出现，后续可按需补充。

## 数据清洗 Pipeline（`data_pipelines/065_value_3_4_lr_rise_106_statistics/`）

### `inv_rating_logical_anomaly_cleaning.py`
- **作用**：Stage 1，加载所有原始文本，附加 `rating_year` 与 `source_file`，仅保留方法 3/4，检测两类时间逻辑异常：相邻年份承载力反常升高、重建年份晚于首次承载力年份。
- **核心逻辑**：  
  - `_parse_float/_parse_int` 统一处理字符串数值；`detect_rating_increase_anomalies` 与 `detect_reconstruction_after_first_rating_anomalies` 对 `(STRUCTURE_NUMBER_008, INV_RATING_METH_065)` 组内进行时间序列扫描。  
  - `build_clean_dataset` 通过 `_row_id` 剔除异常记录，形成干净基础样本。  
  - `write_summary_yaml` 生成 `logical_anomaly_summary_stage1.yaml`，含方法级别的记录数/桥梁数统计。
- **输入输出**：读取 `../all_States_in_a_single_file_raw/`；输出 `rating_increase_records.csv`, `reconstructed_after_first_rating_records.csv`, `clean_stage1_inventory_rating.csv`, `logical_anomaly_summary_stage1.yaml`。

### `inv_rating_missing_value_cleaning.py`
- **作用**：Stage 2，在 Stage 1 clean 基础上删除 `INVENTORY_RATING_066` 缺失/异常值记录，仅做“标签合法性”过滤。
- **关键函数**：  
  - `analyze_missing_stats` & `_is_missing_numeric_like` 统计缺失情况（NaN、空白、`N`、`N/A`、`-`）。  
  - `drop_missing_label_records` 分离 clean 与 dropped 样本。  
  - `write_summary_yaml` 输出含总体/分方法数量的 `missing_value_summary_stage2.yaml`。
- **输出**：`clean_stage2_inventory_rating.csv`, `missing_value_records_stage2.csv`, `missing_value_summary_stage2.yaml`。Stage2 汇总显示前后样本数 397332→397313，标签缺失 19 行被移除（method3:18, method4:1）。

### `inv_rating_outlier_cleaning.py`
- **作用**：Stage 3，对 Stage 2 clean 数据执行数值异常检测并剔除。  
- **步骤**：  
  - `parse_numeric_rating` 将标签转浮点；`detect_iqr_outliers` 用全局 IQR（Q1=32.4, Q3=51.7, 阈值 3.45~80.65）检测异常；`detect_engineering_rule_outliers` 基于工程经验（≤0、>200、0~1 记录）。  
  - `combine_outliers` 拼接 `outlier_type`，对需剔除的类型生成 `clean_stage3_inventory_rating.csv`，并聚合统计（Stage3 YAML 显示共剔除 28,772 条，其中 method3 占 28,096）。
- **输出**：`outlier_records_stage3.csv`, `clean_stage3_inventory_rating.csv`, `outlier_summary_stage3.yaml`（包含 IQR 阈值、类型计数、方法分布）。

### CSV/YAML 资产摘要
- `clean_stage[1-3]_inventory_rating.csv`：Stage 1/2/3 的整表快照，保留全部 NBI 字段及新增 `rating_year`,`source_file` 等辅助列。
- `rating_increase_records.csv`, `reconstructed_after_first_rating_records.csv`：Stage1 检测出的两类异常详细记录，列结构与 clean CSV 相同。
- `missing_value_records_stage2.csv`：被删除的标签缺失行；字段与 clean CSV 一致。
- `outlier_records_stage3.csv`：Stage3 被标记的异常行，额外带 `outlier_type`。
- `logical_anomaly_summary_stage1.yaml`, `missing_value_summary_stage2.yaml`, `outlier_summary_stage3.yaml`：分别描述各阶段异常计数、阈值及方法级别统计，均以中文注释说明字段定义。

## 模型与特征工程

### Method3 流程（`models/fs_ftt_method3/`）
- `config_fs_ftt_method3.yaml`：路径（输入 Stage3 clean 数据 → FS 数据集 → 划分 → 指标 → 预测）、列定义（6 数值 + 8 类别 + `STRUCTURE_NUMBER_008` ID + `INVENTORY_RATING_066` 目标）、划分比例 (0.7/0.15/0.15)、FT-Transformer 超参（`d_model=128, n_heads=4, n_layers=2, dropout=0.1`）与训练参数（batch 64、AdamW、max 200 epochs、patience 20）。
- `prepare_fs_dataset_method3.py`：  
  - 读取 Stage3 clean CSV，仅保留 `INV_RATING_METH_065=3`。  
  - 数值列用中位数填补，类别列通过 `sanitize_categorical` 归一化空值 → `"Unknown"`。  
  - 去掉方法列，输出 `fs_dataset_method3.csv`（列：结构号、目标、6 数值、8 类别）。
- `split_by_structure_number_method3.py`：  
  - 对 FS 数据集中唯一 `STRUCTURE_NUMBER_008` 随机打乱（`random_state=42`），按比例分配 Train/Val/Test，保证结构号不跨集合。  
  - 结果写入 `split_indices_method3.yaml`（附中文注释且列出每个集合的结构号列表）。
- `train_fs_ft_transformer_method3.py`：  
  - 依赖 CUDA，运行时直接检查 `torch.cuda.is_available()`。  
  - `FTDataset` 在 `__getitem__` 中对数值特征做标准化，对类别特征用字典映射（0 号未知）。  
  - `FTTransformer` 针对每个数值列建单独线性层，将类别列嵌入后拼接，通过 `nn.TransformerEncoder` → `LayerNorm + Linear` 输出回归值。  
  - 训练过程中记录首个 batch 的 device 诊断、使用早停逻辑保存最佳模型，训练完毕后在 Train/Val/Test 上评估 MSE/MAE/RMSE/R²。  
  - 指标写到 `fs_ftt_metrics_method3.yaml`（示例：test RMSE ≈ 11.16，R² ≈ 0.36），预测写到 `fs_ftt_predictions_method3.csv`（列：结构号、`y_true`,`y_pred`）。

### Method4 流程（`models/fs_ftt_method4/`）
- `config_fs_ftt_method4.yaml` 内容与 method3 基本一致，仅区分输出文件名。  
- `prepare_fs_dataset_method4.py`, `split_by_structure_number_method4.py`, `train_fs_ft_transformer_method4.py` 逻辑与 method3 对称，仅保留方法 4 样本并生成对应产物（`fs_dataset_method4.csv`, `split_indices_method4.yaml`, `fs_ftt_metrics_method4.yaml`, `fs_ftt_predictions_method4.csv`）。  
- Method4 当前训练结果：`fs_ftt_metrics_method4.yaml` 显示 test RMSE ≈ 8.00、R² ≈ 0.49（性能优于 method3，反映方法 4 标签分布可能更稳定）。

## 配置、统计与依赖关系
- **YAML 配置/统计**：  
  - Pipeline 各阶段的 `*_summary_stage*.yaml` 记录异常计数与阈值，供监控和审计。  
  - 模型配置文件集中定义列、路径与超参，使脚本解耦并便于 AI 复用。  
  - `split_indices_method*.yaml` 的结构是 `train/val/test` → 结构号列表，可直接被后续模型或推理脚本读取，确保无桥梁泄漏。
- **CSV 产物关系**：  
  - `clean_stage1_inventory_rating.csv` → Stage2 过滤 → `clean_stage2_inventory_rating.csv` → Stage3 异常剔除 → `clean_stage3_inventory_rating.csv`。  
  - `clean_stage3_inventory_rating.csv` + `config_fs_ftt_method*.yaml` → `fs_dataset_method*.csv` → `split_indices_method*.yaml` → `train_fs_ft_transformer_method*.py` → `fs_ftt_metrics_method*.yaml` & `fs_ftt_predictions_method*.csv`。  
  - 所有 CSV 头部包含完整 NBI 字段，便于 AI 脚本识别字段含义（见“CSV 资产摘要”段落）。
- **依赖**：主要使用 `pandas`, `numpy`, `pyyaml` 处理数据，建模阶段依赖 `torch`（需 GPU），并使用 `scikit-learn` 以外的自定义 Transformer 模型；命令行脚本均以 `python script.py` 形式运行。

## 模块依赖关系与数据流
1. **原始数据导入**：`inv_rating_logical_anomaly_cleaning.py` 读取 `all_States_in_a_single_file_raw/` 下所有 `NBI_*` 文本，通过 `Path.glob` 批量解析并聚合，附加 `rating_year` 与 `source_file` 元数据。
2. **逻辑异常剔除**：Stage1 输出 clean + 异常记录 CSV，并生成 YAML 汇总，以结构号和方法维度描述异常规模。
3. **缺失/数值异常处理**：Stage2/3 脚本串行运行，逐步剔除标签缺失与 outlier，保证 `clean_stage3_inventory_rating.csv` 成为唯一可信输入。  
4. **特征筛选与数据集构建**：`prepare_fs_dataset_method*.py` 读取 Stage3 clean 数据，按方法过滤并保留配置中的特征列，完成最小填补与类别归一化。  
5. **数据集划分**：`split_by_structure_number_method*.py` 保证结构号唯一地落入 Train/Val/Test，避免同一桥梁跨集合所致的信息泄漏；输出 YAML 供多模型复用。  
6. **模型训练与评估**：`train_fs_ft_transformer_method*.py` 读取 FS 数据与 split YAML，构建张量数据集与 FT-Transformer 模型，在 GPU 上训练并通过早停挑选最优模型；最后写出指标 YAML 与测试集预测 CSV。

## 关键技术与设计思想
- **分阶段数据治理**：每个 Python 脚本聚焦单一职责（逻辑异常、缺失值、数值异常），并通过命名规范的 CSV/YAML 交付物进行串联，符合可复现数据管道设计。
- **可审计性**：所有清洗阶段都导出“异常明细 CSV + 汇总 YAML”，并在 YAML 顶部添加中文注释，说明指标的业务含义，方便 AI 或审阅者理解。
- **结构号划分以防泄漏**：训练/验证/测试划分在结构号层面完成，配合 `split_indices_method*.yaml` 保证任何后续模型都可以复用相同切分，避免桥梁级信息泄漏。
- **简化 FT-Transformer 实现**：自定义 `FTTransformer` 将每个数值特征视作单独 token，经线性层映射到 `d_model` 维；类别特征使用 embedding，再通过 `nn.TransformerEncoder` 建模交互，整体结构轻量且完全可控。
- **GPU 前置校验**：训练脚本一开始即检查 `torch.cuda.is_available()`，在缺少 GPU 的环境中主动失败，避免 AI 在不具备资源的情况下浪费时间。
- **配置驱动**：`config_fs_ftt_method*.yaml` 将路径、列、模型超参集中管理，使 AI 可以轻松调整特征或模型策略，而无需修改多个脚本。

## 对 AI 的使用提示
- 若需扩展清洗逻辑，可在 `data_pipelines/...` 中新增 Stage 4+，但需保持 CSV/YAML 命名一致并记录在 README 中，方便下游脚本查找最新 clean 数据。
- 在生成新模型脚本时，应优先读取现有 `config_fs_ftt_method*.yaml` 与 `split_indices_method*.yaml`，保持特征口径与划分一致。
- 所有 CSV 均为 UTF-8 编码，列名与 NBI 字段一致，可直接用 `pandas.read_csv` 加载并通过列名定位字段。若需采样，请注意结构号是字符串，包含前导空格的少数样本需要 `.str.strip()` 处理。
