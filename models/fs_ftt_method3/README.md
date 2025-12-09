# method3 FT-Transformer 基线

本目录用于训练基于 FT-Transformer 的 method3（INV_RATING_METH_065=3）基线模型。
输入数据来源：`../../data_pipelines/065_value_3_4_lr_rise_106_statistics/clean_stage3_inventory_rating.csv`

## 运行流程
```bash
python prepare_fs_dataset_method3.py
python split_by_structure_number_method3.py
python train_fs_ft_transformer_method3.py
```

## 输出文件
- `fs_dataset_method3.csv`：method3 的 FS 数据集（完成筛选与简单填补）。
- `split_indices_method3.yaml`：按结构号划分的 Train/Val/Test 列表（带中文注释）。
- `fs_ftt_metrics_method3.yaml`：训练完成后的评估指标（Train/Val/Test 的 MSE/MAE/RMSE/R²）。
- `fs_ftt_predictions_method3.csv`：测试集的真实值与预测值对照表。
