#!/usr/bin/env python3
"""method4：训练 FT-Transformer 基线模型并输出指标与预测结果。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用：当前环境无法使用 GPU，请停止脚本。")

DEVICE = torch.device("cuda")
print(">>> Using device:", DEVICE)
print(">>> CUDA version:", torch.version.cuda)
print(">>> GPU:", torch.cuda.get_device_name(0))

METHOD_NAME = "method4"


def load_config() -> Dict:
    """读取配置文件。"""
    config_path = Path(__file__).resolve().parent / "config_fs_ftt_method4.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FTDataset(Dataset):
    """将 DataFrame 转为可供 FT-Transformer 使用的张量。"""

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
        id_col: str,
        target_col: str,
        numeric_stats: Dict[str, Dict[str, float]],
        categorical_mappings: Dict[str, Dict[str, int]],
    ) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.id_col = id_col
        self.target_col = target_col
        self.numeric_stats = numeric_stats
        self.categorical_mappings = categorical_mappings

        self.ids = df[id_col].astype(str).tolist()
        self.targets = df[target_col].astype(float).to_numpy(dtype=np.float32)
        self.numeric_values = df[numeric_cols].astype(float).to_numpy(dtype=np.float32)
        self.categorical_values = df[categorical_cols].astype(str).to_numpy()
        self.numeric_means = np.array([numeric_stats[col]["mean"] for col in numeric_cols], dtype=np.float32)
        self.numeric_stds = np.array([numeric_stats[col]["std"] for col in numeric_cols], dtype=np.float32)
        self.numeric_stds[self.numeric_stds == 0] = 1.0

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        numeric = (self.numeric_values[idx] - self.numeric_means) / self.numeric_stds
        cat_indices = []
        for col_idx, col in enumerate(self.categorical_cols):
            value = self.categorical_values[idx, col_idx]
            mapping = self.categorical_mappings[col]
            cat_indices.append(mapping.get(value, 0))
        sample = {
            "numeric": torch.tensor(numeric, dtype=torch.float32),
            "categorical": torch.tensor(cat_indices, dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "id": self.ids[idx],
        }
        return sample


class FTTransformer(nn.Module):
    """简化版 FT-Transformer：数值线性投射 + 类别 embedding + TransformerEncoder。"""

    def __init__(
        self,
        num_numeric: int,
        num_categorical: int,
        categorical_cardinalities: List[int],
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical

        self.numeric_linears = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_numeric)])
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in categorical_cardinalities]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for i in range(self.num_numeric):
            value = numeric[:, i].unsqueeze(-1)
            tokens.append(self.numeric_linears[i](value))
        for i in range(self.num_categorical):
            tokens.append(self.categorical_embeddings[i](categorical[:, i]))
        x = torch.stack(tokens, dim=1)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


def build_numeric_stats(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """计算数值特征的均值/标准差（若标准差为 0 则置为 1）。"""
    stats = {}
    for col in numeric_cols:
        values = df[col].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def build_categorical_mappings(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """为类别特征生成词典，0 号作为未知类别。"""
    mappings = {}
    for col in categorical_cols:
        unique_vals = sorted(df[col].astype(str).unique().tolist())
        mappings[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    return mappings


def split_dataframe(
    df: pd.DataFrame,
    id_col: str,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """根据结构号划分数据行。"""
    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[id_col].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[id_col].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float, List[str], np.ndarray, np.ndarray]:
    """计算 MSE/MAE/RMSE/R²，并返回预测结果与结构号。"""
    criterion = nn.MSELoss()
    model.eval()
    losses: List[float] = []
    preds: List[float] = []
    trues: List[float] = []
    ids: List[str] = []
    with torch.no_grad():
        for batch in dataloader:
            num_x = batch["numeric"].to(device)
            cat_x = batch["categorical"].to(device)
            y = batch["target"].to(device)
            outputs = model(num_x, cat_x)
            loss = criterion(outputs, y)
            losses.append(loss.item())
            preds.extend(outputs.cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())
            ids.extend(batch["id"])

    preds_np = np.array(preds)
    trues_np = np.array(trues)
    mae = float(np.mean(np.abs(preds_np - trues_np)))
    rmse = float(np.sqrt(np.mean((preds_np - trues_np) ** 2)))
    if np.var(trues_np) == 0:
        r2 = 0.0
    else:
        ss_res = float(np.sum((trues_np - preds_np) ** 2))
        ss_tot = float(np.sum((trues_np - np.mean(trues_np)) ** 2))
        r2 = 1 - ss_res / ss_tot
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, mae, rmse, r2, ids, preds_np, trues_np


def main() -> None:
    """执行 method4 FT-Transformer 训练流程。"""
    config = load_config()
    script_dir = Path(__file__).resolve().parent
    paths = config["paths"]
    cols = config["columns"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    dataset_path = script_dir / paths["fs_dataset_csv"]
    split_path = script_dir / paths["split_indices_yaml"]
    metrics_path = script_dir / paths["train_metrics_yaml"]
    pred_path = script_dir / paths["predictions_csv"]

    df = pd.read_csv(dataset_path, dtype=str, encoding="utf-8")
    with split_path.open("r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)

    id_col = cols["id_col"]
    target_col = cols["target_col"]
    numeric_cols = cols["numerical_cols"]
    categorical_cols = cols["categorical_cols"]

    train_df, val_df, test_df = split_dataframe(
        df,
        id_col=id_col,
        train_ids=splits["train"],
        val_ids=splits["val"],
        test_ids=splits["test"],
    )
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("存在空划分，请检查结构号划分文件。")

    numeric_stats = build_numeric_stats(train_df, numeric_cols)
    cat_mappings = build_categorical_mappings(train_df, categorical_cols)
    cat_cardinalities = [len(mapping) + 1 for mapping in cat_mappings.values()]

    def build_dataset(frame: pd.DataFrame) -> FTDataset:
        return FTDataset(
            frame,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            id_col=id_col,
            target_col=target_col,
            numeric_stats=numeric_stats,
            categorical_mappings=cat_mappings,
        )

    train_dataset = build_dataset(train_df)
    val_dataset = build_dataset(val_df)
    test_dataset = build_dataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])

    model = FTTransformer(
        num_numeric=len(numeric_cols),
        num_categorical=len(categorical_cols),
        categorical_cardinalities=cat_cardinalities,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        model.train()
        epoch_losses: List[float] = []
        for batch_idx, batch in enumerate(train_loader):
            num_x = batch["numeric"].to(DEVICE)
            cat_x = batch["categorical"].to(DEVICE)
            y = batch["target"].to(DEVICE)

            if batch_idx == 0:
                print(">>> batch X device:", num_x.device)
                print(">>> batch y device:", y.device)
                for name, param in model.named_parameters():
                    print(">>> model param device:", name, param.device)
                    break

            optimizer.zero_grad()
            outputs = model(num_x, cat_x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss, val_mae, val_rmse, _, _, _, _ = evaluate_model(model, val_loader, DEVICE)
        print(f"{METHOD_NAME} Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                print("Early stopping 触发，结束训练。")
                break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    train_metrics = evaluate_model(model, train_loader, DEVICE)
    val_metrics = evaluate_model(model, val_loader, DEVICE)
    test_metrics = evaluate_model(model, test_loader, DEVICE)

    metrics_dict = {
        "train": {"loss": train_metrics[0], "mae": train_metrics[1], "rmse": train_metrics[2], "r2": train_metrics[3]},
        "val": {"loss": val_metrics[0], "mae": val_metrics[1], "rmse": val_metrics[2], "r2": val_metrics[3]},
        "test": {"loss": test_metrics[0], "mae": test_metrics[1], "rmse": test_metrics[2], "r2": test_metrics[3]},
    }

    yaml_body = yaml.safe_dump(metrics_dict, sort_keys=False, allow_unicode=True)
    comment = "# method4：FT-Transformer 训练集/验证集/测试集的 MSE/MAE/RMSE/R² 评估指标。\n"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(comment + yaml_body)

    _, _, _, _, test_ids, test_preds, test_trues = test_metrics
    pred_df = pd.DataFrame({id_col: test_ids, "y_true": test_trues, "y_pred": test_preds})
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"{METHOD_NAME} 训练完成，指标输出：{metrics_path}，预测输出：{pred_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"{METHOD_NAME} 训练失败：{exc}", file=sys.stderr)
        raise
