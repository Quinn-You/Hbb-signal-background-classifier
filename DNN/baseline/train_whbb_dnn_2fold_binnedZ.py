import os
import json
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. 基础设置
# ============================================================
def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 1. 单 bin Asimov 项
#    z_i^2 = 2 * [ (s+b) ln(1+s/b) - s ]
# ============================================================
def asimov_bin_contribution(s, b, eps=1e-12):
    if s <= 0:
        return 0.0

    b_eff = max(b, eps)

    val = 2.0 * ((s + b_eff) * math.log(1.0 + s / b_eff) - s)
    return max(val, 0.0)


# ============================================================
# 2. binned Asimov significance
#    Z = sqrt( sum_i z_i^2 )
# ============================================================
def binned_asimov_significance(scores, labels, weights, n_bins=20, score_min=0.0, score_max=1.0):
    sig_scores = scores[labels == 1]
    sig_weights = weights[labels == 1]

    bkg_scores = scores[labels == 0]
    bkg_weights = weights[labels == 0]

    bins = np.linspace(score_min, score_max, n_bins + 1)

    s_hist, _ = np.histogram(sig_scores, bins=bins, weights=sig_weights)
    b_hist, _ = np.histogram(bkg_scores, bins=bins, weights=bkg_weights)

    z2_sum = 0.0
    for s_i, b_i in zip(s_hist, b_hist):
        z2_sum += asimov_bin_contribution(float(s_i), float(b_i))

    z = math.sqrt(max(z2_sum, 0.0))

    return {
        "z": z,
        "bins": bins,
        "s_hist": s_hist,
        "b_hist": b_hist,
    }


# ============================================================
# 3. PyTorch Dataset
# ============================================================
class NumpyDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.w = torch.tensor(w, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


# ============================================================
# 4. DNN 模型
# ============================================================
class SimpleDNN(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)   # logits


# ============================================================
# 5. clip + standardize
# ============================================================
def compute_clip_bounds(X_train, continuous_idx, low_q=0.001, high_q=0.999):
    clip_bounds = {}
    for idx in continuous_idx:
        low = np.quantile(X_train[:, idx], low_q)
        high = np.quantile(X_train[:, idx], high_q)
        clip_bounds[idx] = (low, high)
    return clip_bounds


def apply_clip(X, clip_bounds):
    X_out = X.copy()
    for idx, (low, high) in clip_bounds.items():
        X_out[:, idx] = np.clip(X_out[:, idx], low, high)
    return X_out


def compute_standardization(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_standardization(X, mean, std):
    return (X - mean) / std


# ============================================================
# 6. weighted BCE loss
# ============================================================
def weighted_bce_loss(logits, targets, weights):
    loss_per_event = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weighted_loss = (loss_per_event * weights).sum() / weights.sum()
    return weighted_loss


# ============================================================
# 7. 一个 epoch 的训练 / 验证
# ============================================================
def run_epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)

    total_loss_num = 0.0
    total_weight = 0.0

    all_scores = []
    all_labels = []
    all_weights = []

    for X_batch, y_batch, w_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        w_batch = w_batch.to(device)

        logits = model(X_batch)
        loss = weighted_bce_loss(logits, y_batch, w_batch)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            batch_weight = w_batch.sum().item()
            total_loss_num += loss.item() * batch_weight
            total_weight += batch_weight

            scores = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            labels = y_batch.cpu().numpy().reshape(-1)
            weights = w_batch.cpu().numpy().reshape(-1)

            all_scores.append(scores)
            all_labels.append(labels)
            all_weights.append(weights)

    epoch_loss = total_loss_num / total_weight if total_weight > 0 else 0.0
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    all_weights = np.concatenate(all_weights)

    return epoch_loss, all_scores, all_labels, all_weights


# ============================================================
# 8. 作图函数
# ============================================================
def plot_loss(history, outpath):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_auc(history, outpath):
    epochs = np.arange(1, len(history["val_auc"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_auc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_roc(labels, scores, outpath):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.5f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Background efficiency")
    plt.ylabel("Signal efficiency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_score_distribution(train_scores, train_labels, test_scores, test_labels, outpath):
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 50)

    plt.hist(train_scores[train_labels == 1], bins=bins, histtype="step", density=True, label="Train sig")
    plt.hist(train_scores[train_labels == 0], bins=bins, histtype="step", density=True, label="Train bkg")

    plt.hist(test_scores[test_labels == 1], bins=bins, histtype="stepfilled", alpha=0.3, density=True, label="Test sig")
    plt.hist(test_scores[test_labels == 0], bins=bins, histtype="stepfilled", alpha=0.3, density=True, label="Test bkg")

    plt.xlabel("DNN score")
    plt.ylabel("Arbitrary units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_binned_shapes(scores, labels, weights, outpath, n_bins=20):
    bins = np.linspace(0, 1, n_bins + 1)

    plt.figure(figsize=(7, 5))
    plt.hist(
        scores[labels == 0],
        bins=bins,
        weights=weights[labels == 0],
        histtype="stepfilled",
        alpha=0.5,
        label="Background",
    )
    plt.hist(
        scores[labels == 1],
        bins=bins,
        weights=weights[labels == 1],
        histtype="step",
        linewidth=2,
        label="Signal",
    )

    plt.xlabel("DNN score")
    plt.ylabel("Weighted yield / bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ============================================================
# 9. 一个 fold 的训练
# ============================================================
def train_one_fold(
    fold_name,
    X_train_full,
    y_train_full,
    w_train_full,
    X_test,
    y_test,
    w_test,
    event_test,
    feature_names,
    outdir,
    device="cpu",
    batch_size=16384,
    lr=1e-3,
    n_epochs=20,
    val_fraction=0.10,
    low_q=0.001,
    high_q=0.999,
    n_sig_bins=20,
):
    print("\n" + "=" * 80)
    print(f"Start training fold: {fold_name}")
    print("=" * 80)

    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------------------
    # 1. 从 training fold 中切 validation
    # --------------------------------------------------------
    n_train_full = len(X_train_full)
    indices = np.arange(n_train_full)
    np.random.shuffle(indices)

    n_val = int(n_train_full * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    w_train = w_train_full[train_idx]

    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]
    w_val = w_train_full[val_idx]

    print(f"{fold_name}: train = {len(X_train):,}, val = {len(X_val):,}, test = {len(X_test):,}")

    # --------------------------------------------------------
    # 2. 连续变量 clip
    # --------------------------------------------------------
    continuous_idx = [0, 1, 2, 3, 4, 7, 8]

    clip_bounds = compute_clip_bounds(X_train, continuous_idx, low_q=low_q, high_q=high_q)

    X_train = apply_clip(X_train, clip_bounds)
    X_val = apply_clip(X_val, clip_bounds)
    X_test_proc = apply_clip(X_test, clip_bounds)

    # --------------------------------------------------------
    # 3. 标准化
    # --------------------------------------------------------
    mean, std = compute_standardization(X_train)

    X_train = apply_standardization(X_train, mean, std)
    X_val = apply_standardization(X_val, mean, std)
    X_test_proc = apply_standardization(X_test_proc, mean, std)

    prep_info = {
        "feature_names": feature_names.tolist() if isinstance(feature_names, np.ndarray) else list(feature_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "clip_bounds": {str(k): [float(v[0]), float(v[1])] for k, v in clip_bounds.items()},
    }
    with open(os.path.join(outdir, f"{fold_name}_preprocessing.json"), "w") as f_json:
        json.dump(prep_info, f_json, indent=2)

    # --------------------------------------------------------
    # 4. DataLoader
    # --------------------------------------------------------
    train_ds = NumpyDataset(X_train, y_train, w_train)
    val_ds = NumpyDataset(X_val, y_val, w_val)
    test_ds = NumpyDataset(X_test_proc, y_test, w_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # --------------------------------------------------------
    # 5. 模型
    # --------------------------------------------------------
    model = SimpleDNN(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }

    best_metric = -999.0
    best_epoch = -1
    best_state = None

    # --------------------------------------------------------
    # 6. 训练循环
    # --------------------------------------------------------
    for epoch in range(n_epochs):
        train_loss, train_scores_tmp, train_labels_tmp, train_weights_tmp = run_epoch(
            model, train_loader, optimizer=optimizer, device=device
        )

        val_loss, val_scores, val_labels, val_weights = run_epoch(
            model, val_loader, optimizer=None, device=device
        )

        val_auc = roc_auc_score(val_labels, val_scores)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(
            f"[{fold_name}] Epoch {epoch+1:2d}/{n_epochs} | "
            f"train loss = {train_loss:.6f} | "
            f"val loss = {val_loss:.6f} | "
            f"val AUC = {val_auc:.6f}"
        )

        if val_auc > best_metric:
            best_metric = val_auc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[{fold_name}] Best epoch = {best_epoch}, best val AUC = {best_metric:.6f}")

    model.load_state_dict(best_state)

    model_path = os.path.join(outdir, f"{fold_name}_model.pt")
    torch.save(model.state_dict(), model_path)

    # --------------------------------------------------------
    # 7. 重新评估 train / test
    # --------------------------------------------------------
    X_train_full_proc = apply_clip(X_train_full, clip_bounds)
    X_train_full_proc = apply_standardization(X_train_full_proc, mean, std)

    train_full_ds = NumpyDataset(X_train_full_proc, y_train_full, w_train_full)
    train_full_loader = DataLoader(train_full_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    _, train_scores, train_labels, train_weights = run_epoch(
        model, train_full_loader, optimizer=None, device=device
    )

    _, test_scores, test_labels, test_weights = run_epoch(
        model, test_loader, optimizer=None, device=device
    )

    test_auc = roc_auc_score(test_labels, test_scores)
    print(f"[{fold_name}] Test AUC = {test_auc:.6f}")

    # 新定义：binned Asimov significance
    test_binned_Z = binned_asimov_significance(
        test_scores, test_labels, test_weights, n_bins=n_sig_bins
    )
    print(f"[{fold_name}] Binned Asimov Z ({n_sig_bins} bins) = {test_binned_Z['z']:.6f}")

    # --------------------------------------------------------
    # 8. 作图
    # --------------------------------------------------------
    plot_loss(history, os.path.join(outdir, f"{fold_name}_loss.png"))
    plot_auc(history, os.path.join(outdir, f"{fold_name}_val_auc.png"))
    plot_roc(test_labels, test_scores, os.path.join(outdir, f"{fold_name}_roc.png"))
    plot_score_distribution(
        train_scores, train_labels, test_scores, test_labels,
        os.path.join(outdir, f"{fold_name}_score_dist.png")
    )
    plot_binned_shapes(
        test_scores, test_labels, test_weights,
        os.path.join(outdir, f"{fold_name}_score_binned_shapes.png"),
        n_bins=n_sig_bins
    )

    # --------------------------------------------------------
    # 9. 返回结果
    # --------------------------------------------------------
    result = {
        "fold_name": fold_name,
        "model_path": model_path,
        "best_epoch": best_epoch,
        "best_val_auc": best_metric,
        "test_auc": test_auc,
        "binned_z": test_binned_Z["z"],
        "test_event_number": event_test,
        "test_scores": test_scores,
        "test_labels": test_labels,
        "test_weights": test_weights,
    }

    summary = {
        "fold_name": fold_name,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_metric),
        "test_auc": float(test_auc),
        "binned_asimov_z": float(test_binned_Z["z"]),
        "n_sig_bins": int(n_sig_bins),
    }
    with open(os.path.join(outdir, f"{fold_name}_summary.json"), "w") as f_json:
        json.dump(summary, f_json, indent=2)

    return result


# ============================================================
# 10. 主程序
# ============================================================
def main():
    set_seed(12345)

    input_npz = "whbb_dnn_prepared.npz"
    outdir = "whbb_dnn_training_output_binnedZ"
    os.makedirs(outdir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading prepared data from {input_npz}")
    data = np.load(input_npz, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    w = data["w"].astype(np.float32)
    event_number = data["event_number"].astype(np.int64)
    sample = data["sample"]
    feature_names = data["feature_names"]

    print(f"Loaded events: {len(X):,}")
    print(f"Signal count : {(y == 1).sum():,}")
    print(f"Bkg count    : {(y == 0).sum():,}")

    even_mask = (event_number % 2 == 0)
    odd_mask = (event_number % 2 == 1)

    # Fold A
    result_A = train_one_fold(
        fold_name="Even_train_Odd_test",
        X_train_full=X[even_mask],
        y_train_full=y[even_mask],
        w_train_full=w[even_mask],
        X_test=X[odd_mask],
        y_test=y[odd_mask],
        w_test=w[odd_mask],
        event_test=event_number[odd_mask],
        feature_names=feature_names,
        outdir=outdir,
        device=device,
        batch_size=8192,
        lr=1e-3,
        n_epochs=20,
        val_fraction=0.10,
        low_q=0.001,
        high_q=0.999,
        n_sig_bins=20,
    )

    # Fold B
    result_B = train_one_fold(
        fold_name="Odd_train_Even_test",
        X_train_full=X[odd_mask],
        y_train_full=y[odd_mask],
        w_train_full=w[odd_mask],
        X_test=X[even_mask],
        y_test=y[even_mask],
        w_test=w[even_mask],
        event_test=event_number[even_mask],
        feature_names=feature_names,
        outdir=outdir,
        device=device,
        batch_size=8192,
        lr=1e-3,
        n_epochs=20,
        val_fraction=0.10,
        low_q=0.001,
        high_q=0.999,
        n_sig_bins=20,
    )

    # --------------------------------------------------------
    # 合并 out-of-fold 结果
    # --------------------------------------------------------
    all_event_number = np.concatenate([result_A["test_event_number"], result_B["test_event_number"]])
    all_scores = np.concatenate([result_A["test_scores"], result_B["test_scores"]])
    all_labels = np.concatenate([result_A["test_labels"], result_B["test_labels"]])
    all_weights = np.concatenate([result_A["test_weights"], result_B["test_weights"]])

    order = np.argsort(all_event_number)
    all_event_number = all_event_number[order]
    all_scores = all_scores[order]
    all_labels = all_labels[order]
    all_weights = all_weights[order]

    combined_auc = roc_auc_score(all_labels, all_scores)
    combined_binned_Z = binned_asimov_significance(
        all_scores, all_labels, all_weights, n_bins=20
    )

    print("\n" + "=" * 80)
    print("Combined 2-fold result")
    print("=" * 80)
    print(f"Combined AUC                    = {combined_auc:.6f}")
    print(f"Combined binned Asimov Z (20)   = {combined_binned_Z['z']:.6f}")
    print("=" * 80)

    plot_roc(all_labels, all_scores, os.path.join(outdir, "combined_roc.png"))
    plot_binned_shapes(
        all_scores, all_labels, all_weights,
        os.path.join(outdir, "combined_score_binned_shapes.png"),
        n_bins=20
    )

    np.savez_compressed(
        os.path.join(outdir, "whbb_dnn_oof_scores.npz"),
        event_number=all_event_number,
        dnn_score=all_scores,
        label=all_labels,
        weight=all_weights,
    )

    with open(os.path.join(outdir, "combined_summary.json"), "w") as f_json:
        json.dump(
            {
                "combined_auc": float(combined_auc),
                "combined_binned_asimov_z": float(combined_binned_Z["z"]),
                "n_sig_bins": 20,
            },
            f_json,
            indent=2,
        )

    print(f"\nSaved combined out-of-fold scores to {os.path.join(outdir, 'whbb_dnn_oof_scores.npz')}")


if __name__ == "__main__":
    main()
