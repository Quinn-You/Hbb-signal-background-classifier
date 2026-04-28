import argparse
import json
import math
import os
import random
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def asimov_bin_contribution(s, b, eps=1e-12):
    if s <= 0:
        return 0.0

    b_eff = max(b, eps)
    value = 2.0 * ((s + b_eff) * math.log(1.0 + s / b_eff) - s)
    return max(value, 0.0)


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

    return math.sqrt(max(z2_sum, 0.0))


class NumpyDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.w = torch.tensor(w, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


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
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


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


def weighted_bce_loss(logits, targets, weights):
    loss_per_event = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return (loss_per_event * weights).sum() / weights.sum()


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

            all_scores.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            all_labels.append(y_batch.cpu().numpy().reshape(-1))
            all_weights.append(w_batch.cpu().numpy().reshape(-1))

    epoch_loss = total_loss_num / total_weight if total_weight > 0 else 0.0
    return (
        epoch_loss,
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_weights),
    )


def make_dataloaders(
    X_train_full,
    y_train_full,
    w_train_full,
    X_test,
    y_test,
    w_test,
    batch_size,
    val_fraction,
    low_q,
    high_q,
):
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

    continuous_idx = [0, 1, 2, 3, 4, 7, 8]
    clip_bounds = compute_clip_bounds(X_train, continuous_idx, low_q=low_q, high_q=high_q)

    X_train = apply_clip(X_train, clip_bounds)
    X_val = apply_clip(X_val, clip_bounds)
    X_test_proc = apply_clip(X_test, clip_bounds)
    X_train_full_proc = apply_clip(X_train_full, clip_bounds)

    mean, std = compute_standardization(X_train)

    X_train = apply_standardization(X_train, mean, std)
    X_val = apply_standardization(X_val, mean, std)
    X_test_proc = apply_standardization(X_test_proc, mean, std)
    X_train_full_proc = apply_standardization(X_train_full_proc, mean, std)

    train_loader = DataLoader(
        NumpyDataset(X_train, y_train, w_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        NumpyDataset(X_val, y_val, w_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        NumpyDataset(X_test_proc, y_test, w_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    train_full_loader = DataLoader(
        NumpyDataset(X_train_full_proc, y_train_full, w_train_full),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, train_full_loader, X_train.shape[1]


def train_one_fold(
    fold_name,
    X_train_full,
    y_train_full,
    w_train_full,
    X_test,
    y_test,
    w_test,
    event_test,
    device,
    batch_size,
    lr,
    n_epochs,
    val_fraction,
    low_q,
    high_q,
    n_sig_bins,
    early_stop_patience,
    early_stop_min_delta,
):
    train_loader, val_loader, test_loader, train_full_loader, input_dim = make_dataloaders(
        X_train_full,
        y_train_full,
        w_train_full,
        X_test,
        y_test,
        w_test,
        batch_size,
        val_fraction,
        low_q,
        high_q,
    )

    model = SimpleDNN(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = -np.inf
    best_epoch = -1
    best_state = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }
    epochs_without_improve = 0
    early_stopped = False

    for epoch in range(n_epochs):
        train_loss, _, _, _ = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss, val_scores, val_labels, _ = run_epoch(model, val_loader, optimizer=None, device=device)
        val_auc = roc_auc_score(val_labels, val_scores)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(
            f"[{fold_name}] epoch {epoch + 1:2d}/{n_epochs} | "
            f"train loss = {train_loss:.6f} | val loss = {val_loss:.6f} | val AUC = {val_auc:.6f}"
        )

        if val_auc > (best_metric + early_stop_min_delta):
            best_metric = val_auc
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            early_stopped = True
            print(
                f"[{fold_name}] Early stopping at epoch {epoch + 1} "
                f"(best val AUC = {best_metric:.6f} at epoch {best_epoch})"
            )
            break

    model.load_state_dict(best_state)

    _, train_scores, train_labels, train_weights = run_epoch(
        model, train_full_loader, optimizer=None, device=device
    )
    _, test_scores, test_labels, test_weights = run_epoch(
        model, test_loader, optimizer=None, device=device
    )

    return {
        "fold_name": fold_name,
        "epochs_ran": len(history["val_auc"]),
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_metric),
        "train_scores": train_scores,
        "train_labels": train_labels,
        "train_weights": train_weights,
        "test_event_number": event_test,
        "test_scores": test_scores,
        "test_labels": test_labels,
        "test_weights": test_weights,
        "test_auc": float(roc_auc_score(test_labels, test_scores)),
        "binned_z": float(
            binned_asimov_significance(test_scores, test_labels, test_weights, n_bins=n_sig_bins)
        ),
        "history": history,
    }


def evaluate_trial(
    X,
    y,
    w,
    event_number,
    device,
    batch_size,
    lr,
    n_epochs,
    val_fraction,
    low_q,
    high_q,
    n_sig_bins,
    early_stop_patience,
    early_stop_min_delta,
):
    even_mask = event_number % 2 == 0
    odd_mask = ~even_mask

    print("\n" + "=" * 80)
    print(f"Trial start: lr={lr:.6g}, batch_size={batch_size}")
    print("=" * 80)

    result_A = train_one_fold(
        fold_name="Even_train_Odd_test",
        X_train_full=X[even_mask],
        y_train_full=y[even_mask],
        w_train_full=w[even_mask],
        X_test=X[odd_mask],
        y_test=y[odd_mask],
        w_test=w[odd_mask],
        event_test=event_number[odd_mask],
        device=device,
        batch_size=batch_size,
        lr=lr,
        n_epochs=n_epochs,
        val_fraction=val_fraction,
        low_q=low_q,
        high_q=high_q,
        n_sig_bins=n_sig_bins,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )

    result_B = train_one_fold(
        fold_name="Odd_train_Even_test",
        X_train_full=X[odd_mask],
        y_train_full=y[odd_mask],
        w_train_full=w[odd_mask],
        X_test=X[even_mask],
        y_test=y[even_mask],
        w_test=w[even_mask],
        event_test=event_number[even_mask],
        device=device,
        batch_size=batch_size,
        lr=lr,
        n_epochs=n_epochs,
        val_fraction=val_fraction,
        low_q=low_q,
        high_q=high_q,
        n_sig_bins=n_sig_bins,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )

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
    combined_binned_z = binned_asimov_significance(
        all_scores, all_labels, all_weights, n_bins=n_sig_bins
    )

    summary = {
        "lr": float(lr),
        "batch_size": int(batch_size),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "combined_auc": float(combined_auc),
        "combined_binned_asimov_z": float(combined_binned_z),
        "n_sig_bins": int(n_sig_bins),
        "folds": [
            {
                "fold_name": result_A["fold_name"],
                "epochs_ran": int(result_A["epochs_ran"]),
                "early_stopped": bool(result_A["early_stopped"]),
                "best_epoch": int(result_A["best_epoch"]),
                "best_val_auc": float(result_A["best_val_auc"]),
                "test_auc": float(result_A["test_auc"]),
                "binned_asimov_z": float(result_A["binned_z"]),
            },
            {
                "fold_name": result_B["fold_name"],
                "epochs_ran": int(result_B["epochs_ran"]),
                "early_stopped": bool(result_B["early_stopped"]),
                "best_epoch": int(result_B["best_epoch"]),
                "best_val_auc": float(result_B["best_val_auc"]),
                "test_auc": float(result_B["test_auc"]),
                "binned_asimov_z": float(result_B["binned_z"]),
            },
        ],
    }

    print(f"Trial result: combined AUC = {combined_auc:.6f}, combined binned Asimov Z = {combined_binned_z:.6f}")
    return summary


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def format_trial_name(lr, batch_size):
    lr_tag = f"{lr:.0e}" if lr < 1e-3 else f"{lr:.6f}".rstrip("0").rstrip(".")
    lr_tag = lr_tag.replace("+", "")
    return f"lr_{lr_tag}_bs_{batch_size}"


def resolve_device(device_arg):
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Grid search lr and batch_size for WHbb DNN")
    parser.add_argument(
        "--input-npz",
        default=os.path.join(os.path.dirname(__file__), "..", "baseline", "whbb_dnn_prepared.npz"),
        help="Path to prepared NPZ file",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "lr_batchsize_tuning_output"),
        help="Directory to save tuning outputs",
    )
    # Default grid; can be overridden from the command line.
    parser.add_argument("--lrs", default="3e-4, 1e-3, 3e-3", help="Comma-separated learning rates")
    parser.add_argument("--batch-sizes", default="8192,16384,32768", help="Comma-separated batch sizes")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per trial")
    parser.add_argument("--val-fraction", type=float, default=0.10, help="Validation fraction")
    parser.add_argument("--low-q", type=float, default=0.001, help="Low quantile for clipping")
    parser.add_argument("--high-q", type=float, default=0.999, help="High quantile for clipping")
    parser.add_argument("--n-sig-bins", type=int, default=20, help="Bins for binned Asimov Z")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Early stopping patience on val AUC (<=0 disables)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val AUC improvement to reset early stopping",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    args = parser.parse_args()

    set_seed(args.seed)

    # Convert the comma-separated CLI values into the actual search grid.
    lrs = parse_float_list(args.lrs)
    batch_sizes = parse_int_list(args.batch_sizes)
    device = resolve_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading data from {args.input_npz}")

    data = np.load(args.input_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    w = data["w"].astype(np.float32)
    event_number = data["event_number"].astype(np.int64)

    print(f"Loaded events: {len(X):,}")
    print(f"Signal count: {(y == 1).sum():,}")
    print(f"Background count: {(y == 0).sum():,}")
    print(f"Total trials: {len(lrs) * len(batch_sizes)}")

    trial_summaries = []
    best_summary = None

    # Evaluate every (lr, batch_size) combination in the grid.
    for lr, batch_size in product(lrs, batch_sizes):
        trial_summary = evaluate_trial(
            X=X,
            y=y,
            w=w,
            event_number=event_number,
            device=device,
            batch_size=batch_size,
            lr=lr,
            n_epochs=args.epochs,
            val_fraction=args.val_fraction,
            low_q=args.low_q,
            high_q=args.high_q,
            n_sig_bins=args.n_sig_bins,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
        )
        trial_summaries.append(trial_summary)

        trial_name = format_trial_name(lr, batch_size)
        with open(os.path.join(args.outdir, f"{trial_name}.json"), "w") as handle:
            json.dump(trial_summary, handle, indent=2)

        if best_summary is None:
            best_summary = trial_summary
            continue

        best_key = (
            best_summary["combined_binned_asimov_z"],
            best_summary["combined_auc"],
        )
        current_key = (
            trial_summary["combined_binned_asimov_z"],
            trial_summary["combined_auc"],
        )
        if current_key > best_key:
            best_summary = trial_summary

    trial_summaries.sort(
        key=lambda item: (item["combined_binned_asimov_z"], item["combined_auc"]),
        reverse=True,
    )

    with open(os.path.join(args.outdir, "all_results.json"), "w") as handle:
        json.dump(trial_summaries, handle, indent=2)

    with open(os.path.join(args.outdir, "best_result.json"), "w") as handle:
        json.dump(best_summary, handle, indent=2)

    print("\n" + "=" * 80)
    print("Best trial")
    print("=" * 80)
    print(f"lr = {best_summary['lr']}")
    print(f"batch_size = {best_summary['batch_size']}")
    print(f"combined AUC = {best_summary['combined_auc']:.6f}")
    print(f"combined binned Asimov Z = {best_summary['combined_binned_asimov_z']:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()