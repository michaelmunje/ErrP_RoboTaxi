#!/usr/bin/env python3

import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from robotaxi.gameplay.wrappers import preprocess_observation_tamer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def handle_collision(f_curr: np.ndarray, f_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f_curr = f_curr.copy()
    f_prev = f_prev.copy()
    if f_curr[0] < f_prev[0]:
        f_curr[2] = 0
        f_curr[4] = 0
    if f_curr[1] < f_prev[1]:
        f_curr[3] = 0
        f_curr[5] = 0
    return f_curr, f_prev


def json_grid_to_array(json_text: str) -> np.ndarray:
    rows: List[str] = json.loads(json_text)
    assert len(rows) == 8
    grid = np.array([[int(ch) for ch in row] for row in rows], dtype=np.int32)
    assert grid.shape == (8, 8)
    return grid


def build_datasets_from_file(aggregated_csv: str):
    pre_states: List[np.ndarray] = []
    states: List[np.ndarray] = []
    gts: List[int] = []

    import csv
    with open(aggregated_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            _, _, pre_state_json, state_json, gt_str = row
            pre_states.append(json_grid_to_array(pre_state_json))
            states.append(json_grid_to_array(state_json))
            gts.append(int(gt_str))

    delta_feats: List[np.ndarray] = []
    concat_feats: List[np.ndarray] = []
    for pre, state in zip(pre_states, states):
        f_prev = preprocess_observation_tamer(pre).astype(np.float32)
        f_curr = preprocess_observation_tamer(state).astype(np.float32)
        f_curr_adj, f_prev_adj = handle_collision(f_curr, f_prev)
        delta = (f_curr_adj - f_prev_adj).astype(np.float32)
        concat = np.concatenate([f_prev_adj, f_curr_adj], axis=0).astype(np.float32)
        delta_feats.append(delta)
        concat_feats.append(concat)

    X_delta_feat = np.stack(delta_feats, axis=0)
    X_concat_feat = np.stack(concat_feats, axis=0)
    y = np.array(gts, dtype=np.int64)

    return (X_delta_feat, X_concat_feat, y)


def build_datasets_combined(paths: List[str]):
    arrays = [build_datasets_from_file(p) for p in paths]
    Xd_list, Xc_list, y_list = zip(*arrays)
    Xd = np.concatenate(Xd_list, axis=0)
    Xc = np.concatenate(Xc_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return Xd, Xc, y


class NNDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate_balanced_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            ps.append(prob)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return balanced_accuracy_score(y_true, y_pred)


def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    logits_list = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            logits_list.append(logits.cpu().numpy())
    return np.concatenate(logits_list)


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    logits = predict_logits(model, loader, device)
    return 1.0 / (1.0 + np.exp(-logits))


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T) + 1e-6
        return logits / T


def fit_temperature_on_calib(model: nn.Module, calib_loader: DataLoader, device: torch.device) -> TemperatureScaler:
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for Xb, yb in calib_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            logits_list.append(logits)
            labels_list.append(yb.float().to(device))
    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_all)
        loss = criterion(scaled, labels_all)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


def apply_temperature(model: nn.Module, loader: DataLoader, scaler: TemperatureScaler, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            scaled = scaler(logits)
            prob = torch.sigmoid(scaled).cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs)


def ppv_npv_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[float, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ppv = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else np.nan
    return ppv, npv


def best_threshold_min_ppv_npv(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    best_t = 0.5
    best_min_val = -np.inf
    best_ppv = np.nan
    best_npv = np.nan
    for t in np.linspace(0.01, 0.99, 99):
        ppv, npv = ppv_npv_from_probs(y_true, y_prob, float(t))
        if np.isnan(ppv) or np.isnan(npv):
            continue
        mval = min(ppv, npv)
        if mval > best_min_val:
            best_min_val = mval
            best_t = float(t)
            best_ppv = ppv
            best_npv = npv
    return best_t, best_ppv, best_npv


def best_threshold_sum_ppv_npv(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    best_t = 0.5
    best_sum_val = -np.inf
    best_ppv = np.nan
    best_npv = np.nan
    for t in np.linspace(0.01, 0.99, 99):
        ppv, npv = ppv_npv_from_probs(y_true, y_prob, float(t))
        if np.isnan(ppv) or np.isnan(npv):
            continue
        sval = ppv + npv
        if sval > best_sum_val:
            best_sum_val = sval
            best_t = float(t)
            best_ppv = ppv
            best_npv = npv
    return best_t, best_ppv, best_npv


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight_value: float,
    eval_every: int,
    patience: int,
    verbose: bool,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    train_loader = DataLoader(NNDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(NNDataset(X_cal, y_cal), batch_size=batch_size)
    val_loader = DataLoader(NNDataset(X_val, y_val), batch_size=batch_size)

    best_state = None
    best_val_ba = -1.0
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if epoch % eval_every == 0:
            val_ba = evaluate_balanced_acc(model, calib_loader, device)
            improved = val_ba > best_val_ba + 1e-6
            if improved:
                best_val_ba = val_ba
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if verbose:
                print(f"  epoch {epoch:03d}: calib_bal_acc={val_ba:.3f} best={best_val_ba:.3f}")

            if epochs_since_improve >= patience:
                if verbose:
                    print("  early stopping (no improvement)")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in model.state_dict().items()})

    scaler = fit_temperature_on_calib(model, calib_loader, device)
    y_prob = apply_temperature(model, val_loader, scaler, device)
    return y_prob


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    hidden: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight_value: float,
    eval_every: int,
    patience: int,
    verbose: bool,
    thr_objective: str,
    n_splits: int = 5,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 1
    ba_scores: List[float] = []
    ppv_scores: List[float] = []
    npv_scores: List[float] = []
    thresholds: List[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        y_prob = train_one_fold(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden,
            epochs,
            batch_size,
            lr,
            pos_weight_value,
            eval_every,
            patience,
            verbose,
        )

        if thr_objective == "sum":
            t_best, ppv_best, npv_best = best_threshold_sum_ppv_npv(y_val, y_prob)
        else:
            t_best, ppv_best, npv_best = best_threshold_min_ppv_npv(y_val, y_prob)
        y_pred = (y_prob >= t_best).astype(np.int64)
        ba = balanced_accuracy_score(y_val, y_pred)

        ba_scores.append(ba)
        ppv_scores.append(ppv_best)
        npv_scores.append(npv_best)
        thresholds.append(t_best)

        print(
            f"{model_name} (pos_w={pos_weight_value}) Fold {fold}: "
            f"balanced_acc={ba:.3f}, PPV={ppv_best:.3f}, NPV={npv_best:.3f}, thr={t_best:.2f}"
        )
        fold += 1

    mean_ba = float(np.mean(ba_scores))
    std_ba = float(np.std(ba_scores))
    mean_ppv = float(np.nanmean(ppv_scores))
    std_ppv = float(np.nanstd(ppv_scores))
    mean_npv = float(np.nanmean(npv_scores))
    std_npv = float(np.nanstd(npv_scores))
    mean_thr = float(np.mean(thresholds))

    print(
        f"{model_name} (pos_w={pos_weight_value}) {n_splits}-fold: "
        f"balanced_acc={mean_ba:.3f}±{std_ba:.3f}, "
        f"PPV={mean_ppv:.3f}±{std_ppv:.3f}, NPV={mean_npv:.3f}±{std_npv:.3f}, thr≈{mean_thr:.2f}"
    )


def run_sweep(
    X: np.ndarray,
    y: np.ndarray,
    model_label: str,
    hidden: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight_grid: List[float],
    eval_every: int,
    patience: int,
    verbose: bool,
    thr_objective: str,
    n_splits: int = 5,
):
    for pw in pos_weight_grid:
        cross_validate(X, y, model_label, hidden, epochs, batch_size, lr, pw, eval_every, patience, verbose, thr_objective, n_splits)


def main():
    parser = argparse.ArgumentParser(description="CV with pos_weight sweep, calibration, and threshold tuning (min or sum of PPV/NPV).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight-grid", type=str, default="3")
    parser.add_argument("--thr-objective", choices=["min", "sum"], default="sum")
    parser.add_argument("--s6", default="/home/zhihan/Documents/Code/ErrP_RoboTaxi/phase2-selected/subject6/additional_log_files/aggregated_states_gt_subject6.csv")
    parser.add_argument("--s7", default="/home/zhihan/Documents/Code/ErrP_RoboTaxi/phase2-selected/subject7/additional_log_files/aggregated_states_gt_subject7.csv")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    def resolve_path(pref: str, alt_dir: str):
        if os.path.isfile(pref):
            return pref
        alt = os.path.join(alt_dir, "aggregated_states_gt.csv")
        if os.path.isfile(alt):
            return alt
        raise FileNotFoundError(f"Neither '{pref}' nor '{alt}' found")

    s6_path = resolve_path(
        args.s6,
        "/home/zhihan/Documents/Code/ErrP_RoboTaxi/phase2-selected/subject6/additional_log_files",
    )
    s7_path = resolve_path(
        args.s7,
        "/home/zhihan/Documents/Code/ErrP_RoboTaxi/phase2-selected/subject7/additional_log_files",
    )

    print("Loading datasets ...")
    Xd6, Xc6, y6 = build_datasets_from_file(s6_path)
    Xd7, Xc7, y7 = build_datasets_from_file(s7_path)

    # hidden_delta = [32, 32, 16]
    hidden_delta = [32, 16]
    # hidden_concat = [32, 32, 16]
    hidden_concat = [32, 16]
    verbose = not args.quiet
    pos_weight_grid = [float(x) for x in args.pos_weight_grid.split(",")]

    print("\nSubject 6 only (pos_weight sweep):")
    run_sweep(Xd6, y6, "DeltaFeat-MLP (S6)", hidden_delta, args.epochs, args.batch_size, args.lr, pos_weight_grid, args.eval_every, args.patience, verbose, args.thr_objective)
    run_sweep(Xc6, y6, "ConcatFeat-MLP (S6)", hidden_concat, args.epochs, args.batch_size, args.lr, pos_weight_grid, args.eval_every, args.patience, verbose, args.thr_objective)

    print("\nSubject 7 only (pos_weight sweep):")
    run_sweep(Xd7, y7, "DeltaFeat-MLP (S7)", hidden_delta, args.epochs, args.batch_size, args.lr, pos_weight_grid, args.eval_every, args.patience, verbose, args.thr_objective)
    run_sweep(Xc7, y7, "ConcatFeat-MLP (S7)", hidden_concat, args.epochs, args.batch_size, args.lr, pos_weight_grid, args.eval_every, args.patience, verbose, args.thr_objective)

    print("\nCombined Subject 6 + 7 (pos_weight sweep):")
    Xd_both = np.concatenate([Xd6, Xd7], axis=0)
    Xc_both = np.concatenate([Xc6, Xc7], axis=0)
    y_both = np.concatenate([y6, y7], axis=0)
    run_sweep(Xd_both, y_both, "DeltaFeat-MLP (Both)", hidden_delta, args.epochs, args.batch_size, args.lr, pos_weight_grid, args.eval_every, args.patience, verbose, args.thr_objective)
    run_sweep(Xc_both, y_both, "ConcatFeat-MLP (Both)", hidden_concat, args.epochs, args.batch_size, args.lr, pos_weight_grid, args.eval_every, args.patience, verbose, args.thr_objective)

    print("\nRecent Combined (last 400 each) (pos_weight sweep):")
    def tail400(X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        k = min(400, n)
        return X[-k:], y[-k:]

    Xd6_recent, y6_recent = tail400(Xd6, y6)
    Xc6_recent, _ = tail400(Xc6, y6)
    Xd7_recent, y7_recent = tail400(Xd7, y7)
    Xc7_recent, _ = tail400(Xc7, y7)

    Xd_recent_both = np.concatenate([Xd6_recent, Xd7_recent], axis=0)
    Xc_recent_both = np.concatenate([Xc6_recent, Xc7_recent], axis=0)
    y_recent_both = np.concatenate([y6_recent, y7_recent], axis=0)

    run_sweep(Xd_recent_both, y_recent_both, "DeltaFeat-MLP (RecentBoth400)", hidden_delta, args.epochs, args.batch_size, args.lr, pos_weight_grid, args.eval_every, args.patience, verbose, args.thr_objective)
    run_sweep(Xc_recent_both, y_recent_both, "ConcatFeat-MLP (RecentBoth400)", hidden_concat, args.epochs, args.batch_size, args.lr, pos_weight_grid, args.eval_every, args.patience, verbose, args.thr_objective)

    print("\nSubject 6 (last 400) 3-fold:")
    run_sweep(Xd6_recent, y6_recent, "DeltaFeat-MLP (S6 last400)", hidden_delta, args.epochs, args.batch_size, args.lr, [pos_weight_grid[0]], args.eval_every, args.patience, verbose, args.thr_objective, n_splits=3)
    run_sweep(Xc6_recent, y6_recent, "ConcatFeat-MLP (S6 last400)", hidden_concat, args.epochs, args.batch_size, args.lr, [pos_weight_grid[0]], args.eval_every, args.patience, verbose, args.thr_objective, n_splits=3)

    print("\nSubject 7 (last 400) 3-fold:")
    run_sweep(Xd7_recent, y7_recent, "DeltaFeat-MLP (S7 last400)", hidden_delta, args.epochs, args.batch_size, args.lr, [pos_weight_grid[0]], args.eval_every, args.patience, verbose, args.thr_objective, n_splits=3)
    run_sweep(Xc7_recent, y7_recent, "ConcatFeat-MLP (S7 last400)", hidden_concat, args.epochs, args.batch_size, args.lr, [pos_weight_grid[0]], args.eval_every, args.patience, verbose, args.thr_objective, n_splits=3)


if __name__ == "__main__":
    main()
