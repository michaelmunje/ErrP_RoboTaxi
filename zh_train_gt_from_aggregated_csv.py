import argparse
import csv
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from robotaxi.gameplay.wrappers import preprocess_observation_tamer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_grid_to_array(json_text: str) -> np.ndarray:
    rows: List[str] = json.loads(json_text)
    grid = np.array([[int(ch) for ch in row] for row in rows], dtype=np.int32)
    return grid


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


def load_aggregated_csv(path: str):
    pre_states: List[np.ndarray] = []
    states: List[np.ndarray] = []
    gts: List[int] = []

    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_pre = header.index("pre_state")
        idx_state = header.index("state")
        idx_gt = header.index("gt")
        for row in reader:
            if not row:
                continue
            pre_states.append(json_grid_to_array(row[idx_pre]))
            states.append(json_grid_to_array(row[idx_state]))
            gts.append(int(row[idx_gt]))
    return pre_states, states, gts


def build_delta_features(pre_states: List[np.ndarray], states: List[np.ndarray]) -> np.ndarray:
    delta_feats: List[np.ndarray] = []
    for pre, state in zip(pre_states, states):
        f_prev = preprocess_observation_tamer(pre).astype(np.float32)
        f_curr = preprocess_observation_tamer(state).astype(np.float32)
        f_curr_adj, f_prev_adj = handle_collision(f_curr, f_prev)
        delta = (f_curr_adj - f_prev_adj).astype(np.float32)
        delta_feats.append(delta)
    return np.stack(delta_feats, axis=0)


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


def ppv_npv_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[float, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ppv = (tp / (tp + fp)) if (tp + fp) > 0 else float('nan')
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else float('nan')
    return ppv, npv


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return (t_minmax, ppv_minmax, npv_minmax, t_equal, diff_min) where:
    - t_minmax maximizes min(PPV, NPV)
    - t_equal minimizes |PPV - NPV|
    """
    best_t_minmax = 0.5
    best_min_val = -1.0
    best_ppv_mm = float('nan')
    best_npv_mm = float('nan')

    best_t_equal = 0.5
    best_abs_diff = float('inf')

    for t in np.linspace(0.01, 0.99, 99):
        ppv, npv = ppv_npv_from_probs(y_true, y_prob, float(t))
        if np.isnan(ppv) or np.isnan(npv):
            continue
        mval = min(ppv, npv)
        if mval > best_min_val:
            best_min_val = mval
            best_t_minmax = float(t)
            best_ppv_mm = ppv
            best_npv_mm = npv
        diff = abs(ppv - npv)
        if diff < best_abs_diff:
            best_abs_diff = diff
            best_t_equal = float(t)

    return best_t_minmax, best_ppv_mm, best_npv_mm, best_t_equal, best_abs_diff


def train_and_eval(X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, lr: float, eval_every: int, patience: int, quiet: bool, hidden: List[int]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = MLP(input_dim=X.shape[1], hidden_dims=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(NNDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NNDataset(X_val, y_val), batch_size=batch_size)

    best_ba = -1.0
    epochs_no_improve = 0

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
            # Evaluate balanced accuracy at 0.5 threshold
            model.eval()
            ys = []
            ps = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(device)
                    logits = model(Xb)
                    prob = torch.sigmoid(logits).cpu().numpy()
                    ps.append(prob)
                    ys.append(yb.numpy())
            y_true = np.concatenate(ys)
            y_prob = np.concatenate(ps)
            y_pred = (y_prob >= 0.5).astype(np.int64)
            ba = balanced_accuracy_score(y_true, y_pred)
            if not quiet:
                print(f"epoch {epoch:03d}: val_bal_acc={ba:.3f}")
            if ba > best_ba + 1e-6:
                best_ba = ba
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if not quiet:
                        print("early stopping")
                    break

    # Final evaluation on validation set for threshold tuning
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            ps.append(prob)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    # Accuracy at threshold 0.5
    y_pred_05 = (y_prob >= 0.5).astype(np.int64)
    acc_05 = float((y_pred_05 == y_true).mean())

    t_minmax, ppv_mm, npv_mm, t_equal, diff_min = find_best_thresholds(y_true, y_prob)
    if not quiet:
        print(f"Best balanced accuracy: {best_ba:.3f}")
        print(f"Accuracy at cutoff=0.50: {acc_05:.3f}")
        print(f"Cutoff (maximize min(PPV,NPV)) t={t_minmax:.2f}: PPV={ppv_mm:.3f}, NPV={npv_mm:.3f}")
        ppv_eq, npv_eq = ppv_npv_from_probs(y_true, y_prob, t_equal)
        print(f"Cutoff (min |PPV-NPV|) t={t_equal:.2f}: PPV={ppv_eq:.3f}, NPV={npv_eq:.3f}")

    return best_ba


def train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int, lr: float, eval_every: int, patience: int, quiet: bool, hidden: List[int]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X_tr.shape[1], hidden_dims=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(NNDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NNDataset(X_val, y_val), batch_size=batch_size)

    best_ba = -1.0
    epochs_no_improve = 0

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
            # Evaluate balanced accuracy at 0.5 threshold
            model.eval()
            ys = []
            ps = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(device)
                    logits = model(Xb)
                    prob = torch.sigmoid(logits).cpu().numpy()
                    ps.append(prob)
                    ys.append(yb.numpy())
            y_true = np.concatenate(ys)
            y_prob = np.concatenate(ps)
            y_pred = (y_prob >= 0.5).astype(np.int64)
            ba = balanced_accuracy_score(y_true, y_pred)
            if not quiet:
                print(f"  epoch {epoch:03d}: val_bal_acc={ba:.3f}")
            if ba > best_ba + 1e-6:
                best_ba = ba
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if not quiet:
                        print("  early stopping")
                    break

    # Final evaluation on validation set for threshold tuning
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            ps.append(prob)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    acc_05 = float(((y_prob >= 0.5).astype(np.int64) == y_true).mean())
    t_minmax, ppv_mm, npv_mm, t_equal, diff_min = find_best_thresholds(y_true, y_prob)
    return best_ba, acc_05, t_minmax, ppv_mm, npv_mm, t_equal


def cross_validate_upsampled(X: np.ndarray, y: np.ndarray, hidden: List[int], epochs: int, batch_size: int, lr: float, eval_every: int, patience: int, quiet: bool, seed: int, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bas: List[float] = []
    accs: List[float] = []
    ppvs: List[float] = []
    npvs: List[float] = []
    ts: List[float] = []
    fold = 1
    for train_idx, val_idx in skf.split(X, y):
        X_tr_raw, X_val = X[train_idx], X[val_idx]
        y_tr_raw, y_val = y[train_idx], y[val_idx]

        # Upsample positives in training set only
        pos_indices = np.where(y_tr_raw == 1)[0]
        neg_indices = np.where(y_tr_raw == 0)[0]
        rng = np.random.default_rng(seed + fold)
        if len(pos_indices) < len(neg_indices):
            sampled_pos = rng.choice(pos_indices, size=len(neg_indices), replace=True)
            keep_tr = np.concatenate([neg_indices, sampled_pos])
        else:
            keep_tr = np.arange(len(y_tr_raw))

        X_tr = X_tr_raw[keep_tr]
        y_tr = y_tr_raw[keep_tr]

        if not quiet:
            print(f"Fold {fold}: train size={len(y_tr)} (pos={int((y_tr==1).sum())}, neg={int((y_tr==0).sum())}), val size={len(y_val)}")

        ba, acc05, t_mm, ppv_mm, npv_mm, t_eq = train_one_fold(
            X_tr, y_tr, X_val, y_val, epochs, batch_size, lr, eval_every, patience, quiet, hidden
        )
        bas.append(ba)
        accs.append(acc05)
        ppvs.append(ppv_mm)
        npvs.append(npv_mm)
        ts.append(t_mm)
        fold += 1

    print(
        f"CV {n_splits}-fold (hidden={hidden}): BA={np.mean(bas):.3f}±{np.std(bas):.3f}, Acc@0.5={np.mean(accs):.3f}±{np.std(accs):.3f}, "
        f"PPV={np.mean(ppvs):.3f}±{np.std(ppvs):.3f}, NPV={np.mean(npvs):.3f}±{np.std(npvs):.3f}, thr≈{np.mean(ts):.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train classifier from aggregated CSV with optional balancing")
    parser.add_argument("--csv", required=True, help="Path to aggregated CSV")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance", action="store_true", help="Downsample gt=0 to match count of gt=-1")
    parser.add_argument("--upsample-pos", action="store_true", help="Upsample gt=-1 to match count of gt=0 (keep all data)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cv", type=int, default=0, help="If >0, run Stratified K-fold CV with this many folds")
    args = parser.parse_args()

    set_seed(args.seed)
    pre_states, states, gts = load_aggregated_csv(args.csv)

    # Filter to gt in {-1, 0}
    idxs = [i for i, g in enumerate(gts) if g in (-1, 0)]
    gts = [gts[i] for i in idxs]
    pre_states = [pre_states[i] for i in idxs]
    states = [states[i] for i in idxs]

    # Count classes
    num_pos = sum(1 for g in gts if g == -1)
    num_neg = sum(1 for g in gts if g == 0)
    print(f"Counts -> positive(gt=-1): {num_pos}, negative(gt=0): {num_neg}")

    # Create labels: pos=1 for gt=-1, neg=0 for gt=0
    y = np.array([1 if g == -1 else 0 for g in gts], dtype=np.int64)

    # Sampling strategy
    if args.balance and args.upsample_pos:
        print("Both --balance and --upsample-pos specified; defaulting to upsample.")
        args.balance = False

    if args.balance and num_pos > 0 and num_neg > 0 and num_neg > num_pos:
        # Downsample negatives to match positives
        pos_indices = [i for i, lab in enumerate(y) if lab == 1]
        neg_indices = [i for i, lab in enumerate(y) if lab == 0]
        rng = np.random.default_rng(args.seed)
        sampled_neg = rng.choice(neg_indices, size=num_pos, replace=False)
        keep = np.array(sorted(list(pos_indices) + list(sampled_neg)))
        print(f"Using downsampled set: pos={len(pos_indices)}, neg={len(sampled_neg)}")
    elif args.upsample_pos and num_pos > 0 and num_neg > 0 and num_pos < num_neg:
        # Upsample positives (with replacement) to match negatives, keep all negatives
        pos_indices = [i for i, lab in enumerate(y) if lab == 1]
        neg_indices = [i for i, lab in enumerate(y) if lab == 0]
        rng = np.random.default_rng(args.seed)
        sampled_pos = rng.choice(pos_indices, size=num_neg, replace=True)
        keep = np.array(sorted(list(neg_indices) + list(sampled_pos)))
        print(f"Using upsampled set: pos={len(sampled_pos)}, neg={len(neg_indices)}")
    else:
        keep = np.arange(len(y))

    X = build_delta_features([pre_states[i] for i in keep], [states[i] for i in keep])
    y_bal = y[keep]

    # Fixed architecture
    hidden = [32, 16]

    if args.cv and args.cv > 1:
        cross_validate_upsampled(X, y_bal, hidden, args.epochs, args.batch_size, args.lr, args.eval_every, args.patience, args.quiet, args.seed, n_splits=args.cv)
    else:
        train_and_eval(X, y_bal, args.epochs, args.batch_size, args.lr, args.eval_every, args.patience, args.quiet, hidden)


if __name__ == "__main__":
    main()


