import argparse
import json
import csv
import ast
import pandas as pd
import os
import re
import sys
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np


# example available data
# exp_id,step,pre_state,state,prev_action,gt,gt_noisy
# 2025-09-09-17-49-02-online-tamer-noisy_v2,0,"[""66666666"",""61000006"",""60000006"",""60104306"",""63005006"",""60000006"",""60000006"",""66666666""]","[""66666666"",""61000006"",""60004006"",""60105306"",""63000006"",""60000006"",""60000006"",""66666666""]",0,0,0


from robotaxi.agent.tamer_agent import compute_delta_features_v3, compute_delta_features_v2, compute_delta_features_v4

def fmt_feat(arr: np.ndarray) -> str:
    # Match feature style: space-separated without commas
    return np.array2string(arr, separator=' ')

def fmt_weight(arr: np.ndarray) -> str:
    # Match weight style: comma-separated with spaces and prevent line wraps
    return np.array2string(arr, separator=', ', max_line_width=np.inf)

# Load input data from csv, return list of tuples of (prev_state, state, gt, gt_noisy)
def load_csv(path: str, shuffle: bool = True, n_samples: int = None, skip_samples: int = 0, proxy_acc: float = 1.0, negative_feedback_ratio = None) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Parse a CSV exported from online TAMER logs using pandas.

    Expected columns (order not enforced): exp_id, step, pre_state, state, prev_action, gt, gt_noisy
    Returns list of (prev_state_grid, state_grid, gt, gt_noisy).
    """
    def parse_rows_field(val: str) -> List[str]:
        if val is None:
            raise ValueError("empty state field")
        # Try JSON first
        try:
            out = json.loads(val)
        except Exception:
            # Try Python literal
            try:
                out = ast.literal_eval(val)
            except Exception:
                # Last resort: fix common quoting
                sanitized = val.replace("''", '"').replace("'", '"')
                out = json.loads(sanitized)
        if not isinstance(out, list):
            raise ValueError("state field is not a list")
        return out

    def parse_grid(list_of_row_strings: List[str]) -> np.ndarray:
        grid_rows: List[List[int]] = []
        for row_str in list_of_row_strings:
            grid_rows.append([int(ch) for ch in row_str])
        return np.array(grid_rows, dtype=int)

    df = pd.read_csv(path)
    # Normalize possible alternative column names
    col_pre = "pre_state" if "pre_state" in df.columns else ("prev_state" if "prev_state" in df.columns else ("previous_state" if "previous_state" in df.columns else None))
    col_cur = "state" if "state" in df.columns else ("current_state" if "current_state" in df.columns else None)
    col_gt = "gt" if "gt" in df.columns else ("reward_gt" if "reward_gt" in df.columns else None)
    col_gt_noisy = "gt_noisy" if "gt_noisy" in df.columns else ("reward_noisy" if "reward_noisy" in df.columns else None)

    if not all([col_pre, col_cur, col_gt, col_gt_noisy]):
        return []

    samples: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
    if skip_samples > 0:
        df = df.iloc[skip_samples:]
    if n_samples is not None:
        assert n_samples <= len(df), "n_samples is greater than the number of samples in the csv"
        df = df.iloc[:n_samples]
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
        
    if proxy_acc != 1.0:
        # randomly flip the gt before returning
        df[col_gt] = df[col_gt].apply(lambda x: 1-x if np.random.rand() > proxy_acc else x)
        
    for _, row in df.iterrows():
        try:
            pre_list = parse_rows_field(row[col_pre])
            cur_list = parse_rows_field(row[col_cur])
            prev_grid = parse_grid(pre_list)
            curr_grid = parse_grid(cur_list)
            gt_val = float(row[col_gt])
            gt_noisy_val = float(row[col_gt_noisy])
        except Exception:
            continue
        samples.append((prev_grid, curr_grid, gt_val, gt_noisy_val))

    return samples

def offline_train_from_features(
    samples: List[Tuple[np.ndarray, np.ndarray, float, float]], # prev_state, state, gt, gt_noisy
    alpha: float,
    alpha_decay: float,
    epochs: int,
    seed: int,
    init_w: Optional[np.ndarray] = None,
    out_log_path: Optional[str] = None,
    use_noisy_reward: bool = False,
    feature_version: str = "",
) -> Tuple[np.ndarray, List[float]]:
    """Run simple TAMER-like offline updates from logged features and user rewards.

    Update rule per step s:
        delta_f = compute_delta_features_v3(state_prev, state_curr)
        projected = w · delta_f
        error = user_rew - projected
        w += alpha * error * delta_f
    Returns final weights and list of per-epoch sum of absolute errors.
    """
    rng = np.random.default_rng(seed)

    if init_w is None:
        # infer feature dimension from first sample
        if not samples:
            raise ValueError("No samples provided for training")
        dim = samples[0][0].shape[0]
        init_w = rng.uniform(-1.0, 1.0, size=dim).astype(float)

    w = init_w.astype(float).copy()
    epoch_abs_error_sums: List[float] = []

    # Prepare logger if requested
    log_fh = None
    if out_log_path is not None:
        os.makedirs(os.path.dirname(out_log_path) or ".", exist_ok=True)
        log_fh = open(out_log_path, "w")

    alpha0 = alpha
    step = 0
    for _ in range(epochs):
        # shuffle samples each epoch
        indices = np.arange(len(samples))
        abs_err_sum = 0.0
        for idx in indices:
            state_prev, state_curr, user_rew, user_rew_noisy = samples[idx]
            if use_noisy_reward:
                user_rew = user_rew_noisy
            if user_rew == 0:
                continue
            assert feature_version in ["v2", "v3", "v4"]
            delta_f = compute_delta_features_v3(state_prev, state_curr).astype(float)
            if feature_version == "v2":
                delta_f = compute_delta_features_v2(state_prev, state_curr).astype(float)
            if feature_version == "v4":
                delta_f = compute_delta_features_v4(state_prev, state_curr).astype(float)
            projected = float(np.dot(w, delta_f))
            error = float(user_rew - projected)

            # Log before updating to mimic online log semantics
            if log_fh is not None:
                log_fh.write(f"[FEATURE] f_prev: {[0,0,0,0,0,0]}, f_curr: {[0,0,0,0,0,0]}\n")
                log_fh.write(f"[REWARD] Projected Reward: {projected}, user_rew: {user_rew}, error: {error}\n")
                log_fh.write(f"[WEIGHT] self.w: {fmt_weight(w)}\n")
                log_fh.write("============\n")

            w += alpha0 * (alpha_decay ** step) * error * delta_f
            abs_err_sum += abs(error)
            step += 1
        epoch_abs_error_sums.append(abs_err_sum)

    if log_fh is not None:
        log_fh.close()

    return w, epoch_abs_error_sums


def main():
    parser = argparse.ArgumentParser(description="Offline TAMER training from [FEATURE]/[REWARD] logs")
    parser.add_argument("log_paths", nargs='+', type=str, help="One or more log files with [FEATURE]/[REWARD]/[WEIGHT] lines")
    parser.add_argument("--alpha", type=float, default=0.01, help="Learning rate alpha")
    parser.add_argument("--alpha_decay", type=float, default=1, help="Learning rate decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (passes over the log)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for init and shuffling")
    parser.add_argument("--init-w", choices=["w1", "w2", "w10", "w20", "random", "log"], default="w1", help="How to initialize weights: preset w1/w2, random, or from first [WEIGHT] in logs")
    parser.add_argument("--save", type=str, default=None, help="Path to save final weights .npy; defaults to weights_<timestamp>.npy")
    parser.add_argument("--out-log", type=str, default=None, help="Path to write the offline training log mirroring the input log format")
    parser.add_argument("--eval", action="store_true", help="Run a quick evaluation after training")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of eval episodes if --eval is set")
    parser.add_argument("--eval-max-steps", type=int, default=100, help="Max steps per episode during eval")
    parser.add_argument("--shuffle", action="store_true", help="Do not shuffle samples")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--skip-samples", type=int, default=0, help="Number of samples to skip from the beginning")
    parser.add_argument("--proxy_acc", type=float, default=1.0, help="Proxy accuracy for noisy reward")
    parser.add_argument("--use_noisy_reward", action="store_true", help="Use noisy reward for training")
    parser.add_argument("--negative_feedback_ratio", type=float, default=0.0, help="Negative feedback ratio")
    parser.add_argument("--feature_version", type=str, required=True, help="Feature version")
    args = parser.parse_args()
    
    if args.use_noisy_reward:
        assert args.proxy_acc == 1.0, "Proxy accuracy must be 1.0 when using noisy reward"

    # Preset weight map
    w_map = {
        "w1": np.array([ 0.09894706, -0.01005191, -0.05182143, -0.02420872,  0.03493194, -0.08516584]),
        "w2": np.array([ 0.07768216, -0.09317696, -0.05280239,  0.03182322, -0.02639944, 0.09958436]),
        "w10": np.array([ 0.9894706, -0.1005191, -0.5182143, -0.2420872,  0.3493194, -0.8516584]),
        "w20": np.array([ 0.7768216, -0.9317696, -0.5280239,  0.3182322, -0.2639944, 0.9958436]),
        "w30": np.array([ 0.7768216, -0.9317696, +0.5280239,  -0.3182322, -0.2639944, 0.9958436]),
    }

    # Parse all inputs in order
    all_samples: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
    total_parsed = 0
    for path in args.log_paths:
        # def load_csv(path: str, shuffle: bool = True, n_samples: int = None, skip_samples: int = 0, proxy_acc: float = 1.0, negative_feedback_ratio: None
        s = load_csv(path, shuffle=args.shuffle, n_samples=args.n_samples, skip_samples=args.skip_samples, proxy_acc=args.proxy_acc, negative_feedback_ratio=args.negative_feedback_ratio)
        all_samples.extend(s)
        total_parsed += len(s)

    if not all_samples:
        print(f"No (f_prev, f_curr, user_rew, user_rew_noisy) samples parsed from inputs: {args.log_paths}")
        sys.exit(1)

    print(f"Parsed {total_parsed} samples from {len(args.log_paths)} file(s)")

    # Resolve initialization
    init_w: Optional[np.ndarray]
    if args.init_w in w_map.keys():
        init_w = w_map[args.init_w]
        print(f"Initializing w from preset {args.init_w}: {init_w}")
    else:
        assert False, "Invalid init_w"

    # Resolve output log path default
    out_log_path = args.out_log
    if out_log_path is None:
        base = os.path.splitext(os.path.basename(args.log_paths[0]))[0]
        out_log_path = os.path.join(os.path.dirname(args.log_paths[0]), f"offline-{base}.log")

    final_w, epoch_abs_err_sums = offline_train_from_features(
        samples=all_samples,
        alpha=args.alpha,
        alpha_decay=args.alpha_decay,
        epochs=args.epochs,
        seed=args.seed,
        init_w=init_w,
        out_log_path=out_log_path,
        use_noisy_reward=args.use_noisy_reward,
        feature_version=args.feature_version,
    )

    print("Training complete.")
    for i, val in enumerate(epoch_abs_err_sums):
        print(f"Epoch {i+1} sum |error| = {val:.6f}")
    print(f"Final weights: {final_w}")

    # save_path = args.save
    # if not save_path:
    #     ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     base = os.path.splitext(os.path.basename(args.log_paths[0]))[0]
    #     save_path = f"tamer_weights_offline_{base}_{ts}.npy"
    # np.save(save_path, final_w)
    # print(f"Saved weights to {save_path}")

    print(f"Offline training log written to {out_log_path}")

    if args.eval:
        avg_rew = maybe_evaluate(final_w, num_episodes=args.eval_episodes, max_steps=args.eval_max_steps)
        print(f"EVALUATION: Average reward over {args.eval_episodes} episodes: {avg_rew}")


if __name__ == "__main__":
    main()



