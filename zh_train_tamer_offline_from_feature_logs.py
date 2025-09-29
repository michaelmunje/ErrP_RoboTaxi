import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np


def parse_array(text: str) -> np.ndarray:
    """Parse a bracket-enclosed numeric array like "[2. 2. 3. 2. 0. 1.]" or with commas.

    Returns a 1D float numpy array.
    """
    inside = text.strip()
    if inside.startswith("[") and inside.endswith("]"):
        inside = inside[1:-1]
    # Normalize separators: replace commas with spaces, collapse whitespace
    inside = inside.replace(",", " ")
    parts = [p for p in inside.split() if p]
    return np.array([float(p) for p in parts], dtype=float)


def parse_feature_reward_log(log_path: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Optional[np.ndarray]]:
    """Parse a log file containing [FEATURE], [REWARD], and [WEIGHT] blocks.

    Returns:
      - samples: list of (f_prev, f_curr, user_rew)
      - first_weight: the first weight vector encountered in the log, if any
    """
    feature_re = re.compile(r"\[FEATURE\]\s*f_prev:\s*(\[[^\]]+\])\s*,\s*f_curr:\s*(\[[^\]]+\])")
    reward_re = re.compile(r"\[REWARD\][^\n]*user_rew:\s*([-+]?\d+(?:\.\d+)?)")
    weight_re = re.compile(r"\[WEIGHT\][^\[]*\[([^\]]+)\]")

    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    first_weight: Optional[np.ndarray] = None

    pending_f_prev: Optional[np.ndarray] = None
    pending_f_curr: Optional[np.ndarray] = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # capture first weight if present
            if first_weight is None:
                m_w = weight_re.search(line)
                if m_w:
                    first_weight = parse_array("[" + m_w.group(1) + "]")

            m_feat = feature_re.search(line)
            if m_feat:
                pending_f_prev = parse_array(m_feat.group(1))
                pending_f_curr = parse_array(m_feat.group(2))
                continue

            m_rew = reward_re.search(line)
            if m_rew and pending_f_prev is not None and pending_f_curr is not None:
                user_rew = float(m_rew.group(1))
                samples.append((pending_f_prev, pending_f_curr, user_rew))
                pending_f_prev = None
                pending_f_curr = None

    return samples, first_weight


def offline_train_from_features(
    samples: List[Tuple[np.ndarray, np.ndarray, float]],
    alpha: float,
    epochs: int,
    seed: int,
    init_w: Optional[np.ndarray] = None,
    clamp_first_two_nonpositive: bool = True,
    out_log_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Run simple TAMER-like offline updates from logged features and user rewards.

    Update rule per step s:
        delta_f = f_curr - f_prev
        if clamp: delta_f[0] = min(delta_f[0], 0); delta_f[1] = min(delta_f[1], 0)
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

    def fmt_feat(arr: np.ndarray) -> str:
        # Match feature style: space-separated without commas
        return np.array2string(arr, separator=' ')

    def fmt_weight(arr: np.ndarray) -> str:
        # Match weight style: comma-separated with spaces and prevent line wraps
        return np.array2string(arr, separator=', ', max_line_width=np.inf)

    for _ in range(epochs):
        # shuffle samples each epoch
        indices = np.arange(len(samples))
        rng.shuffle(indices)
        abs_err_sum = 0.0
        for idx in indices:
            f_prev, f_curr, user_rew = samples[idx]
            delta_f = (f_curr - f_prev).astype(float)
            if clamp_first_two_nonpositive and delta_f.shape[0] >= 2:
                delta_f[0] = min(delta_f[0], 0.0)
                delta_f[1] = min(delta_f[1], 0.0)
            projected = float(np.dot(w, delta_f))
            error = float(user_rew - projected)

            # Log before updating to mimic online log semantics
            if log_fh is not None:
                log_fh.write(f"[FEATURE] f_prev: {fmt_feat(f_prev)}, f_curr: {fmt_feat(f_curr)}\n")
                log_fh.write(f"[REWARD] Projected Reward: {projected}, user_rew: {user_rew}, error: {error}\n")
                log_fh.write(f"[WEIGHT] self.w: {fmt_weight(w)}\n")
                log_fh.write("============\n")

            w += alpha * error * delta_f
            abs_err_sum += abs(error)
        epoch_abs_error_sums.append(abs_err_sum)

    if log_fh is not None:
        log_fh.close()

    return w, epoch_abs_error_sums


def maybe_evaluate(w: np.ndarray, num_episodes: int = 50, max_steps: int = 100) -> float:
    """Optional quick evaluation in Robotaxi env with current weights.

    This reproduces the simulate-then-choose policy used in other scripts.
    """
    # Lazy imports to avoid hard deps if evaluation is not requested
    from robotaxi.gameplay.wrappers import make_gymnasium_environment, preprocess_observation_tamer
    from robotaxi.agent.tamer_agent import TAMERAgent
    from robotaxi.gameplay.entities import ALL_SNAKE_ACTIONS
    import tqdm
    import random

    def simulate_transition(state, action):
        return TAMERAgent.simulate_transition(state, action)

    def get_feature_vector_tamer(state):
        return preprocess_observation_tamer(state)

    def choose_action(s_t, w_vec, env):
        f_t = get_feature_vector_tamer(s_t)
        best_a = None
        max_rew = -float('inf')
        for a in ALL_SNAKE_ACTIONS:
            s_next = simulate_transition(s_t, a)
            f_next = get_feature_vector_tamer(s_next)
            delta_f = f_next - f_t
            projected_rew = float(np.dot(w_vec, delta_f))
            if projected_rew > max_rew:
                max_rew = projected_rew
                best_a = a
        return best_a

    rng = np.random.default_rng(0)
    random.seed(0)
    env = make_gymnasium_environment("./robotaxi/levels/8x8-blank.json")()

    total_reward = 0.0
    for _ in tqdm.tqdm(range(num_episodes)):
        s_t, _ = env.reset()
        for t in range(max_steps):
            a_t = choose_action(s_t, w, env)
            s_next, reward, done, truncated, _ = env.step(a_t)
            total_reward += reward
            s_t = s_next
            if done or truncated or t >= max_steps - 1:
                break
    env.close()
    return total_reward / float(num_episodes)


def main():
    parser = argparse.ArgumentParser(description="Offline TAMER training from [FEATURE]/[REWARD] logs")
    parser.add_argument("log_paths", nargs='+', type=str, help="One or more log files with [FEATURE]/[REWARD]/[WEIGHT] lines")
    parser.add_argument("--alpha", type=float, default=0.01, help="Learning rate alpha")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (passes over the log)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for init and shuffling")
    parser.add_argument("--init-w", choices=["w1", "w2", "w10", "w20", "random", "log"], default="w1", help="How to initialize weights: preset w1/w2, random, or from first [WEIGHT] in logs")
    parser.add_argument("--save", type=str, default=None, help="Path to save final weights .npy; defaults to weights_<timestamp>.npy")
    parser.add_argument("--out-log", type=str, default=None, help="Path to write the offline training log mirroring the input log format")
    parser.add_argument("--eval", action="store_true", help="Run a quick evaluation after training")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of eval episodes if --eval is set")
    parser.add_argument("--eval-max-steps", type=int, default=100, help="Max steps per episode during eval")
    args = parser.parse_args()

    # Preset weight map
    w_map = {
        "w1": np.array([ 0.09894706, -0.01005191, -0.05182143, -0.02420872,  0.03493194, -0.08516584]),
        "w2": np.array([ 0.07768216, -0.09317696, -0.05280239,  0.03182322, -0.02639944, 0.09958436]),
        "w10": np.array([ 0.9894706, -0.1005191, -0.5182143, -0.2420872,  0.3493194, -0.8516584]),
        "w20": np.array([ 0.7768216, -0.9317696, -0.5280239,  0.3182322, -0.2639944, 0.9958436]),
    }

    # Parse all inputs in order
    all_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    first_weight_in_logs: Optional[np.ndarray] = None
    total_parsed = 0
    for path in args.log_paths:
        s, first_w = parse_feature_reward_log(path)
        all_samples.extend(s)
        total_parsed += len(s)
        if first_weight_in_logs is None and first_w is not None:
            first_weight_in_logs = first_w

    if not all_samples:
        print(f"No (f_prev, f_curr, user_rew) samples parsed from inputs: {args.log_paths}")
        sys.exit(1)

    print(f"Parsed {total_parsed} samples from {len(args.log_paths)} file(s)")

    # Resolve initialization
    init_w: Optional[np.ndarray]
    if args.init_w in ("w1", "w2"):
        init_w = w_map[args.init_w]
        print(f"Initializing w from preset {args.init_w}: {init_w}")
    elif args.init_w == "log":
        if first_weight_in_logs is not None:
            init_w = first_weight_in_logs
            print(f"Initializing w from first [WEIGHT] found in logs: {init_w}")
        else:
            init_w = w_map["w1"]
            print("--init-w log specified but no [WEIGHT] found; falling back to w1 preset")
    else:
        init_w = None  # random handled inside trainer

    # Resolve output log path default
    out_log_path = args.out_log
    if out_log_path is None:
        base = os.path.splitext(os.path.basename(args.log_paths[0]))[0]
        out_log_path = os.path.join(os.path.dirname(args.log_paths[0]), f"offline-{base}.log")

    final_w, epoch_abs_err_sums = offline_train_from_features(
        samples=all_samples,
        alpha=args.alpha,
        epochs=args.epochs,
        seed=args.seed,
        init_w=init_w,
        clamp_first_two_nonpositive=True,
        out_log_path=out_log_path,
    )

    print("Training complete.")
    for i, val in enumerate(epoch_abs_err_sums):
        print(f"Epoch {i+1} sum |error| = {val:.6f}")
    print(f"Final weights: {final_w}")

    save_path = args.save
    if not save_path:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        base = os.path.splitext(os.path.basename(args.log_paths[0]))[0]
        save_path = f"tamer_weights_offline_{base}_{ts}.npy"
    np.save(save_path, final_w)
    print(f"Saved weights to {save_path}")

    print(f"Offline training log written to {out_log_path}")

    if args.eval:
        avg_rew = maybe_evaluate(final_w, num_episodes=args.eval_episodes, max_steps=args.eval_max_steps)
        print(f"EVALUATION: Average reward over {args.eval_episodes} episodes: {avg_rew}")


if __name__ == "__main__":
    main()



