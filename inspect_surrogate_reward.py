import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def split_top_level_commas(s: str) -> List[str]:
    """Split a string by commas that are not inside square brackets.

    This is tailored for lines shaped like:
    [[...]], [[...]], prev_action, reward_gt, reward_noisy, reward_surrogate, [weights]
    """
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in s:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts


def parse_rewards(line: str, debug: bool = False) -> Tuple[int, float]:
    """Parse a log line to extract (reward_gt, reward_surrogate).

    Supports rows with or without the trailing 'explored' flag, e.g.:
    [[...]], [[...]], prev_action, reward_gt, reward_noisy, reward_surrogate, [weights], explored
    or
    [[...]], [[...]], prev_action, reward_gt, reward_noisy, reward_surrogate, [weights]

    Returns:
        (reward_gt, reward_surrogate)
    Raises ValueError if parsing fails.
    """
    # Keep only the portion after the ':' label
    if ':' not in line:
        raise ValueError('No colon in line')
    after = line.split(':', 1)[1].strip()

    tokens = [t for t in split_top_level_commas(after) if t != '']
    n = len(tokens)
    # Expect: [big1], [big2], prev_action, reward_gt, reward_noisy, reward_surrogate, [weights] [, explored]
    if n < 7:
        raise ValueError(f'Unexpected token count: {n} in line: {line[:160]}')

    # Determine indices relative to the tail to be robust to optional 'explored'
    # For 8 tokens (with explored): gt at -5, surrogate at -3
    # For 7 tokens (no explored): gt at -4, surrogate at -2
    reward_gt_idx = -5 if n >= 8 else -4
    reward_surrogate_idx = -3 if n >= 8 else -2

    reward_gt_str = tokens[reward_gt_idx].strip()
    reward_surrogate_str = tokens[reward_surrogate_idx].strip()

    if debug:
        # Print a concise debug summary per line
        preview = after[:200].replace('\n', ' ')
        print(f"DEBUG line tokens={n} gt_idx={reward_gt_idx} sur_idx={reward_surrogate_idx} | preview={preview}")

    # Convert to numeric
    reward_gt = int(float(reward_gt_str))  # covers cases like "0" or "0.0"
    reward_surrogate = float(reward_surrogate_str)
    return reward_gt, reward_surrogate


def load_points(log_path: Path, debug: bool = False) -> Tuple[List[int], List[float], List[int], List[float]]:
    """Load the log file and return x,y lists for gt==0 (grey) and gt==-1 (red).

    X is the row index of data lines after the delimiter line of '============'.
    """
    lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    # Find the delimiter line index
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '============':
            start_idx = i
            break
    if start_idx is None:
        raise RuntimeError("Delimiter line '============' not found in log")

    x_grey: List[int] = []
    y_grey: List[float] = []
    x_red: List[int] = []
    y_red: List[float] = []

    data_idx = 0
    ignore_first_n = 50
    for line in lines[start_idx + 1:]:
        if not line.strip():
            continue
        try:
            reward_gt, reward_surrogate = parse_rewards(line, debug=debug)
        except Exception:
            # Not a data line; skip
            if debug:
                preview = line.strip()[:200].replace('\n', ' ')
                print(f"DEBUG skip unparsable | preview={preview}")
            continue

        # Ignore the first N datapoints (counted only over valid data lines)
        if data_idx >= ignore_first_n:
            if reward_gt == 0:
                x_grey.append(data_idx)
                y_grey.append(reward_surrogate)
            elif reward_gt == -1:
                x_red.append(data_idx)
                y_red.append(reward_surrogate)
            if debug:
                print(f"DEBUG idx={data_idx} gt={reward_gt} surrogate={reward_surrogate}")

        data_idx += 1

    return x_grey, y_grey, x_red, y_red


def main():
    default_log = Path('logs/2025-10-15-18-31-31-online-tamer-noisy_v2.log')

    # Simple flag parsing: allow either order of [--debug] and [log_path]
    debug = False
    path_arg: Path = None
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            debug = True
        else:
            path_arg = Path(arg)
    log_path = path_arg if path_arg is not None else default_log
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    x_grey, y_grey, x_red, y_red = load_points(log_path, debug=debug)

    plt.figure(figsize=(12, 5))
    if x_grey:
        plt.scatter(x_grey, y_grey, c='grey', s=10, alpha=0.7, label='reward_gt = 0')
    if x_red:
        plt.scatter(x_red, y_red, c='red', s=10, alpha=0.7, label='reward_gt = -1')

    # Plot running average curves per class starting at row 50
    start_row = 50
    if x_grey or x_red:
        max_row = max([max(x_grey) if x_grey else 0, max(x_red) if x_red else 0])
        grey_map = {x: y for x, y in zip(x_grey, y_grey)}
        red_map = {x: y for x, y in zip(x_red, y_red)}

        grey_sum = 0.0
        grey_count = 0
        red_sum = 0.0
        red_count = 0
        avg_x_grey = []
        avg_y_grey = []
        avg_x_red = []
        avg_y_red = []
        for i in range(max_row + 1):
            if i in grey_map:
                grey_sum += grey_map[i]
                grey_count += 1
            if i in red_map:
                red_sum += red_map[i]
                red_count += 1
            if i >= start_row:
                if grey_count > 0:
                    avg_x_grey.append(i)
                    avg_y_grey.append(grey_sum / grey_count)
                if red_count > 0:
                    avg_x_red.append(i)
                    avg_y_red.append(red_sum / red_count)

        if avg_x_grey:
            plt.plot(avg_x_grey, avg_y_grey, color='grey', linewidth=1.5, label='avg (gt=0)')
        if avg_x_red:
            plt.plot(avg_x_red, avg_y_red, color='red', linewidth=1.5, label='avg (gt=-1)')

    plt.xlabel('Row index after ============')
    plt.ylabel('Surrogate reward')
    plt.title('Surrogate reward over time')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    out_path = log_path.with_name(f"{log_path.stem}-surrogate-scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(str(out_path))


if __name__ == '__main__':
    main()


