#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re
from typing import List, Tuple

TIMESTAMP_RE = re.compile(r".*?(\d{8}-\d{6})_offline_states\.csv$")


def extract_exp_id(filepath: str) -> str:
    """Extract the timestamp exp_id (e.g., 20250822-113047) from the states CSV filename."""
    name = os.path.basename(filepath)
    m = TIMESTAMP_RE.match(name)
    if not m:
        raise ValueError(f"Could not extract exp_id from filename: {name}")
    return m.group(1)


def read_states_csv(path: str) -> List[str]:
    """Read a states CSV produced by parse_log_states_to_csv.py and return list of 201 state strings."""
    states: List[str] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            # Single-column CSV
            states.append(row[0])
    if len(states) < 2:
        raise ValueError(f"Expected at least 2 states in {path}, got {len(states)}")
    return states


def read_gt_csv(path: str) -> List[int]:
    """Read keypress_gt.csv, returns list of integers (after header)."""
    gts: List[int] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            try:
                gts.append(int(row[0]))
            except ValueError:
                # Allow stray whitespace/empty strings
                value = row[0].strip()
                if value == "":
                    continue
                gts.append(int(value))
    return gts


def aggregate(states_files: List[str], gt_values: List[int]) -> List[Tuple[str, int, str, str, int]]:
    """Aggregate rows as (exp_id, step, pre_state, state, gt)."

    For each states file (sorted by exp_id), use 201 states to form 200 transitions,
    paired with 200 GT values. Total GT values must equal 200 * num_trials.
    """
    # Sort by exp_id extracted from filename to match chronological order
    sorted_files = sorted(states_files, key=lambda p: extract_exp_id(p))

    num_trials = len(sorted_files)
    if num_trials == 0:
        raise ValueError("No state CSV files provided")

    expected_gt = 200 * num_trials
    if len(gt_values) < expected_gt:
        raise ValueError(
            f"Not enough GT values: expected at least {expected_gt}, got {len(gt_values)}"
        )

    rows: List[Tuple[str, int, str, str, int]] = []
    gt_index = 0

    for states_path in sorted_files:
        exp_id = extract_exp_id(states_path)
        states = read_states_csv(states_path)
        if len(states) < 201:
            raise ValueError(f"Expected 201 states in {states_path}, got {len(states)}")

        # 200 transitions: step 0..199 -> (s[i], s[i+1]) with GT[i]
        for step in range(200):
            pre_state = states[step]
            state = states[step + 1]
            gt = gt_values[gt_index]
            gt_index += 1
            rows.append((exp_id, step, pre_state, state, gt))

    # Ignore any extra GT values beyond expected
    return rows


def write_aggregated_csv(rows: List[Tuple[str, int, str, str, int]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exp_id", "step", "pre_state", "state", "gt"])
        for exp_id, step, pre_state, state, gt in rows:
            writer.writerow([exp_id, step, pre_state, state, gt])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 4 states CSVs and a keypress_gt.csv into one CSV.")
    parser.add_argument(
        "--states-glob",
        default="",
        help="Glob for states CSV files (e.g., '/path/*_offline_states.csv'). If empty, uses subject6 default.",
    )
    parser.add_argument(
        "--gt",
        default="",
        help="Path to keypress_gt.csv. If empty, uses subject6 default.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. If empty, writes next to GT as 'aggregated_states_gt.csv'.",
    )

    args = parser.parse_args()

    if args.states_glob:
        states_files = glob.glob(args.states_glob)
    else:
        states_files = glob.glob(
            "/home/zhihan/Documents/Code/ErrP_RoboTaxi/phase2-selected/subject6/additional_log_files/*_offline_states.csv"
        )

    if args.gt:
        gt_path = args.gt
    else:
        gt_path = "/home/zhihan/Documents/Code/ErrP_RoboTaxi/phase2-selected/subject6/keypress_gt.csv"

    if not states_files:
        raise FileNotFoundError("No states CSV files found. Check --states-glob or default path.")

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT CSV not found: {gt_path}")

    gt_values = read_gt_csv(gt_path)
    rows = aggregate(states_files, gt_values)

    if args.output:
        output_path = args.output
    else:
        base_dir = os.path.dirname(gt_path)
        output_path = os.path.join(base_dir, "aggregated_states_gt.csv")

    write_aggregated_csv(rows, output_path)

    print(
        f"Aggregated {len(states_files)} trials x 200 steps = {len(rows)} rows -> {output_path}"
    )


if __name__ == "__main__":
    main()




