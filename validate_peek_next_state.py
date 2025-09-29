import argparse
import csv
import ast
from typing import List, Tuple

import numpy as np

from robotaxi.agent.tamer_agent import TAMERAgent


def parse_row_string_list(s: str) -> List[str]:
    """Parse a string like ["66666666","61000006",...] into a list of 8 strings."""
    try:
        rows = ast.literal_eval(s)
        assert isinstance(rows, list)
        return rows
    except Exception as exc:
        raise ValueError(f"Failed to parse row-string list: {s}") from exc


def rows_to_grid(rows: List[str]) -> np.ndarray:
    """Convert list of 8 row-strings to 8x8 numpy int array."""
    assert len(rows) == 8, f"Expected 8 rows, got {len(rows)}"
    grid = np.zeros((8, 8), dtype=int)
    for i, row_str in enumerate(rows):
        assert len(row_str) == 8, f"Row {i} length != 8: {row_str}"
        grid[i, :] = [int(ch) for ch in row_str]
    return grid


def find_unique_pos(grid: np.ndarray, value: int) -> Tuple[int, int]:
    xs, ys = np.where(grid == value)
    if len(xs) != 1:
        raise ValueError(f"Expected exactly one {value} in grid, found {len(xs)}")
    return int(xs[0]), int(ys[0])


def validate(csv_path: str, limit: int = 0) -> None:
    total = 0
    valid = 0
    errors = 0
    mismatches: List[Tuple[str, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []

    with open(csv_path, "r") as fh:
        reader = csv.DictReader(fh)
        required_cols = {"exp_id", "step", "pre_state", "state", "prev_action"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        for row in reader:
            try:
                exp_id = row["exp_id"]
                step = int(row["step"]) if row["step"].strip() else -1
                prev_action = int(row["prev_action"]) if row["prev_action"].strip() else 0

                pre_rows = parse_row_string_list(row["pre_state"])  # list of 8 strings
                tgt_rows = parse_row_string_list(row["state"])      # list of 8 strings
                pre_grid = rows_to_grid(pre_rows)
                tgt_grid = rows_to_grid(tgt_rows)

                # Run peek_next_state
                pred_grid = TAMERAgent.peek_next_state(pre_grid, prev_action)

                # Extract head/body positions
                tgt_head = find_unique_pos(tgt_grid, 4)
                tgt_body = find_unique_pos(tgt_grid, 5)
                pred_head = find_unique_pos(pred_grid, 4)
                pred_body = find_unique_pos(pred_grid, 5)

                if pred_head == tgt_head and pred_body == tgt_body:
                    valid += 1
                else:
                    mismatches.append((exp_id, step, tgt_head, tgt_body, pred_head, pred_body))
                total += 1

                if limit and total >= limit:
                    break

            except Exception:
                errors += 1
                total += 1
                if limit and total >= limit:
                    break

    acc = (valid / total) if total else 0.0
    print(f"Evaluated: {total}, Valid: {valid}, Errors: {errors}, Accuracy: {acc:.4f}")
    if mismatches:
        print("Sample mismatches (up to 10):")
        for m in mismatches[:10]:
            exp_id, step, tgt_h, tgt_b, pred_h, pred_b = m
            print(f"- {exp_id} step {step}: tgt head/body {tgt_h}/{tgt_b} vs pred {pred_h}/{pred_b}")


def main():
    parser = argparse.ArgumentParser(description="Validate TAMERAgent.peek_next_state against aggregated CSV")
    parser.add_argument("--csv", required=True, help="Path to aggregated_states_v2 CSV")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of rows")
    args = parser.parse_args()
    validate(args.csv, args.limit)


if __name__ == "__main__":
    main()


