#!/usr/bin/env python3

import argparse
import csv
import os
import re
import json
from typing import List

STATE_LINE_PATTERN = re.compile(r"^\d{8}$")


def parse_max_step_limit(first_line: str) -> int:
    """Extract the integer max step limit from the first line 'max_step_limit:<n>'."""
    if ":" not in first_line:
        raise ValueError("First line does not contain ':' separating key and value")
    key, value = first_line.split(":", 1)
    key = key.strip()
    if key != "max_step_limit":
        raise ValueError("First line must start with 'max_step_limit'")
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"Could not parse integer from '{value}'") from exc


def is_state_line(s: str) -> bool:
    """Return True if the line looks like a state row: exactly 8 digits."""
    return bool(STATE_LINE_PATTERN.match(s.strip()))


def extract_state_matrices(lines: List[str], needed_states: int) -> List[List[str]]:
    """Extract the first `needed_states` state matrices.

    A state matrix is 8 consecutive lines, each exactly 8 digits.
    Scan top-to-bottom and collect matrices until `needed_states` or EOF.
    """
    matrices: List[List[str]] = []
    i = 0
    total_lines = len(lines)

    while i <= total_lines - 8 and len(matrices) < needed_states:
        candidate_block = lines[i : i + 8]
        if all(is_state_line(line) for line in candidate_block):
            matrices.append([line.strip() for line in candidate_block])
            i += 8
            continue
        i += 1

    return matrices


def serialize_matrix(matrix: List[str], mode: str) -> str:
    """Serialize an 8x8 state matrix into a JSON string for a single CSV cell.

    Modes:
      - 'list-rows' (default): JSON list of 8 strings, each an 8-char row
      - 'matrix-digits': JSON 2D list of 8x8 integers
      - 'list-digits': JSON flat list of 64 integers
    """
    if mode == "matrix-digits":
        data = [[int(ch) for ch in row] for row in matrix]
    elif mode == "list-digits":
        data = [int(ch) for row in matrix for ch in row]
    else:  # 'list-rows'
        data = list(matrix)
    return json.dumps(data, separators=(",", ":"))


def write_states_single_column_csv(
    matrices: List[List[str]], output_path: str, header: str, mode: str
) -> None:
    """Write each matrix as a single JSON string cell in a 1-column CSV with header."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header])
        for matrix in matrices:
            writer.writerow([serialize_matrix(matrix, mode)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse log file and export states to CSV (one matrix per cell).")
    parser.add_argument("input", help="Path to input log file (e.g., autocar_..._offline.log)")
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output CSV. Defaults to <input_basename>_states.csv in same directory.",
    )
    parser.add_argument(
        "--cell-format",
        choices=["list-rows", "matrix-digits", "list-digits"],
        default="list-rows",
        help="How to encode each 8x8 matrix in the single cell (JSON). Default: list-rows",
    )
    parser.add_argument(
        "--header",
        default="state",
        help="CSV column header name. Default: state",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = [line.rstrip("\n") for line in f]

    if not all_lines:
        raise ValueError("Input file is empty")

    # Parse max_step_limit from the first line
    max_steps = parse_max_step_limit(all_lines[0])

    # Collect up to (n+2) to allow dropping the (n+2)th, resulting in (n+1)
    target_with_extra = max_steps + 2

    # Extract from the remainder of the file (after the first line)
    content_lines = all_lines[1:]

    matrices = extract_state_matrices(content_lines, needed_states=target_with_extra)

    # Keep at most (n+1)
    matrices = matrices[: max_steps + 1]

    if not matrices:
        raise ValueError("No state matrices found in the input file")

    output_path = args.output
    if not output_path:
        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_states.csv")

    write_states_single_column_csv(matrices, output_path, header=args.header, mode=args.cell_format)

    print(
        f"Parsed max_step_limit={max_steps}. "
        f"Exported {len(matrices)} state matrices to: {output_path} "
        f"(format={args.cell_format}, header={args.header})"
    )


if __name__ == "__main__":
    main()
