import argparse
import csv
import os
import re
from typing import Iterable, List, Tuple


def grid_to_row_string_list(grid_str: str) -> str:
    """Convert a string like
    [[6 6 6 6 6 6 6 6], [6 1 0 0 0 0 0 6], ...]
    into
    ["66666666","61000006",...]

    Returns a JSON-like string suitable for CSV storage; csv.writer will
    handle escaping of quotes.
    """
    # Extract each bracketed row content
    rows = re.findall(r"\[([^\]]+)\]", grid_str)
    row_strings: List[str] = []
    for row in rows:
        nums = re.findall(r"-?\d+", row)
        if not nums:
            continue
        row_strings.append("".join(nums))
    return "[\"" + "\",\"".join(row_strings) + "\"]"


def iter_transition_lines(path: str) -> Iterable[Tuple[str, str, int, int, int]]:
    """Yield (pre_state_str, state_str, prev_action, gt, gt_noisy) from a transition log file.

    Lines are expected like:
      [previous_state, current_state, previous_action, reward_gt, reward_noisy] : <prev>, <curr>, <prev_action>, <gt>, <noisy>

    The file may start with optional metadata and a '====' delimiter. We skip
    any non-matching lines.
    """
    # Non-greedy capture of DOUBLE-bracketed arrays [[...]] twice, then 3 ints
    line_re = re.compile(
        r"^\[previous_state,\s*current_state,\s*previous_action,\s*reward_gt,\s*reward_noisy\]\s*:\s*"
        r"(\[\[.*?\]\])\s*,\s*(\[\[.*?\]\])\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$"
    )

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = line_re.match(line)
            if not m:
                continue
            pre_state = grid_to_row_string_list(m.group(1))
            state = grid_to_row_string_list(m.group(2))
            prev_action = int(m.group(3))
            gt = int(m.group(4))
            gt_noisy = int(m.group(5))
            yield pre_state, state, prev_action, gt, gt_noisy


def write_aggregated_csv(out_csv: str, groups: List[Tuple[str, List[str]]]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["exp_id", "step", "pre_state", "state", "prev_action", "gt", "gt_noisy"])

        for exp_id, files in groups:
            step = 0
            for path in files:
                for pre_state, state, prev_action, gt, gt_noisy in iter_transition_lines(path):
                    writer.writerow([exp_id, step, pre_state, state, prev_action, gt, gt_noisy])
                    step += 1
            # step resets automatically per exp_id loop


def parse_groups(exp_args: List[str]) -> List[Tuple[str, List[str]]]:
    """Parse repeated --exp arguments of the form:
        exp_id:path1,path2,path3
    Returns list of (exp_id, [paths...]).
    """
    groups: List[Tuple[str, List[str]]] = []
    for spec in exp_args:
        if ":" not in spec:
            raise ValueError(f"Invalid --exp format: {spec}")
        exp_id, files_str = spec.split(":", 1)
        files = [p.strip() for p in files_str.split(",") if p.strip()]
        groups.append((exp_id, files))
    return groups


def main():
    parser = argparse.ArgumentParser(description="Aggregate transition logs into a CSV")
    parser.add_argument("--exp", action="append", required=True,
                        help="Group spec: exp_id:path1,path2,... (repeatable)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    groups = parse_groups(args.exp)
    write_aggregated_csv(args.out, groups)
    print(f"Wrote aggregated CSV to {args.out}")


if __name__ == "__main__":
    main()



