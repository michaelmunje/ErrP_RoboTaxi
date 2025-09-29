import argparse
import csv
import os
import re


def state_block_to_row_strings(block: str) -> str:
    """Convert a [[...], [...], ...] 8x8 block of ints into
    ["66666666","60300006",...,"66666666"] string list literal.
    """
    # Pull out inner rows: sequences between [ and ] that contain digits and spaces
    # Example row: [6 0 3 0 0 0 0 6] -> "60300006"
    row_re = re.compile(r"\[\s*([0-9\s]+)\s*\]")
    rows = []
    for m in row_re.finditer(block):
        digits = m.group(1).split()
        rows.append(''.join(digits))
    # Format as JSON-like array of quoted strings
    return "[" + ",".join(f'"{r}"' for r in rows) + "]"


def reformat_csv_inplace(path: str) -> None:
    tmp_path = path + ".tmp"
    with open(path, "r", newline="") as inf, open(tmp_path, "w", newline="") as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)
        header = next(reader)
        writer.writerow(header)
        # columns: exp_id, step, pre_state, state, gt, gt_noisy
        idx_pre = header.index("pre_state")
        idx_state = header.index("state")
        for row in reader:
            pre_block = row[idx_pre]
            # Ensure the block-like content exists before converting
            if pre_block.startswith("[[") and pre_block.endswith("]]"):
                row[idx_pre] = state_block_to_row_strings(pre_block)
            state_block = row[idx_state]
            if state_block.startswith("[[") and state_block.endswith("]]"):
                row[idx_state] = state_block_to_row_strings(state_block)
            writer.writerow(row)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Reformat pre_state and state columns in aggregated CSV")
    parser.add_argument("csv_path", help="Path to aggregated CSV")
    args = parser.parse_args()
    reformat_csv_inplace(args.csv_path)
    print(f"Reformatted pre_state in {args.csv_path}")


if __name__ == "__main__":
    main()


