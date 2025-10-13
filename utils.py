from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray




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


def parse_grid(row_list: Sequence[str]) -> NDArray[np.int_]:
    
    

        grid = np.zeros((8, 8), dtype=int)
        if len(row_list) != 8:
            raise ValueError(f"Expected 8 rows, got {len(row_list)}")
        for i, row_str in enumerate(row_list):
            if len(row_str) != 8:
                raise ValueError(f"Row {i} length != 8: {row_str}")
            grid[i, :] = np.fromiter((int(ch) for ch in row_str), dtype=int, count=8)
        return grid