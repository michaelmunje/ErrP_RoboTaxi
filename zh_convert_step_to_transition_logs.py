import argparse
import os
import re
from typing import List, Tuple


StepSample = Tuple[str, int, int]  # (state_str, action, reward_gt)


def parse_step_lines(path: str) -> List[StepSample]:
    """Parse lines of the form:
    [state, action, reward_gt] : [[...],[...],...], <action>, <reward>
    Returns list of (state_str, action, reward_gt) in original order.
    """
    samples: List[StepSample] = []
    # Capture a 2D bracketed array like [[...]] (no commas inside), then two ints
    pattern = re.compile(
        r"\[state,\s*action,\s*reward_gt\]\s*:\s*(\[\[.*?\]\])\s*,\s*(-?\d+)\s*,\s*(-?\d+)",
        re.IGNORECASE,
    )
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.search(line)
            if not m:
                continue
            state_str = m.group(1)
            action = int(m.group(2))
            reward_gt = int(m.group(3))
            samples.append((state_str, action, reward_gt))
    return samples


def write_transition_log(out_path: str, samples: List[StepSample]) -> int:
    """Write lines in the form:
    [previous_state, current_state, previous_action, reward_gt, reward_noisy] : <prev_state>, <curr_state>, <prev_action>, <rew>, <rew>
    Returns number of transitions written.
    """
    if len(samples) < 2:
        with open(out_path, "w") as f:
            pass
        return 0

    count = 0
    with open(out_path, "w") as f:
        for i in range(1, len(samples)):
            prev_state, prev_action, _ = samples[i - 1]
            curr_state, _, reward_gt = samples[i]
            reward_noisy = reward_gt
            f.write(
                "[previous_state, current_state, previous_action, reward_gt, reward_noisy] : "
            )
            f.write(f"{prev_state}, {curr_state}, {prev_action}, {reward_gt}, {reward_noisy}\n")
            count += 1
    return count


def convert_file(path: str, suffix: str = "_conti.log") -> str:
    samples = parse_step_lines(path)
    base, ext = os.path.splitext(path)
    out_path = f"{base}{suffix}" if suffix.startswith("_") else os.path.join(
        os.path.dirname(path), f"{os.path.basename(base)}_{suffix}"
    )
    n = write_transition_log(out_path, samples)
    print(f"Converted {path} -> {out_path} with {n} transitions")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert step-level logs to transition-level logs")
    parser.add_argument("inputs", nargs="+", help="Input log files to convert")
    parser.add_argument("--suffix", default="_conti.log", help="Output suffix (default: _conti.log)")
    args = parser.parse_args()

    for p in args.inputs:
        convert_file(p, suffix=args.suffix)


if __name__ == "__main__":
    main()



