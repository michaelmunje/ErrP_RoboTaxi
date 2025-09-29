#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <path/to/online-tamer-noisy.log>"
  exit 1
fi

LOG_PATH="$1"

FEATURE_VERSION="${2:-v2}"

if [ ! -f "$LOG_PATH" ]; then
  echo "Error: log file not found: $LOG_PATH"
  exit 1
fi

# Initialize conda in non-interactive shell and activate ErrP2
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  eval "$("$(command -v conda)" shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Error: conda initialization not found. Run 'conda init bash' once or ensure conda is installed."
  exit 1
fi

conda activate ErrP2 || { echo "Error: failed to activate 'ErrP2'"; exit 1; }

PROJECT_ROOT="/home/zhihan/Documents/Code/ErrP_RoboTaxi"

CUM_LOG="${LOG_PATH%.log}-cumulative-reward.log"

rm -f "$CUM_LOG"
echo "[1/2] Running visualization with: $LOG_PATH"
python "$PROJECT_ROOT/zh_visualize_tamer_online_training.py" "$LOG_PATH" "$FEATURE_VERSION"

# Construct cumulative reward log path by replacing .log with -cumulative-reward.log

echo "[2/2] Running analysis on: $CUM_LOG"
python "$PROJECT_ROOT/analysis/analysis_weight_learning_progress.py" "$CUM_LOG" "$FEATURE_VERSION"

echo "Done."


