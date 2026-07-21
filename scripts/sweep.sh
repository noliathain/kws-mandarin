#!/usr/bin/env bash
# Run a capacity sweep: one full training run per config, sequentially (2 GPUs are shared).
#
#   tmux new-session -d -s sweep -c "$PWD" "bash scripts/sweep.sh configs/scale6.yaml ..."
#
# Each run is --fresh, so each gets its own checkpoints and its own timestamped log. The log
# records `config=<path>`, which is how the plotting step attributes results to a scale.
set -o pipefail
cd "$(dirname "$0")/.."
for cfg in "$@"; do
    echo ">>>> sweep run: $cfg  ($(date -u +%FT%TZ))"
    bash scripts/train.sh 2 "$cfg" --fresh
done
echo ">>>> sweep complete ($(date -u +%FT%TZ))"
