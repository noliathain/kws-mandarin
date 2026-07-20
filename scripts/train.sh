#!/usr/bin/env bash
# Launch DDP training. Designed to run inside tmux so it survives disconnects.
# Logs + checkpoints go to persistent lightning_storage, not the session scratchpad.
#
#   tmux new-session -d -s kws "bash scripts/train.sh 2 configs/base_2gpu.yaml"
#
# Args: $1 = nproc_per_node (GPUs), $2 = config path.
set -o pipefail
cd "$(dirname "$0")/.."

NPROC="${1:-2}"
CONFIG="${2:-configs/base_2gpu.yaml}"
LOGDIR=/teamspace/lightning_storage/kws-mandarin/logs
mkdir -p "$LOGDIR"

echo ">> training start $(date -u +%FT%TZ) nproc=$NPROC config=$CONFIG" | tee -a "$LOGDIR/train.log"
# No --no-resume: resumes from latest.pt if present, else starts fresh.
uv run --extra train torchrun --standalone --nproc_per_node="$NPROC" \
  -m kws_mandarin.train --config "$CONFIG" 2>&1 | tee -a "$LOGDIR/train.log"
echo ">> training exit $? $(date -u +%FT%TZ)" | tee -a "$LOGDIR/train.log"
