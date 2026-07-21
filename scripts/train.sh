#!/usr/bin/env bash
# Launch DDP training. Designed to run inside tmux so it survives disconnects.
# Logs + checkpoints go to persistent lightning_storage, not the session scratchpad.
#
#   tmux new-session -d -s kws -c "$PWD" "bash scripts/train.sh 2 configs/base_2gpu.yaml --fresh"
#
# NOTE the -c "$PWD": without it tmux inherits whatever directory you last cd'd to and
# cannot find this script.
#
# Args: $1 = nproc_per_node (GPUs), $2 = config path, $3 = --fresh to start from scratch.
#
# Without --fresh this RESUMES from latest.pt if one exists. With --fresh, the existing
# checkpoint dir and log are archived under a timestamp and training starts at step 0 --
# nothing is deleted, so a previous run is never silently destroyed.
set -o pipefail
cd "$(dirname "$0")/.."

NPROC="${1:-2}"
CONFIG="${2:-configs/base_2gpu.yaml}"
FRESH="${3:-}"
LOGDIR=/teamspace/lightning_storage/kws-mandarin/logs
mkdir -p "$LOGDIR"

STAMP=$(date -u +%Y%m%d-%H%M%S)
# One log per run. An earlier version appended to a shared train.log and rotated it with mv
# on --fresh; on this storage mount the freshly created replacement became unreadable while
# tee still held it open, leaving a live run with no tailable log. A per-run file never
# renames anything that is open, and a stale "exit 1" can no longer be mistaken for this run.
LOG="$LOGDIR/train_${STAMP}.log"

RESUME_ARG=""
if [ "$FRESH" = "--fresh" ]; then
    CKPT=$(uv run python -c "
from kws_mandarin.config import TrainConfig
print(TrainConfig.from_yaml('$CONFIG').ckpt_dir)" 2>/dev/null | tail -1)
    if [ -n "$CKPT" ] && [ -d "$CKPT" ]; then
        mv "$CKPT" "${CKPT}_pre_${STAMP}"
        echo ">> archived previous checkpoints -> ${CKPT}_pre_${STAMP}" | tee -a "$LOG"
    fi
    RESUME_ARG="--no-resume"
fi

echo ">> logging to $LOG"
echo ">> training start $(date -u +%FT%TZ) nproc=$NPROC config=$CONFIG ${RESUME_ARG:-(resume)}" \
    | tee -a "$LOG"
uv run --extra train torchrun --standalone --nproc_per_node="$NPROC" \
  -m kws_mandarin.train --config "$CONFIG" $RESUME_ARG 2>&1 | tee -a "$LOG"
echo ">> training exit $? $(date -u +%FT%TZ)" | tee -a "$LOG"
