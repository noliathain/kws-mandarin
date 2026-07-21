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

RESUME_ARG=""
if [ "$FRESH" = "--fresh" ]; then
    STAMP=$(date -u +%Y%m%d-%H%M%S)
    CKPT=$(uv run python -c "
from kws_mandarin.config import TrainConfig
print(TrainConfig.from_yaml('$CONFIG').ckpt_dir)" 2>/dev/null | tail -1)
    if [ -n "$CKPT" ] && [ -d "$CKPT" ]; then
        mv "$CKPT" "${CKPT}_pre_${STAMP}"
        echo ">> archived previous checkpoints -> ${CKPT}_pre_${STAMP}"
    fi
    # Rotate the log too: it is appended across runs, so a stale "exit 1" from an earlier
    # run otherwise reads as a failure of the current one.
    [ -f "$LOGDIR/train.log" ] && mv "$LOGDIR/train.log" "$LOGDIR/train_${STAMP}.log"
    RESUME_ARG="--no-resume"
fi

echo ">> training start $(date -u +%FT%TZ) nproc=$NPROC config=$CONFIG ${RESUME_ARG:-(resume)}" \
    | tee -a "$LOGDIR/train.log"
uv run --extra train torchrun --standalone --nproc_per_node="$NPROC" \
  -m kws_mandarin.train --config "$CONFIG" $RESUME_ARG 2>&1 | tee -a "$LOGDIR/train.log"
echo ">> training exit $? $(date -u +%FT%TZ)" | tee -a "$LOGDIR/train.log"
