#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Pull inez HPC outputs from Rivanna: run results and Slurm job logs.
#
#   bash hpc/sync.sh [pull] [--dry-run]
#
# Set HPC_REMOTE to override the SSH host (default: rivanna).
# Set INEZ_REMOTE_PATH to override the remote checkout (default
# ~/scratch/inez). Code and prepared data stay on the remote; only
# results and logs come back.

set -euo pipefail

REMOTE_HOST="${HPC_REMOTE:-rivanna}"
REMOTE_PATH="${INEZ_REMOTE_PATH:-~/scratch/inez}"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        pull)      ;; # Only direction; accept it explicitly.
        --dry-run) DRY_RUN=true ;;
        *)
            echo "Unknown: $arg" >&2
            echo "Usage: bash hpc/sync.sh [pull] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_NAME="$(basename "$ROOT_DIR")"

RSYNC_FLAGS=(-avz --progress)
$DRY_RUN && RSYNC_FLAGS+=(--dry-run)

echo "=== Pull [${PROJECT_NAME}] from ${REMOTE_HOST}:${REMOTE_PATH} ==="
$DRY_RUN && echo "  Mode: DRY RUN"

if ! ssh "$REMOTE_HOST" "test -d ${REMOTE_PATH}"; then
    echo "Remote checkout not found: ${REMOTE_HOST}:${REMOTE_PATH}" >&2
    exit 1
fi

# Pull the whole results tree: run directories are dynamic and checkpoints
# are large, so do not enumerate subdirectories here.
if ssh "$REMOTE_HOST" "test -d ${REMOTE_PATH}/results"; then
    mkdir -p "${ROOT_DIR}/results"
    rsync "${RSYNC_FLAGS[@]}" \
        "${REMOTE_HOST}:${REMOTE_PATH}/results/" "${ROOT_DIR}/results/"
    echo "  results/ synced"
else
    echo "  results/ not found on remote; skipped"
fi

# Slurm writes bootstrap logs to the remote submission directory (the
# checkout root). Collect them under hpc/logs locally, off the root.
mkdir -p "${ROOT_DIR}/hpc/logs"
rsync "${RSYNC_FLAGS[@]}" \
    --include='/inez_smoke_*.out' \
    --include='/inez_smoke_*.err' \
    --include='/inez_train_*.out' \
    --include='/inez_train_*.err' \
    --include='/inez_prepare_*.out' \
    --include='/inez_prepare_*.err' \
    --include='/inezita_smoke_*.out' \
    --include='/inezita_smoke_*.err' \
    --include='/inezita_train_*.out' \
    --include='/inezita_train_*.err' \
    --include='/inezita_prepare_*.out' \
    --include='/inezita_prepare_*.err' \
    --exclude='*' \
    "${REMOTE_HOST}:${REMOTE_PATH}/" "${ROOT_DIR}/hpc/logs/"
echo "  Job logs synced into hpc/logs/"

echo "  Pull complete."
