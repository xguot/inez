#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Submit Inezita preparation and training from any directory. Environment overrides are exported by sbatch.
set -euo pipefail
ROOT_DIR="${INEZ_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT_DIR}"
command -v sbatch >/dev/null || { echo "Run this submitter on Rivanna." >&2; exit 1; }
if [[ "${INEZ_PYTHON:-}" == HOME/* ]]; then
    export INEZ_PYTHON="${HOME}/${INEZ_PYTHON#HOME/}"
fi
ACCOUNT="${INEZ_ACCOUNT:-nssac_students}"
MODE="${1:-prepare}"
case "${MODE}" in
    continue)
        : "${INIT_ARTIFACT:?Set INIT_ARTIFACT to the accepted model artifact to improve}"
        export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/inezita-0.1.3.json}"
        export RUN_DIR="${RUN_DIR:-${ROOT_DIR}/results/training/inezita-0.1.3}"
        export DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/processed/inezita}"
        TRAIN_JOB="$(sbatch --parsable --account="${ACCOUNT}" hpc/train_inezita.slurm)"
        echo "Inezita continuation job: ${TRAIN_JOB}; config=${TRAIN_CONFIG}; run=${RUN_DIR}"
        exit 0
        ;;
    prepare) ;;
    *) echo "Usage: bash hpc/submit_inezita.sh [prepare|continue]" >&2; exit 1 ;;
esac
PREP_JOB="$(sbatch --parsable --account="${ACCOUNT}" hpc/prepare_inezita.slurm)"
PREP_JOB="${PREP_JOB%%;*}"
echo "Inezita data preparation job: ${PREP_JOB}"
TRAIN_JOB="$(sbatch --parsable --account="${ACCOUNT}" --dependency="afterok:${PREP_JOB}" --kill-on-invalid-dep=yes hpc/train_inezita.slurm)"
echo "Inezita training job: ${TRAIN_JOB}; starts after successful preparation."
