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
PREP_JOB="$(sbatch --parsable --account="${ACCOUNT}" hpc/prepare_inezita.slurm)"
PREP_JOB="${PREP_JOB%%;*}"
echo "Inezita data preparation job: ${PREP_JOB}"
TRAIN_JOB="$(sbatch --parsable --account="${ACCOUNT}" --dependency="afterok:${PREP_JOB}" --kill-on-invalid-dep=yes hpc/train_inezita.slurm)"
echo "Inezita training job: ${TRAIN_JOB}; starts after successful preparation."
