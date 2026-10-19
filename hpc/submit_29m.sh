#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Submit from any directory. Environment overrides are exported by sbatch.
set -euo pipefail
ROOT_DIR="${INEZ_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT_DIR}"
command -v sbatch >/dev/null || { echo "Run this submitter on Rivanna." >&2; exit 1; }
ACCOUNT="${INEZ_ACCOUNT:-nssac_students}"
PREP_JOB="$(sbatch --parsable --account="${ACCOUNT}" hpc/prepare_29m.slurm)"
PREP_JOB="${PREP_JOB%%;*}"
echo "Data preparation job: ${PREP_JOB}"
TRAIN_JOB="$(sbatch --parsable --account="${ACCOUNT}" --dependency="afterok:${PREP_JOB}" hpc/train_29m.slurm)"
echo "29M training job: ${TRAIN_JOB}; starts after successful preparation."
