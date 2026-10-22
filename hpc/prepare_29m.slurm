#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# CPU preparation: download Elite PGNs, split games and generate Stockfish labels.
#SBATCH --job-name=inez_prepare29m
#SBATCH --output=inez_prepare_%j.out
#SBATCH --error=inez_prepare_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -Eeuo pipefail
trap 'rc=$?; echo "Preparation failed at line ${LINENO}; exit=${rc}" >&2' ERR
ROOT_DIR="${INEZ_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "${ROOT_DIR}"
if command -v module >/dev/null 2>&1 && [[ -z "${CXX:-}" ]]; then
    module load "${INEZ_COMPILER_MODULE:-gcc}"
    export CXX="$(command -v g++)"
fi
if [[ -n "${INEZ_STOCKFISH_MODULE:-}" ]]; then
    module load "${INEZ_STOCKFISH_MODULE}"
fi
if [[ -n "${INEZ_PYTHON:-}" ]]; then
    PY="${INEZ_PYTHON}"
elif [[ -x "${HOME}/.conda/envs/inez/bin/python" ]]; then
    PY="${HOME}/.conda/envs/inez/bin/python"
elif [[ -x "${ROOT_DIR}/.venv-model/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv-model/bin/python"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv/bin/python"
else
    PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    if [[ -z "${PY}" ]]; then
        echo "ERROR: No Python interpreter found. Set INEZ_PYTHON or create a virtualenv." >&2
        exit 1
    fi
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
"${PY}" -u tools/train_inez.py prepare \
    --config "${TRAIN_CONFIG:-configs/train/inezita.json}" \
    --data-dir "${DATA_DIR:-data/processed/inezita}" \
    --stockfish "${INEZ_STOCKFISH:-stockfish}" --threads "${OMP_NUM_THREADS}"
