#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Submit the frozen prepare/train pipeline from a login shell or a short CPU job.
# With sbatch, submit from the checkout root so relative log paths resolve.
# Slurm executes a copy of this script from its spool directory; INEZ_ROOT
# can explicitly select the checkout used by the dispatcher.
#SBATCH --job-name=inezita_submit
#SBATCH --output=hpc/logs/inezita_submit_%j.out
#SBATCH --error=hpc/logs/inezita_submit_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
ROOT_DIR="${INEZ_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
if [[ ! -f "${ROOT_DIR}/tools/submit_training.py" ]]; then
    echo "ERROR: Inezita checkout not found at ${ROOT_DIR}. Submit from the checkout root or set INEZ_ROOT." >&2
    exit 1
fi
cd "${ROOT_DIR}"
ROOT_DIR="$(pwd -P)"
if [[ "${INEZ_PYTHON:-}" == HOME/* ]]; then
    INEZ_PYTHON="${HOME}/${INEZ_PYTHON#HOME/}"
fi
if [[ -n "${INEZ_PYTHON:-}" ]]; then
    if ! PY="$(command -v -- "${INEZ_PYTHON}")" || [[ ! -f "${PY}" || ! -x "${PY}" ]]; then
        echo "ERROR: INEZ_PYTHON is not an executable Python interpreter: ${INEZ_PYTHON}" >&2
        exit 1
    fi
elif [[ -x "${HOME}/.conda/envs/inez/bin/python" ]]; then
    PY="${HOME}/.conda/envs/inez/bin/python"
elif [[ -x "${ROOT_DIR}/.venv-model/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv-model/bin/python"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv/bin/python"
elif ! PY="$(command -v python3)"; then
    echo "ERROR: No Python interpreter found. Set INEZ_PYTHON to the Inezita environment." >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "${PY}" "${ROOT_DIR}/tools/submit_training.py" "$@"
