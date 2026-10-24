#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Explicit, frozen training submission; use continue, scratch, or resume.
# Direct sbatch submission of this wrapper runs the dispatcher in a short
# allocation; it schedules the frozen prepare and train jobs and exits.
#SBATCH --job-name=inezita_submit
#SBATCH --output=inezita_submit_%j.out
#SBATCH --error=inezita_submit_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
ROOT_DIR="${INEZ_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ "${INEZ_PYTHON:-}" == HOME/* ]]; then
    INEZ_PYTHON="${HOME}/${INEZ_PYTHON#HOME/}"
fi
if [[ -n "${INEZ_PYTHON:-}" ]]; then
    PY="$(command -v "${INEZ_PYTHON}")"
elif [[ -x "${HOME}/.conda/envs/inez/bin/python" ]]; then
    PY="${HOME}/.conda/envs/inez/bin/python"
elif [[ -x "${ROOT_DIR}/.venv-model/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv-model/bin/python"
else
    PY="$(command -v python3)"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "${PY}" "${ROOT_DIR}/tools/submit_training.py" "$@"
