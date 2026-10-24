#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Explicit, frozen training submission; use continue, scratch, or resume.
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
