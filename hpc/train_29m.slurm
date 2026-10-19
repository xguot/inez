#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Reference neural PUCT + gated 29M training. Persistent RUN_DIR enables resume.
#SBATCH --job-name=inez_train29m
#SBATCH --output=inez_train_%j.out
#SBATCH --error=inez_train_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
trap 'rc=$?; echo "Training job failed at line ${LINENO}; exit=${rc}" >&2' ERR
ROOT_DIR="${INEZ_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "${ROOT_DIR}"
OUT_DIR="${RUN_DIR:-${ROOT_DIR}/results/training/inez-29m-pilot}"
mkdir -p "${OUT_DIR}"
# A named pipe keeps this shell as the Slurm signal recipient and works on
# systems where /dev/fd process-substitution redirections are restricted.
LOG_TMP="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/inez-training-log.XXXXXX")"
mkfifo "${LOG_TMP}/pipe"
tee -a "${OUT_DIR}/run.log" < "${LOG_TMP}/pipe" &
LOGGER_PID=$!
exec > "${LOG_TMP}/pipe" 2>&1
finish() {
    local rc=$?
    trap - EXIT
    exec 1>&- 2>&-
    if ! wait "${LOGGER_PID}"; then rc=1; fi
    rm -rf -- "${LOG_TMP}"
    exit "${rc}"
}
trap finish EXIT
echo "Inez 29M training; host=$(hostname) job=${SLURM_JOB_ID:-none} output=${OUT_DIR}"

if [[ -n "${INEZ_PYTHON:-}" ]]; then
    PY="${INEZ_PYTHON}"
elif [[ -x "${HOME}/.conda/envs/inez/bin/python" ]]; then
    PY="${HOME}/.conda/envs/inez/bin/python"
elif [[ -x "${ROOT_DIR}/.venv-model/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv-model/bin/python"
else
    PY=python3
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
# Required by deterministic CUDA matmul on supported CUDA/PyTorch builds.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CONFIG="${TRAIN_CONFIG:-configs/train/inez-29m.json}"
"${PY}" -u tools/train_inez.py preflight --config "${CONFIG}" --device cuda
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

ARGS=(train --config "${CONFIG}" --data-dir "${DATA_DIR:-data/processed/elite-29m-pilot}"
      --run-dir "${OUT_DIR}" --device cuda --threads "${OMP_NUM_THREADS}")
if [[ -n "${INIT_ARTIFACT:-}" ]]; then
    ARGS+=(--init-artifact "${INIT_ARTIFACT}")
fi
TRAIN_PID=""
STOP_PENDING=0
request_checkpoint() {
    STOP_PENDING=1
    if [[ -n "${TRAIN_PID}" ]]; then
        kill -USR1 "${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap request_checkpoint USR1 TERM
"${PY}" -u tools/train_inez.py "${ARGS[@]}" &
TRAIN_PID=$!
if [[ "${STOP_PENDING}" == 1 ]]; then request_checkpoint; fi
# wait is interrupted by the Slurm warning signal; keep waiting for the child
# to finish its atomic checkpoint rather than exiting at the interrupted wait.
while true; do
    if wait "${TRAIN_PID}"; then
        STATUS=0
        break
    else
        STATUS=$?
        if kill -0 "${TRAIN_PID}" 2>/dev/null; then continue; fi
        break
    fi
done
echo "Training process finished; exit=${STATUS}"
if [[ "${STATUS}" == 75 ]]; then
    echo "Checkpoint saved. Resubmit with the same RUN_DIR, DATA_DIR, configuration and INIT_ARTIFACT."
fi
exit "${STATUS}"
