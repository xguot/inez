#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Single-GPU smoke test and diagnostic overfit for Slurm clusters.
# Builds the C++20 engine, runs CTest contract/model checks, verifies CUDA
# overfit on a small diagnostic batch, and exports a verified v1 model artifact.
# Run: mkdir -p hpc/logs && sbatch -A <account> hpc/run_smoke.slurm
#SBATCH --job-name=inez_smoke
#SBATCH --output=hpc/logs/inez_smoke_%j.out
#SBATCH --error=hpc/logs/inez_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
ROOT_DIR="${INEZ_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "${ROOT_DIR}"

JOB_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="${ROOT_DIR}/results/smoke/${JOB_ID}"
mkdir -p "${OUT_DIR}"

if [[ -n "${INEZ_PYTHON:-}" ]]; then
    PY="${INEZ_PYTHON}"
elif [[ -x "${HOME}/.conda/envs/inez/bin/python" ]]; then
    PY="${HOME}/.conda/envs/inez/bin/python"
elif [[ -x "${ROOT_DIR}/.venv-model/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv-model/bin/python"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv/bin/python"
else
    PY=python3
fi

JOB_TMP="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/inez-smoke.XXXXXX")"
trap 'rm -rf -- "${JOB_TMP}"' EXIT
export TMPDIR="${JOB_TMP}"
export XDG_CACHE_HOME="${JOB_TMP}/cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${ROOT_DIR}/tools:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${XDG_CACHE_HOME}"

BUILD_DIR="${JOB_TMP}/build"
{
    echo "Inez Slurm GPU smoke test; job=${JOB_ID} host=$(hostname)"
    echo "Output: ${OUT_DIR}; Python: ${PY}"

    # Batch jobs can otherwise pick /usr/bin/c++ (GCC 8 on Rivanna), which
    # accepts -std=c++2a but lacks the C++20 features required by the engine.
    # Respect an explicit CXX; allow other clusters to select their module.
    COMPILER_MODULE="${INEZ_COMPILER_MODULE:-}"
    if [[ -z "${COMPILER_MODULE}" && -z "${CXX:-}" ]] && command -v module >/dev/null 2>&1; then
        COMPILER_MODULE=gcc/13.3.0
    fi
    if [[ -n "${COMPILER_MODULE}" ]]; then
        if ! command -v module >/dev/null 2>&1; then
            echo "Compiler module requested, but module is unavailable; initialize modules or set CXX instead." >&2
            exit 1
        fi
        echo "Loading compiler module: ${COMPILER_MODULE}"
        module load "${COMPILER_MODULE}"
        export CXX="${CXX:-$(command -v g++)}"
    fi
    echo "C++ compiler: ${CXX:-c++}"
    "${CXX:-c++}" --version

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    fi

    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DINEZ_DIFFERENTIAL_TESTS=ON \
        -DINEZ_MODEL_TESTS=ON \
        -DPython3_EXECUTABLE="${PY}"
    cmake --build "${BUILD_DIR}" --parallel "${OMP_NUM_THREADS}"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure

    if "${PY}" -u tools/smoke_gpu.py \
        --out-dir "${OUT_DIR}" \
        --device "${DEVICE:-cuda}" \
        --steps "${STEPS:-40}" \
        --lr "${LR:-1e-3}" \
        --seed "${SEED:-20261019}"; then
        echo "C++ engine build, CTest contracts, and GPU diagnostic overfit passed."
    else
        exit "$?"
    fi
} 2>&1 | tee -a "${OUT_DIR}/run.log"
