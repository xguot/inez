#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Single-GPU smoke test and diagnostic overfit for Slurm clusters.
# Builds the C++20 engine, runs CTest contract/model checks, verifies CUDA
# overfit on a small diagnostic batch, and exports a verified v1 model artifact.
# Run from the repository root: sbatch hpc/run_smoke.slurm
#SBATCH --job-name=inez_smoke
# Keep bootstrap logs in the existing submission directory: Slurm opens these
# before the script runs, so a missing log subdirectory prevents startup.
#SBATCH --output=inez_smoke_%j.out
#SBATCH --error=inez_smoke_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -Eeuo pipefail
trap 'status=$?; echo "ERROR: smoke test failed at line ${LINENO} (exit ${status})" >&2' ERR
ROOT_DIR="${INEZ_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "${ROOT_DIR}"

JOB_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="${ROOT_DIR}/results/smoke/${JOB_ID}"
mkdir -p "${OUT_DIR}"

cleanup() {
    local status=$?
    if [[ -n "${JOB_TMP:-}" ]]; then
        rm -rf -- "${JOB_TMP}" || true
    fi
    echo "Smoke test finished; exit=${status}"
    exit "${status}"
}

(
    # Start logging before Python selection, mktemp, or module initialization.
    JOB_TMP=""
    trap cleanup EXIT
    echo "Inez Slurm GPU smoke test; job=${JOB_ID} host=$(hostname)"
    echo "Output: ${OUT_DIR}"

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

    JOB_TMP="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/inez-smoke.XXXXXX")"
    export TMPDIR="${JOB_TMP}"
    export XDG_CACHE_HOME="${JOB_TMP}/cache"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="${ROOT_DIR}/training:${ROOT_DIR}/tools:${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    mkdir -p "${XDG_CACHE_HOME}"

    BUILD_DIR="${JOB_TMP}/build"
    SMOKE_DEVICE="${DEVICE:-cuda}"
    echo "Python: ${PY}"
    echo "Diagnostic device: ${SMOKE_DEVICE}"

    # Batch jobs can otherwise pick /usr/bin/c++ (GCC 8 on Rivanna), which
    # accepts -std=c++2a but lacks the C++20 features required by the engine.
    # Use the site's default GCC rather than assuming a version is installed.
    # Respect an explicit CXX; allow a specific module to be requested.
    COMPILER_MODULE="${INEZ_COMPILER_MODULE:-}"
    if [[ -z "${COMPILER_MODULE}" && -z "${CXX:-}" ]] && command -v module >/dev/null 2>&1; then
        COMPILER_MODULE=gcc
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
        --device "${SMOKE_DEVICE}" \
        --steps "${STEPS:-40}" \
        --lr "${LR:-1e-3}" \
        --seed "${SEED:-20261019}"; then
        echo "C++ engine build, CTest contracts, and ${SMOKE_DEVICE} diagnostic overfit passed."
    else
        exit "$?"
    fi
) 2>&1 | tee -a "${OUT_DIR}/run.log"
