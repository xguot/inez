#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Submit from the checkout root: sbatch hpc/benchmark_inezita.slurm
# Default is the 20-game pilot; no training is performed.
#SBATCH --job-name=inezita_bench
#SBATCH --output=hpc/logs/inezita_bench_%j.out
#SBATCH --error=hpc/logs/inezita_bench_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
trap 'rc=$?; echo "Benchmark failed at line ${LINENO}; exit=${rc}" >&2' ERR
ROOT_DIR="${INEZ_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "${ROOT_DIR}"
STAGE="${1:-pilot}"
case "${STAGE}" in pilot|baseline|gauntlet|confirmation) ;; *) echo "Use pilot, baseline, gauntlet, or confirmation" >&2; exit 1 ;; esac
if [[ "${STAGE}" != pilot && -z "${PILOT_DIR:-}" ]]; then
    echo "Set PILOT_DIR to the output directory of a completed 20-game pilot." >&2
    exit 1
fi
if command -v module >/dev/null 2>&1 && [[ -z "${CXX:-}" ]]; then
    module load "${INEZ_COMPILER_MODULE:-gcc}"
    export CXX="$(command -v g++)"
fi
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
export OMP_NUM_THREADS="${BENCH_THREADS:-4}" MKL_NUM_THREADS="${BENCH_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Fail before compiling/downloading if the selected card cannot execute kernels.
PYTHONPATH="${ROOT_DIR}/training:${PYTHONPATH:-}" "${PY}" -m inez.device --device "${BENCH_DEVICE:-cuda}"

# Pin the tournament engine instead of following a moving branch.
FASTCHESS_REV=f618e34540f94f4719ad3817950618dabe441318
if [[ -z "${FASTCHESS:-}" ]]; then
    # Separate portable objects from older builds made with -march=native.
    FASTCHESS_DIR="${ROOT_DIR}/data/raw/benchmarks/fastchess-${FASTCHESS_REV}-portable"
    if [[ ! -d "${FASTCHESS_DIR}" ]]; then
        mkdir -p "$(dirname "${FASTCHESS_DIR}")"
        git clone --depth 1 --branch v1.8.2-alpha https://github.com/Disservin/fastchess.git "${FASTCHESS_DIR}"
    fi
    [[ "$(git -C "${FASTCHESS_DIR}" rev-parse HEAD)" == "${FASTCHESS_REV}" ]] || {
        echo "Cached Fastchess source revision differs from the pin." >&2; exit 1;
    }
    make -C "${FASTCHESS_DIR}" -j "${SLURM_CPUS_PER_TASK:-8}" CXX="${CXX:-g++}" NATIVE=
    FASTCHESS="${FASTCHESS_DIR}/fastchess"
fi
BUILD_DIR="${ROOT_DIR}/build/benchmark"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build "${BUILD_DIR}" --target inezita --parallel "${SLURM_CPUS_PER_TASK:-8}"
OUT_DIR="${BENCH_OUTPUT:-${ROOT_DIR}/results/evaluation/inezita-0.1.3/${STAGE}-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
ARGS=(--stage "${STAGE}" --fastchess "${FASTCHESS}" --binary "${BUILD_DIR}/inezita"
      --output "${OUT_DIR}" --device "${BENCH_DEVICE:-cuda}" --threads "${BENCH_THREADS:-4}")
if [[ -n "${BENCH_MODEL:-}" ]]; then ARGS+=(--model "${BENCH_MODEL}"); fi
if [[ -n "${BENCH_OPPONENT:-}" ]]; then ARGS+=(--opponent "${BENCH_OPPONENT}"); fi
if [[ -n "${BENCH_REFERENCE:-}" ]]; then ARGS+=(--reference "${BENCH_REFERENCE}"); fi
if [[ -n "${BENCH_OPPONENT_ENGINE:-}" ]]; then ARGS+=(--opponent-engine "${BENCH_OPPONENT_ENGINE}"); fi
if [[ -n "${BENCH_OPPONENT_BINARY:-}" ]]; then ARGS+=(--opponent-binary "${BENCH_OPPONENT_BINARY}"); fi
if [[ -n "${BENCH_OPPONENT_SOURCE_ROOT:-}" ]]; then ARGS+=(--opponent-source-root "${BENCH_OPPONENT_SOURCE_ROOT}"); fi
if [[ -n "${BENCH_OPPONENT_MODEL:-}" ]]; then
    ARGS+=(--opponent-option "ModelPath=${BENCH_OPPONENT_MODEL}"
           --opponent-option "Device=${BENCH_DEVICE:-cuda}"
           --opponent-option "Threads=${BENCH_THREADS:-4}")
fi
if [[ -n "${BENCH_OPPONENT_NAME:-}" ]]; then ARGS+=(--opponent-name "${BENCH_OPPONENT_NAME}"); fi
if [[ -n "${BENCH_OPPONENT_NODES:-}" ]]; then ARGS+=(--opponent-nodes "${BENCH_OPPONENT_NODES}"); fi
if [[ -n "${LC0_WEIGHTS:-}" ]]; then ARGS+=(--lc0-weights "${LC0_WEIGHTS}"); fi
if [[ -n "${OPENINGS_PGN:-}" ]]; then ARGS+=(--openings "${OPENINGS_PGN}"); fi
if [[ -n "${PILOT_DIR:-}" ]]; then ARGS+=(--pilot-dir "${PILOT_DIR}"); fi
if [[ -n "${BENCH_PAIRS:-}" ]]; then ARGS+=(--pairs "${BENCH_PAIRS}"); fi
if [[ -n "${BENCH_START:-}" ]]; then ARGS+=(--start "${BENCH_START}"); fi
if [[ -n "${BENCH_TC:-}" ]]; then ARGS+=(--time-control "${BENCH_TC}"); fi
if [[ -n "${BENCH_NODES:-}" ]]; then ARGS+=(--nodes "${BENCH_NODES}"); fi
echo "Inezita ${STAGE}; output=${OUT_DIR}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi
CHILD=""
stop_pending=0
request_stop() {
    stop_pending=1
    if [[ -n "${CHILD}" ]]; then kill -TERM "${CHILD}" 2>/dev/null || true; fi
}
trap request_stop USR1 TERM
"${PY}" -u tools/benchmark_inezita.py "${ARGS[@]}" &
CHILD=$!
if [[ "${stop_pending}" == 1 ]]; then request_stop; fi
while true; do
    if wait "${CHILD}"; then status=0; break; else status=$?; fi
    if ! kill -0 "${CHILD}" 2>/dev/null; then break; fi
done
exit "${status}"
