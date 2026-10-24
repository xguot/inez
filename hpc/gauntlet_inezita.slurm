#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Submit from the checkout root: sbatch hpc/gauntlet_inezita.slurm
# Default runs the 20-game pilot gauntlet across external reference opponents
# (Stockfish 17.1 at Skill 0/128n, UCI_Elo 1320, UCI_Elo 1600, plus Lc0 when
# LC0_BINARY and LC0_WEIGHTS are provided). Pass `gauntlet` after completing
# the pilot to run 1,000 games (500 pairs) per opponent.
# To test the 0.1.3 target ladder, set GAUNTLET_REFERENCES to the space-separated
# stockfish-elo-1320 stockfish-elo-1600 stockfish-elo-1900 stockfish-elo-2200.
# These are opponent settings, not calibrated human ratings for Inezita.
#SBATCH --job-name=inezita_gauntlet
#SBATCH --output=hpc/logs/inezita_gauntlet_%j.out
#SBATCH --error=hpc/logs/inezita_gauntlet_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
trap 'rc=$?; echo "Gauntlet benchmark failed at line ${LINENO}; exit=${rc}" >&2' ERR
ROOT_DIR="${INEZ_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "${ROOT_DIR}"
STAGE="${1:-pilot}"
case "${STAGE}" in pilot|gauntlet|confirmation) ;; *) echo "Use pilot, gauntlet, or confirmation" >&2; exit 1 ;; esac
if [[ "${STAGE}" != pilot && -z "${PILOT_GAUNTLET_DIR:-}" && -z "${PILOT_DIR:-}" ]]; then
    echo "Set PILOT_GAUNTLET_DIR (or PILOT_DIR for a single reference) to a completed 20-game pilot directory." >&2
    exit 1
fi
if command -v module >/dev/null 2>&1 && [[ -z "${CXX:-}" ]]; then
    module load "${INEZ_COMPILER_MODULE:-gcc}"
    export CXX="$(command -v g++)"
fi
if [[ -n "${INEZ_STOCKFISH_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
    module load "${INEZ_STOCKFISH_MODULE}"
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

FASTCHESS_REV=f618e34540f94f4719ad3817950618dabe441318
if [[ -z "${FASTCHESS:-}" ]]; then
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

GAUNTLET_ROOT="${BENCH_OUTPUT:-${ROOT_DIR}/results/evaluation/inezita-0.1.3/gauntlet-${STAGE}-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
mkdir -p "${GAUNTLET_ROOT}"

if [[ -n "${GAUNTLET_REFERENCES:-}" ]]; then
    read -r -a REFERENCES <<< "${GAUNTLET_REFERENCES}"
else
    REFERENCES=(stockfish-skill-0-n128 stockfish-elo-1320 stockfish-elo-1600)
    if [[ -n "${LC0_BINARY:-}" || -n "${LC0_WEIGHTS:-}" ]]; then
        REFERENCES+=(lc0)
    fi
fi

echo "Inezita external gauntlet (${STAGE}); output=${GAUNTLET_ROOT}; references=${REFERENCES[*]}"
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

overall_status=0
for ref in "${REFERENCES[@]}"; do
    if [[ "${stop_pending}" == 1 ]]; then
        overall_status=1
        break
    fi
    REF_OUT="${GAUNTLET_ROOT}/${ref}"
    ARGS=(--stage "${STAGE}" --reference "${ref}" --fastchess "${FASTCHESS}"
          --binary "${BUILD_DIR}/inezita" --output "${REF_OUT}"
          --device "${BENCH_DEVICE:-cuda}" --threads "${BENCH_THREADS:-4}")
    if [[ -n "${BENCH_MODEL:-}" ]]; then ARGS+=(--model "${BENCH_MODEL}"); fi
    if [[ -n "${OPENINGS_PGN:-}" ]]; then ARGS+=(--openings "${OPENINGS_PGN}"); fi
    if [[ -n "${PILOT_GAUNTLET_DIR:-}" ]]; then
        ARGS+=(--pilot-dir "${PILOT_GAUNTLET_DIR}/${ref}")
    elif [[ -n "${PILOT_DIR:-}" ]]; then
        ARGS+=(--pilot-dir "${PILOT_DIR}")
    fi
    if [[ -n "${BENCH_PAIRS:-}" ]]; then ARGS+=(--pairs "${BENCH_PAIRS}"); fi
    if [[ -n "${BENCH_START:-}" ]]; then ARGS+=(--start "${BENCH_START}"); fi
    if [[ -n "${BENCH_TC:-}" ]]; then ARGS+=(--time-control "${BENCH_TC}"); fi
    if [[ -n "${BENCH_NODES:-}" ]]; then ARGS+=(--nodes "${BENCH_NODES}"); fi
    if [[ -n "${BENCH_OPPONENT_NODES:-}" ]]; then ARGS+=(--opponent-nodes "${BENCH_OPPONENT_NODES}"); fi
    if [[ "${ref}" == "lc0" ]]; then
        if [[ -n "${LC0_BINARY:-}" ]]; then ARGS+=(--opponent-engine "${LC0_BINARY}"); fi
        if [[ -n "${LC0_WEIGHTS:-}" ]]; then ARGS+=(--lc0-weights "${LC0_WEIGHTS}"); fi
    fi
    echo "=== Running reference: ${ref} -> ${REF_OUT} ==="
    "${PY}" -u tools/benchmark_inezita.py "${ARGS[@]}" &
    CHILD=$!
    if [[ "${stop_pending}" == 1 ]]; then request_stop; fi
    while true; do
        if wait "${CHILD}"; then status=0; break; else status=$?; fi
        if ! kill -0 "${CHILD}" 2>/dev/null; then break; fi
    done
    CHILD=""
    if [[ "${status}" != 0 ]]; then
        overall_status="${status}"
        break
    fi
done

"${PY}" - "${GAUNTLET_ROOT}" <<'EOF'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
opponents = {}
for manifest_path in sorted(root.glob('*/manifest.json')):
    data = json.loads(manifest_path.read_text())
    opponents[manifest_path.parent.name] = {
        'status': data.get('status'),
        'opponent': data.get('opponent'),
        'summary': data.get('summary'),
        'elapsed_seconds': data.get('elapsed_seconds'),
        'projected_1000_game_hours': data.get('projected_1000_game_hours'),
    }
summary_path = root / 'summary.json'
summary_path.write_text(json.dumps({'gauntlet_root': str(root), 'opponents': opponents}, indent=2) + '\n')
print(json.dumps({'gauntlet_summary': str(summary_path), 'opponents': opponents}, indent=2), flush=True)
EOF

exit "${overall_status}"
