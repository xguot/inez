# Inez

Stealing the quiet hours of my twenties.

## Development

See [TODO](TODO) for the development roadmap and next implementation milestone.

The engine includes:

- A C++20 standard-chess rules library with legal move generation, FEN
  import/export, reversible moves, and draw-history handling
  ([rules guide](docs/rules.md)).
- A serial and batched PUCT search baseline with material/uniform evaluators,
  UCI engine (`inez`), and automated match/benchmark runner
  ([search guide](docs/search.md)).

### Build and test

Requires CMake 3.20 or newer and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For the independent python-chess oracle, automated match suite, and debug
sanitizers:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements.txt
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DINEZ_SANITIZERS=ON -DINEZ_DIFFERENTIAL_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-debug --parallel
ctest --test-dir build-debug --output-on-failure
```

Python is only needed for the differential and match tests. The engine has no
runtime dependencies beyond the C++ standard library.

### Slurm GPU smoke test

Install `training/requirements.txt` in a Python environment with CUDA-enabled
PyTorch, then submit from the repository root:

```sh
INEZ_PYTHON=/absolute/path/to/env/bin/python sbatch hpc/run_smoke.slurm
```

The job builds the engine, runs CTest (including model checks), and performs a
GPU diagnostic overfit and checkpoint round trip. The script submits to the
`nssac_students` allocation (override it with `sbatch -A <account>`), the same
account used by the lensless-recon jobs. Logs and artifacts are written
to `results/smoke/<job-id>/`. Slurm bootstrap output and errors go to
`inez_smoke_<job-id>.out` and `inez_smoke_<job-id>.err` in the submission
directory, so no log directory needs to exist before the job starts.

If a job disappears quickly, inspect its final state and log paths:

```sh
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
scontrol show job JOBID
cat inez_smoke_JOBID.out inez_smoke_JOBID.err
```

`CG` means Slurm is completing cleanup; use `sacct` to check whether the job
succeeded. The run log includes startup failures and a final exit status.

Interpret copied results using the recorded execution metadata. Copying an HPC
result to this checkout does not make it a local run, and CPU execution does
not imply a laptop run: HPC jobs can also run on CPU. Folder names and artifact
paths alone do not establish where computation happened.

New summaries record the execution hostname, Slurm job ID, Python executable,
PyTorch/CUDA versions, and requested device in `run_context`. The top-level
`device` records the model's actual device, and `gpu_name` identifies its CUDA
GPU. A CPU smoke pass verifies learning and checkpoint export/reload; a CUDA
smoke pass additionally verifies GPU execution. Older summaries without
`run_context` should be read alongside their original job logs.

The diagnostic weights come from overfitting a tiny batch; they are a pipeline
check. The next development milestone is a bounded dataset training loop with
held-out validation and resumable training checkpoints, followed by engine
inference integration and controlled baseline matches. Confirm CUDA execution
in the first HPC training pilot before scaling up.

When environment modules are available, the job loads the site's default `gcc`
module. Use `module spider gcc` to list installed versions and select one with
`INEZ_COMPILER_MODULE=gcc/<installed-version>` if the default lacks C++20
support. Set `CXX=/absolute/path/to/compiler` to use a prepared toolchain
without loading a module. GCC 8.5, the Rivanna system compiler seen in the first
smoke run, lacks the required C++20 features; CMake now checks these at configure
time. Python selection can be overridden with `INEZ_PYTHON`; otherwise the job
tries `~/.conda/envs/inez`, `.venv-model`, `.venv`, then `python3`.

### Run the UCI engine and match runner

Run `build/inez` for standard UCI communication (`uci`, `isready`, `position`,
`go`, `stop`, `quit`), or run the paired-opening match and benchmark suite:

```sh
.venv/bin/python tools/match.py build/inez --pairs 4 --nodes 64 --batch-size 4
```

### Inspect a position

Run `build/inez-rules` and enter one command per line:

```text
start
push e2e4
state
pop
perft 4
quit
```

The tool returns JSON snapshots with legal moves, FEN, and draw information.
Use `fen <six-field FEN>` to inspect another position.

## License

Inez is licensed under the GNU General Public License v3.0.
See [LICENSE](LICENSE) for details.
