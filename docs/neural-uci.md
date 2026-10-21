# Play the trained Inez in En Croissant

Select the executable **`inez-neural` in the repository root** when adding a
local UCI engine to your chess GUI. It launches the Python neural model and
the same reference PUCT search used for training gates. It does not require
rebuilding `build/inez` or retraining the model.

The launcher finds Python in `.venv-model`, then `.venv`, then `PATH`. Install
`training/requirements.txt` in that interpreter first if necessary. Set
`INEZ_PYTHON=/absolute/path/to/python` to select another interpreter. The launcher
works when the GUI starts it from another working directory. Keep it inside the
checkout; it needs the Python sources and a model artifact, not just the launcher.

The default **ModelPath** is `results/training/inez-29m-pilot`. The engine reads
that run's `accepted.json` and loads the selected artifact. With the current
results this is `teacher/candidate-0002000`. Absolute cluster paths in copied
results resolve against the local copy of the same run, without editing any
training metadata. If you renamed a copied run, select the artifact directory
directly in ModelPath. Copy both `manifest.json` and `weights.pt`.

The engine appears as **Inez Neural 0.1.1**. Its stderr log records the actual
artifact path, parameter count and device when loaded. The manifest, encoding
contract and weight checksum are validated before playing. Missing, corrupt or
incompatible artifacts cause an explicit error; there is no heuristic fallback.
The existing `build/inez` remains the separate C++ material/uniform engine.

## Settings

| UCI option | Default | Meaning |
| --- | --- | --- |
| ModelPath | Original pilot run | Run directory, `accepted.json`, or an artifact directory |
| Device | `cpu` | `cpu`, Apple `mps`, or NVIDIA `cuda`; explicit device availability is checked |
| Threads | Up to 4 | PyTorch CPU threads, 1–64 |
| Simulations | 128 | PUCT simulations when `go` supplies no search limits |
| Cpuct | 2.5 | Positive PUCT exploration coefficient |
| MoveOverhead | 50 ms | Reserve subtracted from movetime / remaining clock |

CPU execution is the default and needs no GPU or cluster. Optional accelerators
need a compatible PyTorch installation. Changing ModelPath or Device reloads the
model on the next `isready` or `go`; it does not modify saved weights.

Choose a generous move time initially, for example 2–5 seconds, and adjust after
observing your computer's search speed. Timed searches use their full time budget;
Simulations is not an additional cap. Clock allocation uses the moving side's
remaining time, increment and `movestogo` (30 moves when omitted).

## Protocol and limits

The command loop stays responsive during search. Supported commands are `uci`,
`isready`, `setoption`, `ucinewgame`, `position startpos`, `position fen`, `go`,
`stop` and `quit`. Position move lists preserve history for neural features and
repetition. Invalid positions leave the previous position unchanged.

`go` supports `nodes`, `movetime`, `wtime`, `btime`, `winc`, `binc`, `movestogo`,
`depth`, `searchmoves` and `infinite`. Nodes count completed PUCT simulations;
depth is the deepest explored PUCT path, not an exhaustive alpha-beta depth.
A depth-limited search finishes when it first reaches that depth. If multiple
limits are supplied, the first reached ends search. Infinite analysis continues
until `stop`; a terminal root waits without calculating. Pondering, mate-search
limits, Chess960 and MultiPV are not supported or advertised.

Search emits UCI `info` with actual nodes, depth, elapsed milliseconds, nps and
a legal principal variation. Scores are a display mapping of backed-up neural
values, using the same mapping as the native engine; they are not calibrated
Stockfish centipawn evaluations. Terminal positions return `bestmove 0000`.
An immediate stop or exhausted clock returns a legal move without claiming an
unevaluated neural score. Timers and cancellation are checked between evaluations
and during traversal; an already-running model forward pass must finish, so move
time is not a hard real-time guarantee.

Only protocol output goes to stdout; Python warnings, selected-model information
and diagnostics go to stderr. Search failures are reported explicitly and make
the engine unusable for subsequent searches rather than selecting another evaluator.

To run from a terminal:

```sh
./inez-neural
# Then enter: uci, isready, position startpos, go movetime 3000, quit
```

Or choose another model/device:

```sh
./inez-neural --model results/training/inez-29m-continue --device cpu --threads 4
```

Protocol tests, including independent python-chess client integration:

```sh
.venv-model/bin/python tests/neural_uci_tests.py
```

For model behavior tests, including the bishop-loss regression from the recorded
game, see the [neural diagnostic suite](neural-diagnostics.md). It measures the
accepted weights at fixed search budgets without changing promotion decisions.

The UCI behavior follows the [UCI protocol specification](https://backscattering.de/chess/uci/).
