# PUCT search and UCI baseline

The search layer (`include/inez/search.hpp`, `src/search.cpp`) and UCI engine
(`include/inez/uci.hpp`, `src/uci.cpp`, `src/main.cpp`) provide a playable,
measurable MCTS/PUCT baseline on top of the rules core.

To play trained neural weights in a GUI, use `build/inezita` in its default
[neural UCI mode](neural-uci.md); pass `--baseline` (or `INEZ_BASELINE=1`) for
the C++ heuristic baseline.

## Evaluators and search mechanics

`inez::Evaluator` defines the leaf evaluation interface:

- `evaluate(position, legal)` returns an `Evaluation` with `value` in `[-1, 1]`
  from the side to move's perspective (`+1` = side to move wins) and `policy`
  priors over `legal` summing to `1`.
- `evaluate_batch(queries)` evaluates a batch of `BatchQuery` leaf requests in
  one call. Its default implementation delegates serially to `evaluate()`;
  batched neural-network runtimes and test stubs override `evaluate_batch()`.

Two built-in evaluators are provided for Step 2:

- `UniformEvaluator`: returns `value = 0` and uniform priors (`1 / N`).
- `MaterialEvaluator`: combines centipawn material, piece-square tables, and a
  bishop-pair bonus mapped into `[-1, 1]` via `tanh(cp / 600)`, with softmax
  policy logits favoring MVV-LVA captures, promotions, piece-square gains, and
  enemy-king tropism.

During tree traversal, `Search` selects edges maximizing `Q + U`:

- Exploitation `Q` uses backed-up values from the parent's side-to-move
  perspective, penalized by `virtual_loss` per in-flight trajectory, with
  parent-relative First Play Urgency (`initial_value - 0.15`) on unvisited
  siblings after the first visit.
- Exploration `U` uses `cpuct * prior * sqrt(max(1, total_visits)) / (1 + eff_visits)`.
- Terminal leaves (`checkmate`, `stalemate`, `insufficient_material`, and
  interior `threefold` / `50-move` draws at `depth > 0`) are scored directly
  (`-1.0` for checkmate from the mated side's perspective, `0.0` for draws)
  without invoking the leaf evaluator.
- When `batch_size > 1`, virtual loss (`in_flight`) routes concurrent
  within-batch trajectories across distinct unexpanded leaves before calling
  `evaluate_batch()`. If a trajectory collides with a leaf already pending in
  the active batch, the batch flushes immediately so subsequent batches can
  descend into the newly expanded children.

## UCI protocol and stream separation

The `inezita` executable speaks UCI on `stdin`/`stdout` and writes diagnostic
telemetry (`diag nodes=... batches=... avg_batch=... batch_util=... eval_us=...`)
and warnings/errors exclusively to `stderr` (`log`). Supported commands:

- `uci`, `isready`, `ucinewgame`, `quit`
- `setoption name <Name> value <Value>`:
  - `Evaluator`: `material` (default) or `uniform`
  - `BatchSize`: `1..256` (default `1`)
  - `Cpuct`: nonnegative float (default `2.5`)
  - `VirtualLoss`: nonnegative float (default `1.0`)
  - `InfoInterval`: simulation interval for `info` lines (default `500`)
- `position startpos [moves ...]` and `position fen <6-field FEN> [moves ...]`
- `go [nodes N] [movetime MS] [wtime MS] [btime MS] [winc MS] [binc MS] [movestogo N] [infinite]`
- `stop`

## Integration checks and strength evaluation

The native material/uniform evaluators remain explicit test/reference modes.
`tests/baseline_uci_tests.py` exercises both through an independent python-chess
client, with reversed colors and legal-move checks. A smoke-test ply cap does not
supply a game result or strength estimate. It runs under CTest when
`INEZ_DIFFERENTIAL_TESTS=ON`.

Use [the benchmark protocol](benchmarking.md) for neural checkpoint comparisons,
reference-engine matches, sample sizes, paired statistics and Slurm execution.
The earlier handcrafted-only match runner and its approximate strength
statistics have been removed. Its [historical report](evaluation/handcrafted-baseline.md)
is retained for engineering context.
