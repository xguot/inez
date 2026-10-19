# PUCT search and UCI baseline

The search layer (`include/inez/search.hpp`, `src/search.cpp`) and UCI engine
(`include/inez/uci.hpp`, `src/uci.cpp`, `src/main.cpp`) provide a playable,
measurable MCTS/PUCT baseline on top of the rules core.

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

The `inez` executable speaks UCI on `stdin`/`stdout` and writes diagnostic
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

## Automated match runner and baseline report

`tools/match.py` pairs two UCI configurations (`inez-material` vs.
`inez-uniform`) across 8 balanced openings with reversed colors, validates every
move with `python-chess`, tracks crashes/illegal moves/timeouts, computes 95%
Elo confidence bounds and SPRT log-likelihood ratios, runs a 6-position
tactical/endgame suite, and benchmarks fixed-node and timed runs:

```sh
.venv/bin/python tools/match.py build/inez --pairs 4 --nodes 64 --batch-size 4
```

### Baseline verification results (Release build)

- **Paired-opening match (`4` pairs / `8` games, `64` nodes/move, `BatchSize=4`)**:
  `inez-material` vs. `inez-uniform` scored **+8 =0 -0** (`100.0%`, all 8 won by
  checkmate across `startpos`, `italian`, `sicilian_open`, and `french_advance`
  with colors reversed), `0` crashes, `0` illegal moves, `0` timeouts, and
  `SPRT(0, 50)` `LLR = +5.31 > +2.94` (`H1_accepted`).
- **Tactical & endgame suite (`6/6` solved, `100%`)**:
  Solved `back_rank_mate_1` (`d1d8`), `queen_mate_1` (`g5g7`),
  `smothered_mate_1` (`h6f7`), `free_queen_capture` (`c3d5`),
  `promote_with_mate` (`e7e8q`), and `avoid_stalemate_mate_1` (`g4g7`).
- **Throughput, batch utilization & leaf latency diagnostics (`startpos`)**:
  - `fixed_nodes_serial` (`128` nodes, `BatchSize=1`): `~120k nodes/sec`,
    `128` batches, `avg_batch_size = 1.00`, `batch_utilization = 1.00`,
    `avg_eval_latency = 0.71 us`.
  - `fixed_nodes_batched` (`128` nodes, `BatchSize=4`): `~128k nodes/sec`,
    `33` batches (`1` root + `32` batches of `4`), `avg_batch_size = 3.88`,
    `batch_utilization = 0.97`, `avg_eval_latency = 2.76 us`.
  - `timed_movetime_batched` (`40 ms`, `BatchSize=4`): `4,033` nodes
    (`max_depth = 8`, `~101k nodes/sec`), `1,009` batches,
    `avg_batch_size = 4.00`, `batch_utilization = 1.00`,
    `avg_eval_latency = 2.54 us`.
