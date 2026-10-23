# Historical handcrafted-engine verification

This archived report describes the earlier material/uniform C++ evaluator, not
the trained Inezita model. The old `tools/match.py` runner has been removed.
Its small sample, approximate SPRT and confidence calculations are not accepted
release-strength evidence; retain the values below only as historical outputs.
Use the [current benchmark protocol](../benchmarking.md) for neural evaluation.

- **Paired-opening match (`4` pairs / `8` games, `64` nodes/move, `BatchSize=4`)**:
  `inezita-material` vs. `inezita-uniform` scored **+8 =0 -0** (`100.0%`, all 8 won by
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
