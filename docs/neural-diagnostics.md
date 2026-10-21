# Neural Inez diagnostics

Run the accepted model through the actual Python UCI adapter, with fixed search
budgets and no training or promotion changes:

```sh
.venv-model/bin/python tools/evaluate_neural.py \
  --model results/training/inez-29m-pilot \
  --simulations 32 128 512 --threads 4 \
  --output results/evaluation/neural-diagnostics/my-run.json
```

The interpreter needs `training/requirements.txt`. CPU is the default;
`--device mps` and `--device cuda` require an available compatible accelerator.
Model selection uses the same strict artifact loader as `build/inez`, including
the accepted checkpoint pointer and weight checksum verification. The runner
starts a separate UCI process using the same interpreter, model and device.
It checks legal moves, legal principal variations and completion of the requested
simulation count. Nodes are PUCT simulations, not Stockfish nodes or full-width
search depth.

Use a **new output path** for each run; existing reports are never overwritten.
The report is saved after each case, with `running`, `complete`, `error` or
`interrupted` status. Only a complete report has a score summary. Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | All diagnostic checks passed |
| 2 | The evaluation completed, but the model failed at least one behavior check |
| 1 | Runner, input, artifact or UCI error; do not treat this as a strength score |

## What is measured

The frozen fixture is `tests/fixtures/neural_diagnostics.json`. Every position is
tested in its original orientation and with colors and ranks mirrored. These
paired cases are correlated, not independent statistical samples.

| Section | Cases | Scope |
| --- | --- | --- |
| Move choice | 16 per budget | Saving attacked pieces, free queen/rook captures, a poisoned pawn, mate, stalemate avoidance and underpromotion |
| Raw value | 6 | Prefer queen/rook advantage to disadvantage; prefer saving the bishop to losing it in Xiyuan's game |
| Terminal controls | 8 | Checkmate, stalemate, automatic 75-move draw and fivefold repetition |

The bishop case replays all moves through **7...b5** from the recorded neural
game. This preserves the history used by the neural encoding and avoids silently
substituting a history-free FEN. Both bishop continuations are scored from the
same player's perspective even when the opponent is to move. The mirrored
repetition control also retains its move stack.

Labels have deliberately limited meanings. The knight test accepts the four safe
retreats. The poisoned-pawn test only rejects the specified queen sacrifice. The
stalemate test rejects every immediately stalemating move, without demanding the
fastest win. The underpromotion test restricts the root to four promotions:
queen stalemates, bishop/knight leave insufficient material, and rook wins.
The restriction is recorded in the report.

Value is the raw network's side-to-move `P(win) - P(loss)`, reoriented to the
specified player. Queen and rook pairs require a positive value for the advantage,
a negative value for the disadvantage, and a gap of at least 0.05. The bishop
pair only requires the gap. **0.05 is an explicit diagnostic margin**, not a
calibration threshold derived from game outcomes. Raw WDL probabilities, policy
leaders, actual moves, PVs and timings are retained for investigation.

Terminal positions bypass the neural network. Their successes are reported
separately and must not inflate the learned model's tactical score.

## Independent label verification

Optionally require a local Stockfish executable to confirm move labels and value
ordering before testing Inez:

```sh
.venv-model/bin/python tools/evaluate_neural.py \
  --model results/training/inez-29m-pilot --simulations 32 128 512 \
  --stockfish /absolute/path/to/stockfish --reference-nodes 100000 \
  --output results/evaluation/neural-diagnostics/verified-run.json
```

Reference analysis uses one thread, 64 MiB hash and full strength, with a fixed
node budget. A disagreement aborts evaluation instead of relabeling the case to
match Inez. The reference binary checksum, identity, options and returned scores
are recorded. Finite search is supporting evidence; mate, stalemate, promotion
outcomes and history controls also have direct rule-based tests.

The report pins the model checksum, parameter count, suite and source checksums,
Git base revision, interpreter/library versions, device, thread count, Cpuct and
search budgets. Timing depends on hardware and concurrent load; it is not a
strength score. The runner never edits weights, `accepted.json` or promotion gates.

## Interpreting results

This is a **development diagnostic**, including a known failure from a human
game. It supplies neither an Elo rating nor a held-out generalization estimate.
Do not train directly on these fixtures and then describe improvement here as
independent evidence. Keep a separate unseen tactical set and use color-swapped
engine matches across multiple openings and opponents for strength comparisons.
Compare the same opponent versions and hardware/time or simulation limits; report
wins, draws, losses, faults and uncertainty rather than inferring Elo from a
single game or parameter count.

The initial accepted-model measurement is recorded in
[the baseline report](evaluation/teacher-1000-neural-diagnostics.md).

To validate the runner without the 29M checkpoint or Stockfish:

```sh
.venv-model/bin/python tests/diagnostics_tests.py
```

These tests use a tiny deterministic random artifact and an independent
python-chess UCI client. They verify perspective handling, mirror/history
preservation, labels, report exit codes, error handling and read-only model use.
They also run under CTest when `INEZ_MODEL_TESTS=ON`.
