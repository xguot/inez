# Inezita 0.1.3 development

The objective is to train the 29.2M model from scratch on our cleanest curated
corpus and 50,000-node Stockfish 17.1 teacher toward the Stockfish 2200–2400
reference range under frozen practical time/hardware conditions. This is a
development target. No newly trained 0.1.3 model or 2200+ result is claimed by
this code change. The [measured 0.1.2 baseline](evaluation/inezita-0.1.2/README.md)
is preserved.

## Training and data changes

Self-play minibatches now draw teacher examples from the complete verified
teacher training pool on every step. Stochastic rounding preserves the requested
mixture in expectation, including tiny batches; priority weights apply within
each pool. Checkpoints record realized primary/replay counts and the sampler RNG.
Both pools, the validation split, mixture and configuration are fingerprinted;
changing any of them requires a fresh run. Rejected proposals never initialize
the next self-play round.

Every exported checkpoint is reloaded through the strict artifact loader before
validation. This evaluates the actual EMA bytes used by games and records the
weights hash. Raw optimizer weights and their AdamW state remain together in
`last.pt`, including the final step. Validation reports policy cross-entropy,
policy top-one agreement, WDL cross-entropy/top-one agreement, expected-value
MAE/MSE, and squared WDL-distribution distance on a reproducible 2,048-position
validation sample. `validation_limit: null` requests the complete split. These
measure agreement with teacher distributions; actual outcome calibration still
needs independent game results. The evaluation split is never used for updates.

The 0.1.3 full-curriculum recipe (`configs/train/inezita.json`) trains the
29.2M model architecture (`configs/models/inezita.json`) end-to-end from scratch
so no 0.1.2 biases or weak-teacher artifacts carry over:

| Setting | 0.1.2 baseline | 0.1.3 (`configs/train/inezita.json`) |
| --- | ---: | ---: |
| Imported games | 20,000 | 100,000 |
| Minimum player rating (`min_rating`) | 2300 | 2400 |
| Minimum time control (`base + 40 * inc`) | Any (`0s`) | `>= 260s` (`180+2`, `300+0`, `600+0`, etc.) |
| Termination filter | Any completed result | `Normal` only (excludes time forfeits) |
| Human initialization steps / peak LR | 20,000 / 3e-4 | 20,000 / 3e-4 (every 5,000 steps) |
| Teacher Stockfish 17.1 budget | 10,000 nodes | 50,000 nodes (`label_contract: 2`, fresh `ucinewgame` per position) |
| Teacher steps / peak LR | 30,000 / 1e-4 | 30,000 / 1e-4 (every 5,000 steps) |
| Self-play rounds / games per round | 2 / 32 | 4 / 256 |
| Simulations per self-play move | 128 | 256 |
| Self-play steps / peak LR | 4,000 / 1.5e-5 | 1,000 / 1.5e-5 |
| Teacher replay fraction | 75% (round-subset) | 80% (full verified teacher pool) |

## Promotion

Each candidate plays 32 distinct opening pairs against the incumbent and up to
one earlier accepted model, at 128 simulations per move. All games must finish,
score must reach 55%, and previously solved tactical controls must remain solved.
An exact one-sided pair-level sign-flip test keeps the two color-reversed games
together. It assumes independent opening-pair engine-label exchangeability under
the null; it is a fixed-sample engineering gate, not an Elo interval or SPRT.
Opening positions are fixed before play and transposed duplicates are rejected.
Both self-play generation and promotion gates commit atomic per-game progress
(`training/inez/progress.py`), so a Slurm time limit or signal resumes at the
last completed game boundary without losing finished games.

A 0.05 run-wide alpha budget is divided across the predeclared candidate
checkpoints. A candidate must pass every required opponent, so no separate
opponent multiplicity factor is used. Inconclusive results retain the incumbent.
Repeating unchanged candidates in new runs is not additional independent
evidence. Cached reports must match the artifact, parent history, openings,
tactics, source/model hashes and significance budget before recovering a
promotion. Old reports remain readable but cannot authorize new acceptance.

## Submit on Rivanna (One Command)

Both launching the new 0.1.3 run and resuming it after a Slurm time limit use a
**single command** with no required environment variables:

```sh
bash hpc/submit_inezita.sh
```

What this single command does automatically:
- **First run** (when `results/training/inezita-0.1.3/job.json` does not exist):
  snapshots the repository into `results/training/inezita-0.1.3/source/`, writes
  `results/training/inezita-0.1.3/job.json`, submits `hpc/prepare_inezita.slurm`
  to curate 100,000 `Normal`-termination 2400+ games (`>= 260s` time control)
  into `data/processed/inezita-0.1.3` with 50,000-node Stockfish labels, and
  automatically chains `hpc/train_inezita.slurm` (`--dependency=afterok`) on an
  A100 40GB GPU for the full human $\rightarrow$ teacher $\rightarrow$ 4-round
  self-play curriculum.
- **Subsequent runs** (when `results/training/inezita-0.1.3/job.json` already
  exists): automatically detects the frozen experiment and resumes preparation or
  training from the exact committed chunk, step, or game boundary.

Use `RESUME_MODE=portable bash hpc/submit_inezita.sh` only when migrating across
different GPU architectures or PyTorch versions. Preserve the frozen 0.1.2
artifact; select a trained model explicitly with `ModelPath`/`BENCH_MODEL` until
release acceptance.

## Release evidence

Run fresh pilots for each new code/model combination; older pilots cannot
approve larger runs with changed sources, binaries or weights. Measure candidate
vs frozen 0.1.2 in both equal-simulation and equal-clock experiments, then use the
Stockfish 1600/1900/2200 ladder. Start with 20-game pipeline pilots, followed by
predeclared 1,000-game reference matches and a 2,000-game direct release comparison
when costs are established. Extend an inconclusive result only under a newly
specified test; do not inspect every few games and stop on a favorable score.
Keep the final opening slice separate from development matches and training.

```sh
export BENCH_MODEL="$PWD/results/training/inezita-0.1.3"
export BENCH_START=2010
export GAUNTLET_REFERENCES="stockfish-elo-1600 stockfish-elo-1900 stockfish-elo-2200"
sbatch hpc/gauntlet_inezita.slurm
```

The 2200 preset is Stockfish's configured handicap, not a human-rating certificate.
Freeze its binary/settings and publish W/D/L, paired outcomes, uncertainty,
throughput, hardware and faults. Achieving the development target requires the
predeclared evidence; parameter count and training loss cannot establish it.
See the [benchmark protocol](benchmarking.md).

To compare frozen 0.1.2 code with 0.1.3, create a separate checkout at
`3edaf018e9dc5a5e11f739d27cec5475ed1f8637` and build its launcher there. Supply that
binary, its exact source root, and the pinned 0.1.2 model to the current driver:

```sh
export BENCH_MODEL="$PWD/results/training/inezita-0.1.3"
export BENCH_OPPONENT_BINARY=/absolute/frozen-0.1.2/build/inezita
export BENCH_OPPONENT_SOURCE_ROOT=/absolute/frozen-0.1.2
export BENCH_OPPONENT_MODEL="$PWD/results/training/inezita/teacher/candidate-0020000"
export BENCH_OPPONENT_NAME=inezita-0.1.2
export BENCH_START=2020
sbatch hpc/benchmark_inezita.slurm
```

The source root is operator-declared because the UCI identity alone cannot prove
which Python checkout a launcher executes. Build from that same checkout; the
driver pins and rechecks source, binary and model hashes. Same-weight code
controls require distinct frozen source and binary hashes. Clear the external
opponent variables before returning to same-code checkpoint comparisons.
