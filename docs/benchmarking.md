# Inezita benchmark protocol

This is the evaluation protocol for the 29,228,779-parameter Inezita model.
Inze is the future larger model. The game counts below are our proposed testing
budget, not measurements already obtained and not a universal industry minimum.
No benchmark or Slurm job has been launched by writing this protocol.

The [frozen 0.1.2 results](evaluation/inezita-0.1.2/README.md) preserve the measured
baseline. See [the 0.1.3 guide](inezita-0.1.3.md) for training, frozen-code controls
and the 1900/2200 reference ladder. Set `BENCH_MODEL` explicitly to the new run;
the default model path continues to use the existing accepted artifact.

## What major engines do

Stockfish separates development tests from progression measurements. Functional
changes normally go through short-time-control and then long-time-control SPRT
tests. Its published progression/regression criteria use 60,000 games at
60 seconds plus 0.6 seconds per move, in separate one-thread and eight-thread
tests. Those large samples are useful for measuring small improvements.

Sources:
[Stockfish regression tests](https://official-stockfish.github.io/docs/stockfish-wiki/Regression-Tests.html),
[Stockfish development testing](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html),
[Fishtest statistics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html).

Leela's testing guide describes tournaments between networks and other engines,
reusing each opening with reversed colors, and estimating relative ratings from
the games. It recommends controlling resource contention. Its example is
500 two-game rounds, or 1,000 games; that example is not a mandatory sample size.
See [Lc0 testing guide](https://lczero.org/dev/wiki/testing-guide/).

## Release testing for Inezita 0.1.3

Use Fastchess for games and statistical reporting, with complete PGNs retained.
The existing neural diagnostic runner remains a separate behavioral check.

| Stage | Games | Purpose |
| --- | ---: | --- |
| Pipeline pilot | 20 per matchup, 10 opening pairs | Check loading, clocks, hardware utilization, faults, PGN output and cost |
| Checkpoint comparison | 2,000, 1,000 opening pairs | Compare current accepted weights with one frozen earlier 29M checkpoint |
| Reference gauntlet | 1,000 per opponent, three opponents | Establish performance across different opponents; 3,000 games total |
| Longer-clock confirmation | 2,000 against the closest useful reference | Check that the result holds at a longer time control |

The main first baseline is 5,000 games after the pilot. Longer-clock confirmation
adds 2,000. These are initial estimates: the measured confidence interval may
still be too wide for a small claimed improvement. Do not describe a marginal
2,000-game lead as conclusive. Set fixed sample sizes before starting and finish
them; use a separately specified sequential test when early stopping is wanted.

Select references using pilot results: one somewhat weaker, one near Inezita's
strength, and one stronger or with a different playing style. Prefer multiple
engine families. A matchup with almost all wins or almost all losses is useful
as a bound but a poor instrument for detecting small changes. If Stockfish is
handicapped, freeze its exact version and handicap settings; a configured
`UCI_Elo` is not a calibrated human rating for Inezita.

## Which model sizes to compare

The primary development control is a previous **29M Inezita checkpoint**, running
through the same search code. This isolates the effect of changed weights.
Comparing whole releases requires freezing both releases' code and weights.

Choose external opponents by playing strength and diversity, not parameter
count. Include smaller and larger neural networks when they provide informative
opposition, alongside conventional engines. Stockfish's parameter count is not
a meaningful fairness criterion for this comparison. There is no requirement
to find three other models with exactly 29M parameters.

When a larger Inze artifact exists, compare it directly with Inezita in two
separate tracks:

1. Equal PUCT simulations per move, initially 128, to study the network/search
   combination per unit of search. This does not equalize arithmetic work:
   a larger model costs more per evaluation.
2. Equal clock time on specified equivalent hardware, to measure practical
   playing strength, including inference speed and time management.

Record simulations/second, inference latency and GPU-hours alongside strength.
An Inezita PUCT simulation is not a Stockfish node or an Lc0 visit with identical
semantics. Never call equal node counts across different engines equal compute.
Keep a CPU desktop track distinct from a CUDA track.

## Experimental controls

- Resolve `accepted.json` once before the experiment. Freeze the selected
  artifact directory, manifest and actual weights SHA256; the release tag alone
  does not identify untracked model weights. Record both engines' code revisions,
  executable hashes, versions, configuration and dependencies.
- Use a frozen, sufficiently large opening book, a recorded order/seed, and two
  games per opening with reversed colors. Each pair should have a distinct
  starting position. Repeating deterministic games does not add independent
  evidence. For the 2,000-game test, select 1,000 distinct opening pairs.
- Curate an evaluation book separately from training, MCO preparation, self-play
  sources, and promotion-gate openings. Audit overlaps and disclose unavoidable
  common-opening overlap. Until provenance is checked, call it a fixed evaluation
  book, not a held-out test. PGN opening lines preserve the history required by
  Inezita's encoding; record when EPD/FEN starts have no preceding history.
- Use disjoint opening slices for pilot, main comparison, confirmation, and
  future retests. Record the slice and source-book checksum. Do not train on the
  release evaluation set or tune repeatedly against it.
- Start the practical CUDA track at `60+0.6`; confirm at `180+1.8`. The pilot must
  establish that these clocks allow meaningful search. Freeze any adjustment
  before the main run. Keep time odds out of the equal-clock track.
- Use one game at a time per GPU initially, with pondering disabled. Fix GPU
  model, CPU allocation, thread settings, precision and engine versions. Run
  throughput experiments separately before changing concurrency. Engines that
  ponder would otherwise compete for the same device even during the other turn.
- Use natural game endings and supported rule-based draw claims. Do not enable
  evaluation-score resignation/draw adjudication: Inezita's displayed centipawns
  are not calibrated to Stockfish's. Tablebase adjudication is optional if the
  exact tablebase set and rules are frozen. A job interruption or arbitrary
  move cap is an unfinished game, never a manufactured draw.
- Preserve clock losses, crashes and illegal moves as explicit faults and retain
  their PGNs/logs; do not drop their losses to improve the score. Any such faults
  require investigation before treating the run as release acceptance evidence.

## Statistics and stopping

For a fixed-size measurement, publish opponent-by-opponent W/D/L, score, a 95%
interval for relative Elo, and the five paired-opening outcome counts
(`0`, `0.5`, `1`, `1.5`, `2` points). Use analysis that accounts for the two games
sharing an opening. Do not reuse the removed runner's approximate SPRT or treat
each game pair as two independent samples. Preserve raw PGNs so statistics can
be independently checked. An external gauntlet measures performance against
that pool under those conditions, not a FIDE/Lichess rating.

For future candidate acceptance, pre-register a two-engine SPRT. A reasonable
initial Inezita sensitivity target is Fastchess **normalized Elo** bounds
`[0, 10]`, `alpha=0.05`, `beta=0.05`, using paired outcomes. Normalized Elo is not
the same scale as ordinary logistic Elo. Treat these bounds as an explicit
project choice, not a copied Stockfish default. Specify a maximum budget, e.g.
20,000 games, before starting. Reaching that cap without a decision means
**inconclusive**. Smaller target improvements can require much more than 20,000
games. Do not keep retrying an unchanged candidate until a test happens to pass.

## Submission entry point

From the updated checkout on Rivanna, submit the pilot first:

```sh
sbatch hpc/benchmark_inezita.slurm
```

The job requests one A100 40GB (`--constraint=a100_40gb`) so the installed
CUDA 13 PyTorch build can execute its kernels. This is a 20-game, ten-opening-pair, CUDA comparison of the current accepted
Inezita artifact with the previous accepted artifact in the same run history.
For the pulled run, these are `teacher/candidate-0020000` and
`teacher/candidate-0005000`. Both use the same current search code. This is a
checkpoint comparison; the external reference gauntlet is a subsequent experiment.

Before downloads or builds, the job executes a CUDA tensor/matrix probe; merely
checking `torch.cuda.is_available()` does not establish kernel compatibility.
Each model also performs a real neural search before the tournament.

The job builds the Linux launcher in `build/benchmark`, builds Fastchess from
pinned commit `f618e34540f94f4719ad3817950618dabe441318` (v1.8.2-alpha), and downloads
Stockfish's `8moves_v3.pgn` from pinned book commit
`a804a97e709f40aa75a00f9d19e84915b7a3a45c`. It validates the archive SHA256,
retains opening histories, filters duplicate final positions and exact matches
with known MCO/gate positions, and uses separate eligible opening slices for
pilot, baseline and confirmation. Full human-training overlap has not been
audited; the manifest explicitly labels this a fixed public evaluation book,
not a verified held-out set. Fastchess is built with `NATIVE=` in a separate
`-portable` cache so moving from V100 nodes to A100 nodes does not reuse
CPU-specific `-march=native` objects.

The first run needs network access for the public source/book downloads, or the
pinned resources must already be cached under `data/raw/benchmarks/`. It needs
Git, make, CMake, a C++20 compiler, and a CUDA Python environment with
`training/requirements.txt`; Python selection follows the existing training jobs.
Set `INEZ_PYTHON` if automatic selection does not find that environment.

Results are written to
`results/evaluation/inezita-0.1.3/pilot-<jobid>/`. Check `manifest.json` for
`status: complete`, exactly 20 games, ten pairs, no faults, and unchanged model
checksums. It also records an initial projected cost for 2,000 games. The job
exits nonzero on incomplete or faulted results, and retains logs/PGNs for diagnosis.

After reviewing that pilot, use the same entry point for the 2,000-game baseline:

```sh
export PILOT_DIR="$PWD/results/evaluation/inezita-0.1.3/pilot-<jobid>"
sbatch --time=<wall-time-from-pilot> hpc/benchmark_inezita.slurm baseline
```

Replace the placeholders with the actual pilot directory and a Slurm wall time
that fits the measured cost, with margin. The default four-hour limit is for the
pilot and is not a promise that 2,000 games fit. The driver refuses a larger run
without a completed 20-game pilot and matching weights, code, binaries, device,
thread count, and opening source. Confirmation uses the `confirmation` argument,
a fresh opening slice and `180+1.8`. An interrupted tournament remains incomplete;
automatic resume is not implemented by this wrapper. Its Fastchess state is
retained for an explicitly audited recovery rather than silently starting over.

For the external reference gauntlet (Stockfish 17.1 at `Skill Level=0` / `128`
nodes, `UCI_Elo=1320`, and `UCI_Elo=1600`, with opt-in 1900/2200 presets, plus `lc0` when `LC0_BINARY` and
`LC0_WEIGHTS` are supplied), submit the dedicated gauntlet job:

```sh
sbatch hpc/gauntlet_inezita.slurm
```

By default this runs a 20-game pilot (ten opening pairs) against each reference
under `results/evaluation/inezita-0.1.3/gauntlet-pilot-<jobid>/<reference>/` and
writes an aggregated `summary.json`. To include Lc0 in the pilot gauntlet, export
`LC0_BINARY` and `LC0_WEIGHTS` (and optional `BENCH_OPPONENT_NODES=128` for an
equal-simulation track) before submitting. After reviewing the pilot gauntlet,
run the 1,000-game (`500`-pair) reference matches with:

```sh
export PILOT_GAUNTLET_DIR="$PWD/results/evaluation/inezita-0.1.3/gauntlet-pilot-<jobid>"
sbatch --time=<wall-time-from-pilot> hpc/gauntlet_inezita.slurm gauntlet
```

Overrides: `BENCH_MODEL` and `BENCH_OPPONENT` (artifact/run paths), `BENCH_REFERENCE`
(`stockfish-skill-0-n128`, `stockfish-elo-1320`, `stockfish-elo-1600`,
`inezita-material`, `lc0`), `BENCH_OPPONENT_ENGINE`, `BENCH_OPPONENT_NAME`,
`BENCH_OPPONENT_NODES`, `LC0_BINARY`, `LC0_WEIGHTS`, `GAUNTLET_REFERENCES`,
`OPENINGS_PGN` (a curated alternative PGN book), `FASTCHESS` (absolute executable
path), `BENCH_THREADS`, `BENCH_DEVICE`, `BENCH_PAIRS`, `BENCH_START` (zero-based
eligible opening offset), `BENCH_TC`, `BENCH_NODES` (an additional simulations
cap), and `BENCH_OUTPUT` (a fresh directory). When supplying a direct `BENCH_MODEL`
artifact for a checkpoint match, also supply `BENCH_OPPONENT`; automatic
previous-opponent selection needs a run's `accepted.json`. Keep production CUDA
results separate from CPU validation runs. With `BENCH_NODES`, the clock also
applies; this is an equal-cap test unless actual per-move node counts demonstrate
that clocks did not bind. Pure equal-simulation experiments need a generous clock
and a separate node-budget audit.

## GPU and signal troubleshooting

CUDA 13 PyTorch builds do not support Volta/V100 (`sm_70`). A V100 can still
appear available and accept copied model weights before the first kernel fails.
The benchmark defaults to an A100 40GB; resubmit the normal pilot command with
that default and a fresh output directory. The job does not reinstall packages
or alter the training environment.

If V100 execution is specifically required, use a separate environment with the
same PyTorch release's CUDA 12.6 build from
`https://download.pytorch.org/whl/cu126`, then explicitly select that interpreter
with `INEZ_PYTHON` and override `--constraint=v100`. Installing the same public
version over a `+cu130` build requires `--force-reinstall` (or an explicit
`+cu126` version); otherwise pip can consider the requirement already satisfied.
Run a fresh pilot after changing hardware or libraries. See
[PyTorch's CUDA support matrix](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337).

Native libraries can register signal handlers that Python represents as `None`.
The benchmark restores those to `SIG_DFL`, and always stops/reaps Fastchess
before restoring handlers. Cleanup must not hide a CUDA failure behind a
`signal handler must be ... callable` exception. Interrupted/failed PGNs are
retained, with their faults or incomplete state recorded. Do not resume the
failed V100 run as if it were a successful A100 pilot.

## Running the matches

Install and pin a Fastchess release/build on the compute host. Build Inezita on
that host; the macOS executable cannot be copied to Linux. Supply an audited
opening book and fixed artifact paths. The variables below are explicit inputs;
this lower-level example requires explicit paths. The submission job above supplies
its pinned default public book and selects the previous accepted checkpoint.

For example, a two-checkpoint practical-strength comparison is:

```sh
test -x "$FASTCHESS" && test -x "$INEZITA_BINARY"
test -d "$CANDIDATE_ARTIFACT" && test -d "$INCUMBENT_ARTIFACT"
test -f "$OPENINGS_PGN"
mkdir "$EVALUATION_DIR" # must be a fresh directory
cd "$EVALUATION_DIR"
"$FASTCHESS" \
  -engine "cmd=$INEZITA_BINARY" name=inezita-candidate \
    "option.ModelPath=$CANDIDATE_ARTIFACT" option.Device=cuda option.Threads=4 \
  -engine "cmd=$INEZITA_BINARY" name=inezita-incumbent \
    "option.ModelPath=$INCUMBENT_ARTIFACT" option.Device=cuda option.Threads=4 \
  -each proto=uci tc=60+0.6 \
  -openings "file=$OPENINGS_PGN" format=pgn order=sequential start=1 \
  -rounds 1000 -repeat -concurrency 1 \
  -report penta=true \
  -pgnout file=games.pgn append=false nodes=true timeleft=true \
  -config outname=fastchess-state.json \
  -log file=fastchess.log level=info engine=true \
  > tournament.out 2>&1
```

All paths supplied above must be absolute. The book must already contain curated
opening lines, not full completed games. Verify the intended unique pair count
and color reversal in the pilot before scaling. For the pilot use ten rounds;
for a 1,000-game external match use 500 rounds and replace the second engine
definition with the reference's pinned command/options. Engine-specific options
belong under that engine, not under `-each`.

For a future SPRT, add
`-sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 model=normalized` and set the predeclared
maximum to 10,000 rounds. Do not add SPRT to a fixed-count baseline measurement.
See the [Fastchess manual](https://github.com/Disservin/fastchess/blob/master/man.md)
for output and recovery settings. Audit saved state, artifact hashes, completed
opening pairs and engine settings before resuming; do not blindly append a new
tournament to a partial PGN. A scheduler wall-time limit is not a statistical
stopping decision.

On Rivanna, put preflight checks, environment activation and this command in a
Slurm batch job, initially requesting one GPU, eight CPU cores and 16 GiB host
memory. Set wall time from the measured pilot cost, including startup and log
writing; larger models may need more memory. Use independent jobs/output
directories for opponents. If sharding, assign non-overlapping opening pairs
and aggregate paired results, not reported per-shard Elo estimates. Small SPRT
tests are easiest to keep in one coordinator process. Submit using `sbatch`;
see [Slurm documentation](https://slurm.schedmd.com/sbatch.html).

Each run should retain a manifest with the protocol, checksums, exact command,
Git revisions and working-tree changes, Fastchess build/version, GPU/CPU details,
CUDA/PyTorch versions, clocks, precision, threads, opening slice, start/end time,
Slurm job ID and completion status. Keep `games.pgn`, tournament output,
Fastchess state, engine logs, and neural diagnostic reports. Recheck weights
hashes at the end. Store generated output under ignored
`results/evaluation/inezita-0.1.3/<unique-run>/` and check in only reviewed summaries.

## Active checks and legacy cleanup

`tools/evaluate_neural.py` remains the frozen behavioral diagnostic runner.
It already records weight and source checksums. Its known regression fixtures
and training acceptance checks are development evidence, not independent release
ratings. A new held-out tactical set can supplement match results.

The handcrafted-only `tools/match.py` has been removed. Its approximate Elo/SPRT
report and forced material/uniform engines must not be used to rate Inezita.
Native baseline legality/protocol coverage now lives in
`tests/baseline_uci_tests.py`; raw neural protocol assertions use the test-only
`tests/uci_client.py`. Neither emits strength estimates. The old baseline report
is retained as [historical engineering data](evaluation/handcrafted-baseline.md).

Data preparation, Inezita training, benchmark matches, and neural diagnostics
remain active workflows. CUDA execution checks run in the training preflight;
the standalone GPU smoke job and its overfit script have been removed.
