# Staged learning and Inezita (~29M) training on Rivanna

Human games initialize **Inezita** (`inezita`, our ~29.2M model; the name
**`inze`** is reserved for the future ~229M grand model). Later stages use
independent Stockfish targets and neural PUCT self-play. This reduces dependence
on human mistakes; it does not guarantee monotonic true playing strength.
Promotion requires game results and tactical regression checks.

## Inezita 0.1.3 continuation

Use [the 0.1.3 development and submission guide](inezita-0.1.3.md) for the new
full-corpus replay, EMA validation, curated 2400+ data filters, and paired
promotion protocol targeting the 2200–2400 reference range. Old accepted
artifacts and results remain intact.

## Submit

Use Python 3.12 or newer with `training/requirements.txt` installed. On Rivanna,
use the frozen submission entry point (`bash hpc/submit_inezita.sh`, backed by
`tools/submit_training.py`) with an explicit mode (`continue`, `scratch`, or
`resume`):

```sh
export INIT_ARTIFACT="$PWD/results/training/inezita/teacher/candidate-0020000"
export TRAIN_CONFIG=configs/train/inezita-0.1.3.json
export DATA_DIR="$PWD/data/processed/inezita-0.1.3"
export RUN_DIR="$PWD/results/training/inezita-0.1.3"
bash hpc/submit_inezita.sh continue --prepare
```

The submitter snapshots all training sources and configurations under
`RUN_DIR/source/`, writes a pinned `RUN_DIR/job.json`, and schedules CPU
preparation (`hpc/prepare_inezita.slurm`, when `--prepare` is passed) followed
by a dependent single-GPU training job (`hpc/train_inezita.slurm`) under
`nssac_students`. Set `INEZ_ACCOUNT` to override the account. Training requests
one A100 40GB GPU, 16 CPU cores, 96 GiB host memory and eight hours. CUDA is
required; there is no automatic CPU fallback.

Preparation reads the September, October, and November 2025 Lichess Elite
archives. The 0.1.3 configurations (`configs/train/inezita-0.1.3.json` and
`configs/train/inezita.json`) import 100,000 completed 2400+ standard games with
`Normal` termination and `base_seconds + 40 * increment_seconds >= 260`
(excluding zero-increment blitz and time forfeits, while sampling up to 16
positions plus 2 off-policy tactical blunder/recapture counterfactual pairs per
training game from ply 0), and merge all 5,607 unique positions from the
3,871-line Modern Chess Openings corpus (`data/openings/mco_openings.tsv`,
covering ECO Volumes A–E) into the training split. Validation and evaluation
splits remain strictly disjoint and post-opening so common opening positions are
never excluded from training or self-play. If Stockfish is unavailable, it
caches the official 17.1 release for Linux x86-64 or Apple Silicon. Set
`INEZ_STOCKFISH=/absolute/path/to/stockfish` to use an existing engine;
`INEZ_STOCKFISH_MODULE` can load a site module first. If an Ubuntu release
binary cannot run against the cluster's older libc, the preparer rebuilds the
bundled release source using `make` and the loaded GCC. The Makefile downloads
the release's NNUE files when needed. Offline clusters need the archives and
engine copied into place beforehand. Downloads stay in ignored `data/raw/`.
Observed SHA256 hashes are recorded; the Elite publisher provides no archive
checksum here.

Sources: [Lichess Elite](https://database.nikonoel.fr/), underlying
[Lichess standard games and CC0 terms](https://database.lichess.org/#standard_games),
[Lichess ECO/MCO chess-openings (CC0)](https://github.com/lichess-org/chess-openings),
[Stockfish 17.1 release/source](https://github.com/official-stockfish/Stockfish/releases/tag/sf_17.1).

Overrides exported before submission:

- `TRAIN_CONFIG`: default `configs/train/inezita-0.1.3.json` (`continue`) or `configs/train/inezita.json` (`scratch`).
- `DATA_DIR`: default `data/processed/inezita-0.1.3`.
- `RUN_DIR`: default `results/training/inezita-0.1.3` (`continue`) or `results/training/inezita-0.1.3-scratch` (`scratch`).
- `INIT_ARTIFACT`: accepted Inezita artifact to improve; required for `continue` and forbidden for `scratch`.
- `RESUME_MODE`: `exact` (default) or `portable` to continue across GPU/PyTorch changes.

Use a fresh data/run directory when changing the experiment. Inezita
(`configs/models/inezita.json`) has 29,228,779 parameters. The older 4M smoke
weights have different tensor shapes and cannot initialize her.

## Curriculum and isolation

1. **Human + MCO + tactical initialization:** played-move policy, MCO opening
   repertoire distributions, off-policy blunder/recapture counterfactuals, and
   side-to-move game-result WDL, for 20,000 AdamW steps (batch size 128) at
   learning rate 3e-4.
2. **Stockfish correction:** replace human targets, including bad moves and
   off-policy blunder counterfactuals, with fresh analysis (`label_contract: 2`,
   issuing a fresh `ucinewgame` token per position) across a parallel worker
   pool of independent single-threaded Stockfish 17.1 instances with
   move-stack-aware transposition caching. Defaults: 50,000 nodes, MultiPV 4,
   native `UCI_ShowWDL`. Policy is a softmax over each line's
   centipawn-tiebroken `(win-loss)/temperature`, with temperature 0.15 (blended
   with MCO book priors on `mco-*` opening records). Value is the best line's
   full WDL. Defaults: 12,000 steps (batch size 128) at learning rate 2e-5.
   Engine version, binary checksum, analysis budget and target transformation
   are recorded.
3. **Self-play:** the accepted model plays complete games with neural PUCT
   equipped with the MCO opening book, exact Static Exchange Evaluation (SEE)
   piece-safety priors, and bare-king endgame conversion shaping. Policy
   targets are normalized root visits; WDL comes from actual outcomes (filtering
   out any botched decisive-material draw). Defaults: four rounds of 256 games,
   256 simulations per move, and 1,000 training steps per round at learning rate
   5e-6 with 80% teacher replay drawn from the complete verified teacher
   training pool. Human imitation labels are never reintroduced. Root Dirichlet
   noise and early-game move sampling add exploration. Subsequent runs with
   `INIT_ARTIFACT` skip human initialization. Each new self-play round
   initializes optimization from the accepted model recorded in that round's
   frozen source metadata. A rejected tail checkpoint from teacher training or a
   previous round cannot initialize a new round. An interrupted round continues
   from its own optimizer checkpoint.

Games are split 80/10/10 by trajectory. Duplicate trajectories stay together.
Positions overlapping held-out tactical boards or another split are removed
from training. Reserved validation/evaluation/tactical board identities are
excluded from future self-play targets. Features reconstruct only the prefix,
preserving history without exposing future game moves.

Games that hit the self-play move cap without a rules-defined result are
discarded. A round with no completed games stops instead of inventing draw
labels. Earlier accepted models remain available for diagnosis and improvement.

## Evaluation and acceptance

Every exported candidate is tested with the **Python neural PUCT reference**
against up to two previously accepted versions. The native UCI engine still
uses heuristic evaluators; native neural inference integration is future work.
The trained Python player is available to chess GUIs through the
[`build/inezita` UCI engine](neural-uci.md).
Both models receive equal search budgets and paired openings with colors
reversed.

Promotion requires at least 55% score, complete color-reversed opening pairs,
an exact one-sided pair-level randomization test, and no tactical regression.
The significance threshold is allocated across the planned exported candidates.
This is a fixed-sample screening test under pair-level exchangeability, not an
Elo confidence interval. Uncertain results retain the incumbent. Older saved
Wilson-style reports remain readable; they cannot authorize a new promotion.
The first reference model is the initialized network, stored as `initial`.

`tests/fixtures/independent_tactics.json` contains six hand-authored regression
boards excluded from all training. Expand this small starting suite before
making strength claims. Since gates inspect it regularly, reserve another
untouched final tactical set for later research comparisons.

`candidate-*` artifacts are proposals. Only `RUN_DIR/accepted.json` identifies
the accepted model and history. Rejected candidates cannot replace it. Gate
reports retain complete game move lists, results, opponents and per-position
tactical outcomes. The corpus evaluation split stays reserved from optimization.

`"promoted": false` means the candidate did not pass every acceptance condition.
Training can finish successfully with exit 0 while retaining an earlier accepted
checkpoint. Gate output includes match scores, completed/unfinished games,
tactical regressions and rejection reasons. `insufficient_paired_evidence` means the paired p-value exceeds the allocated
threshold even if raw score is sufficient. Legacy reports may instead contain
`insufficient_match_evidence`, referring to their former score-bound screen.

Inspect an existing run on a login node without allocating a GPU:

```sh
"${INEZ_PYTHON:-$HOME/.conda/envs/inez/bin/python}" tools/train_inezita.py status \
    --run-dir results/training/inezita-0.1.3
```

This reads saved reports, including reports from older runs, and leaves run
files untouched. A `complete` run has no further optimizer steps to perform;
resubmitting the same directory does not start a new training round. To improve
the accepted model further, initialize a new run as described below. The gate
report provides the evidence needed before tuning training or search budgets.

## Checkpoint and resume

`last.pt` is an atomic training-state checkpoint: model, AdamW state, completed
step, minibatch/Python/PyTorch/CUDA RNGs, data fingerprints, configuration,
source revision and hardware context. Long stages (`steps >= 100`) apply a
deterministic linear warmup followed by cosine learning-rate decay to 10% of the
stage peak LR. Portable model artifacts contain EMA weights for stages with
100 or more steps. Candidate validation reloads these exact bytes and records
separate policy/value metrics; optimizer checkpoints retain raw training weights.

The job forwards Slurm's 120-second warning and termination signals to the
trainer. It checkpoints after an optimizer step and exits 75. Both self-play
and promotion gates commit atomic per-game chunks so resumed jobs continue from
the last completed game boundary; a partial evaluation cannot promote a model.
Completed self-play experiences are frozen with hashes so resumed optimization
consumes the same examples.

Resume an interrupted 0.1.3 run using its frozen `RUN_DIR/job.json` specification:

```sh
export RUN_DIR="$PWD/results/training/inezita-0.1.3"
bash hpc/submit_inezita.sh resume
```

Configuration or data changes require a fresh directory. Exact resume requires
the same numerical environment and visible CUDA device count; another GPU
architecture can change floating-point behavior. The job does not automatically
requeue or schedule indefinite training.

If Slurm allocates a different GPU model, continue the **same run** explicitly:

```sh
export RUN_DIR="$PWD/results/training/inezita-0.1.3"
export RESUME_MODE=portable
bash hpc/submit_inezita.sh resume
```

Portable resume preserves the saved model, AdamW state, completed step, sampler
order, CPU/Python RNGs, frozen self-play records and promotion history. It still
rejects changed configuration or data. When the numerical environment differs,
CUDA RNGs are reseeded with `(training seed + checkpoint step) mod 2^64` rather
than loading device-specific RNG state. Subsequent results are not bitwise
identical to an uninterrupted run. Environment differences are printed in
`run.log` and retained in subsequent training checkpoints' `resume_history`.
Completed curriculum stages validate their step and data/configuration signature
before being skipped. Existing 0.1.2 self-play optimizer checkpoints used a different
replay pool and must not resume under 0.1.3 code: finish them on their original
revision or start the new continuation from an accepted artifact and fresh
`RUN_DIR`. Model artifacts and old status reports remain readable. Keep
`RESUME_MODE=portable` for compatible submissions on mixed GPU nodes.

For later improvement, set `INIT_ARTIFACT` to the `artifact` value in an accepted
checkpoint's `accepted.json` and select a new `RUN_DIR`. The new run performs
teacher correction and new self-play, with no human imitation phase.
Initialization is restricted to artifacts listed in an earlier run's accepted
history; a rejected candidate cannot bypass promotion by starting a new run.

Logs: `inezita_prepare_<job>.out/.err`, `inezita_train_<job>.out/.err`, and
`RUN_DIR/run.log`. Inspect `validation-*`, `gate-*`, `accepted.json`, and the
recorded actual device alongside loss curves. Failure to promote means the
run has not established a stronger playing model, even if training loss fell.

## References

- Monroe et al. (ICLR 2026), [*Chessformer: A Unified Architecture for Chess Modeling*](https://arxiv.org/abs/2605.19091) — 64-square token representation, dynamic Geometric Attention Bias (GAB), `from–to` query-key policy head with promotion offsets, and 5% history-frame masking.
- Braun (2026), [*Engineering Efficient Self-Play Chess: Search, Replay, and Throughput Under Limited Compute*](https://arxiv.org/abs/2609.37447) — dynamic reduced-parent FPU, per-ply search/outcome discounting, outcome + root-WDL target blending, and 30/70 policy-surprise prioritized sampling.
- Weichart (2026), [*Variance-Aware Prior-Based Tree Policies for Monte Carlo Tree Search*](https://arxiv.org/abs/2512.21648) — Welford $O(1)$ online variance tracking and Bernstein `PUCT-V` exploration.
- Farebrother et al. (ICML 2024), [*Stop Regressing: Training Value Functions via Classification for Scalable Deep RL*](https://arxiv.org/abs/2403.03950) & Ruoss et al. (NeurIPS 2024), [*Amortized Planning with Large-Scale Transformers: A Case Study on Chess*](https://arxiv.org/abs/2402.04494) — ordinal `HL-Gauss` value target smoothing and 15–16 layer transformer depth scaling.
- Morales-Brotons et al. (TMLR 2024), [*Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits*](https://arxiv.org/abs/2411.18704) — non-bootstrapped shadow EMA weight averaging (`period = 16`, warmup decay to `0.992`).
- Helfenstein et al. (IEEE ToG 2026), [*Checkmating One, by Using Many: Combining Mixture of Experts With MCTS to Improve in Chess*](https://doi.org/10.1109/TG.2026.3659995) — phase-aware endgame curriculum stratification.
