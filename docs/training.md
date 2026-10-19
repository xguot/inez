# Staged learning and the 29M Rivanna pilot

Human games initialize Inez. Later stages use independent Stockfish targets and
neural PUCT self-play. This reduces dependence on human mistakes; it does not
guarantee monotonic true playing strength. Promotion requires game results and
tactical regression checks.

## Submit

Use Python 3.12 or newer with `training/requirements.txt` installed. On Rivanna:

```sh
export INEZ_PYTHON="$HOME/.conda/envs/inez/bin/python"
bash hpc/submit_29m.sh
```

The submitter schedules CPU preparation and a dependent single-GPU training
job under `nssac_students`. Set `INEZ_ACCOUNT` to override the account. Training
requests one GPU, eight CPU cores, 64 GiB host memory and 24 hours. CUDA is
required; there is no automatic CPU fallback. After preparation, a particular
available GPU can be requested with a site-supported `--gres` override when
submitting `hpc/train_29m.slurm` directly.

Preparation downloads the November 2025 Lichess Elite archive (about 77 MiB),
imports at most 1,000 completed standard games, and samples up to 16 positions
per game. If Stockfish is unavailable, it caches the official 17.1 release for
Linux x86-64 or Apple Silicon. Set `INEZ_STOCKFISH=/absolute/path/to/stockfish`
to use an existing engine; `INEZ_STOCKFISH_MODULE` can load a site module first.
If an Ubuntu release binary cannot run against the cluster's older libc, the
preparer rebuilds the bundled release source using `make` and the loaded GCC.
The Makefile downloads the release's NNUE files when needed.
Offline clusters need the archive and engine copied into place beforehand.
Downloads stay in ignored `data/raw/`. Observed SHA256 hashes are recorded;
the Elite publisher provides no archive checksum here.

Sources: [Lichess Elite](https://database.nikonoel.fr/), underlying
[Lichess standard games and CC0 terms](https://database.lichess.org/#standard_games),
[Stockfish 17.1 release/source](https://github.com/official-stockfish/Stockfish/releases/tag/sf_17.1).

Overrides exported before submission:

- `TRAIN_CONFIG`: default `configs/train/inez-29m.json`.
- `DATA_DIR`: default `data/processed/elite-29m-pilot`.
- `RUN_DIR`: default `results/training/inez-29m-pilot`.
- `INIT_ARTIFACT`: accepted 29M artifact to improve; skips human initialization.

The default is a bounded engineering pilot. Use a fresh data/run directory when
changing the experiment. The model has 29,228,779 parameters. The older 4M smoke
weights have different tensor shapes and cannot initialize it.

## Curriculum and isolation

1. **Human initialization:** played-move policy and side-to-move game-result
   WDL, for 2,000 AdamW steps at learning rate 3e-4.
2. **Stockfish correction:** replace both human targets, including bad moves,
   with fresh analysis using the sampled position and its history. Defaults:
   10,000 nodes, MultiPV 4, native `UCI_ShowWDL`. Policy is a softmax over each
   line's `(win-loss)/temperature`, with temperature 0.15. Value is the best
   line's full WDL. These are teacher score targets, not Stockfish visit counts.
   Defaults: 2,000 steps at learning rate 1e-4. Engine version, binary checksum,
   analysis budget and target transformation are recorded.
3. **Self-play:** the accepted model plays complete games with neural PUCT.
   Policy targets are normalized root visits; WDL comes from actual outcomes.
   Defaults: four games, 32 simulations per move, and 500 training steps.
   Teacher examples comprise 50% of replay samples; human imitation labels are
   never reintroduced. Root Dirichlet noise and early-game move sampling add
   exploration. Subsequent runs with `INIT_ARTIFACT` skip human initialization.

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
Both models receive equal search budgets and paired openings with colors
reversed.

Promotion requires at least 55% score, a Wilson-style score screen above 50%,
all match games finishing, and no loss of a previously solved tactical board.
This is an engineering screen, not an exact Elo confidence interval or proof
against all regressions. Small matches have limited statistical power. Ties
and uncertain results retain the incumbent. The first reference model is the
initialized network, stored explicitly as `initial`.

`tests/fixtures/independent_tactics.json` contains six hand-authored regression
boards excluded from all training. Expand this small starting suite before
making strength claims. Since gates inspect it regularly, reserve another
untouched final tactical set for later research comparisons.

`candidate-*` artifacts are proposals. Only `RUN_DIR/accepted.json` identifies
the accepted model and history. Rejected candidates cannot replace it. Gate
reports retain complete game move lists, results, opponents and per-position
tactical outcomes. The corpus evaluation split stays reserved from optimization.

## Checkpoint and resume

`last.pt` is an atomic training-state checkpoint: model, AdamW state, completed
step, minibatch/Python/PyTorch/CUDA RNGs, data fingerprints, configuration,
source revision and hardware context. Each stage uses its configured constant
learning rate. Portable model artifacts contain weights only.

The job forwards Slurm's 120-second warning and termination signals to the
trainer. It checkpoints after an optimizer step and exits 75. Gates and
self-play generation may be recomputed after interruption; a partial evaluation
cannot promote a model. Completed self-play experiences are frozen with hashes
so resumed optimization consumes the same examples.

Resume with the same environment, configuration, data and persistent `RUN_DIR`:

```sh
sbatch hpc/train_29m.slurm
```

Configuration or data changes require a fresh directory. Exact resume requires
the same numerical environment and visible CUDA device count; another GPU
architecture can change floating-point behavior. The job does not automatically
requeue or schedule indefinite training.

For later improvement, set `INIT_ARTIFACT` to the `artifact` value in an accepted
checkpoint's `accepted.json` and select a new `RUN_DIR`. The new run performs
teacher correction and new self-play, with no human imitation phase.
Initialization is restricted to artifacts listed in an earlier run's accepted
history; a rejected candidate cannot bypass promotion by starting a new run.

Logs: `inez_prepare_<job>.out/.err`, `inez_train_<job>.out/.err`, and
`RUN_DIR/run.log`. Inspect `validation-*`, `gate-*`, `accepted.json`, and the
recorded actual device alongside loss curves. Failure to promote means the
pilot has not established a stronger playing model, even if training loss fell.
