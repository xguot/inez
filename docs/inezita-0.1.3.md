# Inezita 0.1.3 Training Guide

Inezita (`configs/models/inezita.json`, 29,228,779 parameters) is trained from
scratch on a curated **Classical FIDE OTB Broadcast + Superhuman TCEC Engine
Championship** corpus (`configs/train/inezita.json`), completely eliminating
online blitz and short rapid games.

## Single-Command Rivanna Submission

Submit and resume the entire 0.1.3 pipeline on Rivanna with a single command:

```sh
bash hpc/submit_inezita.sh
```

- **First run**: Creates a frozen code/config snapshot under
  `results/training/inezita-0.1.3/source/`, writes
  `results/training/inezita-0.1.3/job.json`, and submits
  `hpc/prepare_inezita.slurm` (48 CPU cores, 96 GiB RAM, 24h) followed
  automatically by `hpc/train_inezita.slurm` (`--dependency=afterok`, 1 A100
  40GB GPU, 16 CPU cores, 96 GiB RAM, 36h).
- **Automatic resume**: If the job hits the Slurm wall-clock limit and saves an
  atomic checkpoint (`exit 75`), running the exact same command
  (`bash hpc/submit_inezita.sh`) detects `results/training/inezita-0.1.3/job.json`
  and resumes from the latest checkpoint automatically.

Both shell and scheduler entry points are supported. From the checkout root:

```sh
bash hpc/submit_inezita.sh --dry-run
sbatch hpc/submit_inezita.sh
```

The scheduler wrapper uses a short CPU job (`standard`, four cores, 16 GiB,
ten minutes) to submit the preparation and A100 training jobs. It resolves the
checkout through `SLURM_SUBMIT_DIR`; the script copied into Slurm's spool is
never treated as the repository. `INEZ_ROOT` can explicitly select a checkout.
The account remains `nssac_students`.

Run directories are preserved, including incomplete attempts. A nonempty run
without `job.json` requires inspection and a fresh `RUN_DIR`; neither normal
submission nor `--dry-run` deletes it. A completed curriculum queues no more
work. Resume validates complete prepared data and skips another preparation job
when possible. Queued/running jobs for the same run block duplicate submission.

Child job logs are retained in `RUN_DIR/slurm/{prepare,train}-<jobid>.out/.err`;
job IDs and submission failures are recorded in `RUN_DIR/submissions.json`.
Dispatcher logs are stored in `hpc/logs/inezita_submit_<jobid>.out/.err`.
Directly submitted preparation, training, benchmark and gauntlet jobs also use
`hpc/logs/`. Submit with `sbatch` from the checkout root. The tracked
`hpc/logs/.gitkeep` ensures the directory exists before Slurm opens its logs;
creating it inside the batch script would be too late. `bash hpc/sync.sh` pulls
these logs along with run results. Already queued jobs retain their original paths.
Python, configuration, source hashes and selected preparation tools
are pinned in `job.json`. Portable resume reaches both preflight and training:

```sh
RESUME_MODE=portable bash hpc/submit_inezita.sh
```

The configured `.pgn.zst` archives require Python 3.14's built-in decoder or the
`zstd` command (available on Rivanna). Dependency checks happen before downloads.
Cached and downloaded archives are fully decoded to reject truncated frames;
a decoder failure cannot be accepted as the end of a valid corpus. The training
GPU preflight runs both CUDA arithmetic and a model forward/backward pass.

The dispatcher creates the snapshot before queuing jobs. Existing frozen runs
continue using their saved code; use a fresh `RUN_DIR` to adopt these fixes if a
previous `job.json` was already created with an older implementation.

## Time limits and queue states

The preparation `24:00:00` and GPU training `36:00:00` settings are maximum
allocation runtimes, not durations the jobs must consume or reliable estimates.
Preparation exits as soon as it finishes. `afterok` then makes the GPU job
eligible immediately; GPU scheduling can still add queue time. `PD (Resources)`
means waiting for hardware, while `PD (Dependency)` means waiting for successful
preparation. Neither state is active training time. Pressing Ctrl-C on `squeue`
does not cancel the queued jobs.

Total turnaround is queue time plus actual preparation, optimization, self-play
and promotion matches. This larger scratch experiment has no measured end-to-end
runtime yet and can require multiple allocations. Shorter limits may help the
scheduler fit a job but can also interrupt it; choose them from observed progress.
Completed teacher chunks, games and optimizer checkpoints survive supported
resumes. A failed/timed-out preparation does not release its GPU dependency.

The submitter now accepts validated overrides and records the effective ceilings:

```sh
PREP_TIME_LIMIT=08:00:00 TRAIN_TIME_LIMIT=24:00:00 bash hpc/submit_inezita.sh
```

Those values illustrate the syntax, not a promise the run fits them. New
submissions inherit the saved ceilings unless explicitly overridden on resume.
Changes to code or these environment variables do not alter already queued jobs.
The default ceilings remain 24 hours and 36 hours. `--dry-run` displays the limits
without submitting work.

## Dataset & Curriculum (`configs/train/inezita.json`)

1. **Pure Classical OTB + Superhuman Engine Corpus (`50,000` games, `>= 20 min` per side)**:
   - Combines `TCEC-all-in-one-compact.zip` ([TCEC Season 1–29 Final](https://github.com/TCEC-Chess/tcecgames/releases/tag/S29-final), 46,667 classical engine championship games at 2500–3800+ Elo) with 15 monthly [Lichess FIDE OTB Broadcast](https://database.lichess.org/broadcast/) archives (`lichess_db_broadcast_2025-07.pgn.zst` through `2026-09.pgn.zst`, 16,262 classical OTB games at 2400+ FIDE).
   - Enforces `min_rating: 2400`, `min_time_control_seconds: 1200` (at least 20 minutes per side, rejecting all blitz and rapid games), and `require_normal_termination: true` (rejecting all time forfeits, defaults, and abandoned games).
   - Samples up to 16 positions + 2 off-policy tactical blunder/recapture counterfactuals per training game and merges all 5,607 unique positions from the 3,871-line Modern Chess Openings corpus (`data/openings/mco_openings.tsv`).
2. **Stage 1 — Classical Prior + MCO + Tactical Counterfactuals (`20,000` steps)**:
   - Trains for 20,000 AdamW steps (`batch_size: 128`, peak `lr: 3e-4`, linear warmup + cosine decay, non-bootstrapped shadow EMA `decay=0.992`).
3. **Stage 2 — 25,000-Node Stockfish 17.1 Distillation (`30,000` steps)**:
   - Relabels every position (count recorded after preparation) using up to 48 parallel single-threaded Stockfish 17.1 workers at **25,000 nodes per position** (`MultiPV 4`, `temperature: 0.15`, `label_contract: 2` with fresh `ucinewgame` isolation).
   - Trains for 30,000 steps (`batch_size: 128`, `teacher_lr: 1e-4`), evaluating EMA candidate checkpoints every 5,000 steps.
4. **Stage 3 — Neural PUCT Self-Play with 80% Full-Corpus Teacher Replay (`4` rounds)**:
   - 4 rounds of 256 games at 256 simulations/move, 1,000 training steps per round (`selfplay_lr: 1.5e-5`) with 80% full-corpus teacher replay.
   - Candidates are gated by 32 color-reversed opening pairs (64 games) at 128 simulations/move (`minimum_score: 0.55`, exact pair-level randomization test at `familywise_alpha: 0.05`) plus the held-out tactical regression suite.
