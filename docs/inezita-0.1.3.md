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
  `hpc/prepare_inezita.slurm` (32 CPU cores, 96 GiB RAM, 12h) followed
  automatically by `hpc/train_inezita.slurm` (`--dependency=afterok`, 1 A100
  40GB GPU, 16 CPU cores, 96 GiB RAM, 12h).
- **Automatic resume**: If the job hits the Slurm wall-clock limit and saves an
  atomic checkpoint (`exit 75`), running the exact same command
  (`bash hpc/submit_inezita.sh`) detects `results/training/inezita-0.1.3/job.json`
  and resumes from the latest checkpoint automatically.

## Dataset & Curriculum (`configs/train/inezita.json`)

1. **Pure Classical OTB + Superhuman Engine Corpus (`50,000` games, `>= 20 min` per side)**:
   - Combines `TCEC-all-in-one-compact.zip` ([TCEC Season 1–29 Final](https://github.com/TCEC-Chess/tcecgames/releases/tag/S29-final), 46,667 classical engine championship games at 2500–3800+ Elo) with 15 monthly [Lichess FIDE OTB Broadcast](https://database.lichess.org/broadcast/) archives (`lichess_db_broadcast_2025-07.pgn.zst` through `2026-09.pgn.zst`, 16,262 classical OTB games at 2400+ FIDE).
   - Enforces `min_rating: 2400`, `min_time_control_seconds: 1200` (at least 20 minutes per side, rejecting all blitz and rapid games), and `require_normal_termination: true` (rejecting all time forfeits, defaults, and abandoned games).
   - Samples up to 16 positions + 2 off-policy tactical blunder/recapture counterfactuals per training game and merges all 5,607 unique positions from the 3,871-line Modern Chess Openings corpus (`data/openings/mco_openings.tsv`).
2. **Stage 1 — Classical Prior + MCO + Tactical Counterfactuals (`20,000` steps)**:
   - Trains for 20,000 AdamW steps (`batch_size: 128`, peak `lr: 3e-4`, linear warmup + cosine decay, non-bootstrapped shadow EMA `decay=0.992`).
3. **Stage 2 — 25,000-Node Stockfish 17.1 Distillation (`30,000` steps)**:
   - Relabels every position (~850,000 records) using 32 parallel single-threaded Stockfish 17.1 workers at **25,000 nodes per position** (`MultiPV 4`, `temperature: 0.15`, `label_contract: 2` with fresh `ucinewgame` isolation).
   - Trains for 30,000 steps (`batch_size: 128`, `teacher_lr: 1e-4`), evaluating EMA candidate checkpoints every 5,000 steps.
4. **Stage 3 — Neural PUCT Self-Play with 80% Full-Corpus Teacher Replay (`4` rounds)**:
   - 4 rounds of 256 games at 256 simulations/move, 1,000 training steps per round (`selfplay_lr: 1.5e-5`) with 80% full-corpus teacher replay.
   - Candidates are gated by 32 color-reversed opening pairs (64 games) at 128 simulations/move (`minimum_score: 0.55`, exact pair-level randomization test at `familywise_alpha: 0.05`) plus the held-out tactical regression suite.
