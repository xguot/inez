# Training data

Raw downloads belong in `data/raw/`; generated datasets belong in
`data/processed/`. Both directories are ignored by Git. Source data stays
separate from code and model configuration.

## Classical FIDE OTB Broadcast + Superhuman TCEC Championship Corpus (Inezita 0.1.3)

Inezita (~29.2M parameters) trains exclusively on **Classical Over-The-Board
(OTB) FIDE tournament games** and **Superhuman TCEC Engine Championship games**
(`min_time_control_seconds: 1200`, i.e., at least **20 minutes per side**,
rejecting all blitz and short rapid games) where both players are rated
**2400+** (`min_rating: 2400`) and the game ended by standard rules or tablebase
adjudication (`require_normal_termination: true`, excluding all time forfeits,
defaults, and abandoned games).

Across `data/raw/tcec/TCEC-all-in-one-compact.zip` (all Top Chess Engine
Championship games up to Season 29 Final, where 2500–3800+ Elo engines play
30m+5s to 120m+30s classical time controls) and 15 monthly [Lichess FIDE OTB Broadcast](https://database.lichess.org/broadcast/)
archives (`lichess_db_broadcast_2025-07.pgn.zst` through `2026-09.pgn.zst`,
covering official 90m+30s and 120m+30s FIDE events), there are **62,929**
qualifying classical 2400+ games (46,667 TCEC engine championship games and
16,262 FIDE OTB 2400+ classical grandmaster games).

`configs/train/inezita.json` imports **50,000** of these classical games
(interleaving all 16,262 FIDE OTB 2400+ classical games with ~33,738 TCEC
superhuman engine games), sampling up to 16 positions + 2 off-policy
tactical blunder/recapture counterfactuals per training game, and merges all
5,607 unique positions from the 3,871-line Modern Chess Openings corpus
(`data/openings/mco_openings.tsv`). Every position (~850,000 total) is then
relabeled by a parallel pool of independent Stockfish 17.1 workers at **25,000
nodes per position** (`MultiPV 4`, native `UCI_ShowWDL`, `label_contract: 2`).

- **TCEC Games Archive (Season 29 Final)**: https://github.com/TCEC-Chess/tcecgames/releases/tag/S29-final (`TCEC-all-in-one-compact.zip`, CC-BY-SA-3.0)
- **Official FIDE OTB Broadcasts**: https://database.lichess.org/broadcast/ (`lichess_db_broadcast_2025-07.pgn.zst` .. `lichess_db_broadcast_2026-09.pgn.zst`, CC-BY-SA-4.0)
- **ECO / Modern Chess Openings corpus**: `data/openings/mco_openings.tsv` (CC0-1.0, derived from https://github.com/lichess-org/chess-openings)
- **Teacher oracle**: [Stockfish 17.1](https://github.com/official-stockfish/Stockfish/releases/tag/sf_17.1) (`data/raw/stockfish-17.1/`, GPL-3.0)

Observed SHA256 hashes of the cached archives in `data/raw/`:

| Archive | Classical 2400+ (`>= 1200s`) Games | SHA256 |
| --- | ---: | --- |
| `tcec/TCEC-all-in-one-compact.zip` | 46,667 | `eddbdef04398e4da096c933094c41a34ca3a1237c54b10c8ca84bd5722370591` |
| `broadcast/lichess_db_broadcast_2026-09.pgn.zst` | 453 | `27a39d035767a30901b89602f33f2a7b27d4ab5025842071165fa079f8fff1df` |
| `broadcast/lichess_db_broadcast_2026-08.pgn.zst` | 1,178 | `e227c35c3207ebade754849c1825982e2717ae82c013013d896778e27be77724` |
| `broadcast/lichess_db_broadcast_2026-07.pgn.zst` | 830 | `714d0eb99f99fca8d791142038b6c59b5ca6a51b3339bd3891a92f4bdffcbf0c` |
| `broadcast/lichess_db_broadcast_2026-06.pgn.zst` | 1,278 | `65905d611c0d78d90ab57cd9edd3821f4119b46c22b5489fd7e265a74c6767b3` |
| `broadcast/lichess_db_broadcast_2026-05.pgn.zst` | 975 | `d212987f7972a70268809ff6c726acff8bf0987161aacd7647b160f8564f6a53` |
| `broadcast/lichess_db_broadcast_2026-04.pgn.zst` | 1,084 | `97b036f3a3639ae3be59c1508b1e18c76ac1b2e8fada4587d1b5c5a93c438d5d` |
| `broadcast/lichess_db_broadcast_2026-03.pgn.zst` | 841 | `1a5ad83fc9f440dcdadde17b11cc81a953f4895aaf8b8588470b3078209dc7fc` |
| `broadcast/lichess_db_broadcast_2026-02.pgn.zst` | 940 | `ea977569917718b33940ba5379db2adad77d58876c29084294d357f15fe6a31b` |
| `broadcast/lichess_db_broadcast_2026-01.pgn.zst` | 984 | `9abf10d106a8bf5ecd41359825b3bf55c2f124c8a790a380cb98844c2cca5dca` |
| `broadcast/lichess_db_broadcast_2025-12.pgn.zst` | 1,207 | `fadbe80ee65e9f8499578e934d7600ece69d09b1cf8abc75fdd3bd840f61e63f` |
| `broadcast/lichess_db_broadcast_2025-11.pgn.zst` | 1,200 | `39deb43faf6e2f4569a76f369ca0a3b87aa7787330c9e2ff87705c6f3bf32ff1` |
| `broadcast/lichess_db_broadcast_2025-10.pgn.zst` | 1,064 | `77be4c998ca8d3b303928e1619f0f9dc2a03a71d5ab6a59d02d42291ae0524e3` |
| `broadcast/lichess_db_broadcast_2025-09.pgn.zst` | 1,496 | `da3a261e926a2c0f5e79c393818e643d76448edf94cfdaefa3892258a6f9e7ba` |
| `broadcast/lichess_db_broadcast_2025-08.pgn.zst` | 1,993 | `f925d1f3b494517c8eae223976a886bed77a7b8627b17ec287e50cdda766dae8` |
| `broadcast/lichess_db_broadcast_2025-07.pgn.zst` | 739 | `1ffc6b775d27631ac988dd816a137b6dd0ec05561aeee8b45861e0c08dbcb10b` |

## Preparation before training

`bash hpc/submit_inezita.sh` automatically schedules `hpc/prepare_inezita.slurm`
followed by `hpc/train_inezita.slurm`. Preparation downloads any missing TCEC or
FIDE OTB Broadcast archives into `data/raw/`, verifies their SHA256 provenance,
imports the 50,000 filtered Classical games into version-1 records, and
generates Stockfish 17.1 teacher labels (`label_contract: 2`).
