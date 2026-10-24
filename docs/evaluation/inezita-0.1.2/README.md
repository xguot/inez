# Frozen Inezita 0.1.2 baseline

These are observed pilot results from October 23, 2026. Both runs used the same
29,228,779-parameter `teacher/candidate-0020000` artifact, an A100 40GB, float32,
`60+0.6`, four PyTorch threads for Inezita and one thread for Stockfish 17.1.
The later run used optimized search code at `3edaf018e9dc5a5e11f739d27cec5475ed1f8637`.
Each matchup contains ten color-reversed opening pairs. The two runs reused the
same openings; later release evaluation must use a fresh slice.

| Reference | Earlier W/D/L | Optimized W/D/L |
| --- | --- | --- |
| Stockfish Skill 0, 128 nodes | 18/0/2 | 20/0/0 |
| Stockfish UCI_Elo 1320 | 13/2/5 | 15/0/5 |
| Stockfish UCI_Elo 1600 | 7/1/12 | 11/1/8 |

The optimized run scored 57.5% against the 1600 reference. These samples do not
establish a human rating or superiority over 1900/2200 opponents. No run had
reported faults or unfinished games. Model identity is pinned in `baseline.json`;
complete PGNs, selected openings and original manifests are retained beside each
run's summary. The large weights remain in ignored training storage; the model
checksum must accompany any copied artifact. The earlier `v0.1.2` tag points to
the version-bump commit, so reproduce this measured baseline with the explicit
code revision above rather than assuming that tag identifies these measurements.
