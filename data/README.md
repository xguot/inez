# Training data

Raw downloads belong in `data/raw/`; generated datasets belong in
`data/processed/`. Both directories are ignored by Git. Source data stays
separate from code and model configuration.

## Starter corpus

The official Lichess January 2013 standard-rated PGN archive is a small,
complete source for developing the importer and testing the training pipeline.
It is an older, unfiltered corpus, not the selected dataset for large training
runs. The archive contains 121,332 games and is approximately 17.8 MB compressed.
An initial header scan found only 616 games with both players rated at least
2000, and only four of those with `base_seconds + 40 * increment_seconds >= 600`.
This makes it useful as an importer fixture, but unsuitable for the proposed
strong-player rapid/classical training pilot. Full game legality has not yet
been validated.

- Source: https://database.lichess.org/#standard_games
- Archive: https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
- License: CC0-1.0, as stated by the source database.
- Published checksums: https://database.lichess.org/standard/sha256sums.txt
- Expected archive SHA256:
  `aa40b3671fa3cf1072eb182892cd90b0e1e003a4a5943492f64b77e7f3fd1635`

To obtain it on another machine:

```sh
mkdir -p data/raw/lichess
curl --fail --location --output data/raw/lichess/lichess_db_standard_rated_2013-01.pgn.zst \
  https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
```

Verify the archive against the published checksum before importing it. Read
compressed data with `zstd -dc` or a streaming Zstandard reader.

## Preparation before training

`tools/train_inezita.py prepare` imports Elite PGNs into version-1 records and
generates separate Stockfish labels (`label_contract: 2`). `tools/dataset.py`
validates and splits existing records. Import retains game history, rejects
invalid/unfinished games, enforces configured rating (`min_rating`), time-control
(`min_time_control_seconds`), and normal-termination (`require_normal_termination`)
filters, records selection/label provenance, and excludes tactical and cross-split
board overlap. See [staged training](../docs/training.md) and the
[0.1.3 guide](../docs/inezita-0.1.3.md) for the Sept–Nov 2025 Elite curation,
teacher supervision, self-play, checkpoint gates and resumption.

Split by whole game, remove duplicate samples, and audit overlapping positions
before training. Keep source URLs, licenses, input checksums, preparation
settings, and split checksums in a reproducible manifest.
