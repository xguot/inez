# Recorded games

Keep selected original games in `recorded/`, preserving their PGN headers,
moves, and any supplied annotations. These are source records; future training
imports can extract positions and labels separately.

Use `YYYY-MM-DD-white-vs-black-NNN.pgn` for filenames. Keep player names in the
`White` and `Black` tags and engine versions in the engine's player tag. The
sequence number distinguishes games between the same players on the same day.

The first recorded game is
[Inez 0.1.0 vs Xiyuan](recorded/2026-10-19-inez-vs-xiyuan-001.pgn), played on
October 19, 2026. Xiyuan played Black and won by checkmate, 0-1.

This PGN does not include time controls or engine search settings. Preserve
those settings with future games when they are available.
