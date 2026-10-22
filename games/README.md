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

[Games against Xiyuan](recorded/xiyuan.pgn) includes the October 20 game with Inez winning as
White, 1-0, after `13.Nxa8`. The supplied En Croissant headers retain Black's
display name `Player`, the time control `600+0.001`, and all move clocks. The
final board is not checkmate; the record does not specify the termination reason.

Xiyuan's corrected identification confirms the October 20 opponent was the old
handcrafted Inez, matching its `Inez 0.1.0` player tag. It is a baseline-engine
game, not a neural result.

The second game, dated October 21, explicitly identifies White as
`Inez Neural 0.1.0`. Its supplied result is `*`, with moves through `15.Qh5`;
preserve it as unfinished even though Xiyuan reports being ahead. The record
includes a lost bishop after `7...b5 8.O-O? bxa4 9.Nxa4` and retains the move
clocks and arrow annotations. It does not identify the exact checkpoint or
search limits. Future neural games should record the selected artifact/checksum
and search settings alongside the engine name.

A local replay of the bishop-loss position with the currently accepted pilot
checkpoint (`teacher/candidate-0001000`) reproduced `8.O-O` at 32, 128 and 512
PUCT simulations. A short Stockfish analysis preferred `8.Bb3`. This is a
diagnostic of the current checkpoint, not proof that the unpinned PGN used
identical weights or a calibrated rating measurement.

The third game in `xiyuan.pgn`, also dated October 21, identifies White as
`Inez 0.1.1` against `Player` (`600+0.001`, result `*` after `6.d3`). After
`1.Nc3 d5 2.Nf3 d4 3.Nb5 Nc6`, White played `4.Nbxd4??`, losing a knight for a
pawn after `4...Nxd4 5.Nxd4 Qxd4`. Replaying the position with the accepted
pilot checkpoint (`teacher/candidate-0002000`) reproduced `4.Nbxd4` (`b5d4` had
50.0% policy prior vs. 1.89% for Black's `4...Nxd4` recapture, and the raw
value head evaluated the post-exchange knight-down board as `+0.2375` for
White), motivating the opening-ply (`min_ply: 0`), off-policy blunder
relabeling, MCO opening book, decoupled material value pooling, and PUCT
first-play urgency fixes.

`xiyuan.pgn` uses the player-specific filename requested for this record. New
dated records should continue to follow the naming convention above. Recorded
games are not automatically added to the training corpus or acceptance gates.

