# Chess rules core

The native core uses C++20 and CMake. It implements standard chess with an
independent rules layer for the planned transformer and PUCT search. The
public API is in `include/inez/chess.hpp`; `inez-rules` exposes it for inspection
and independent testing.

## Positions and moves

Squares use `a1 = 0` through `h8 = 63`. A move contains source, destination,
and an optional promotion piece. Castling uses the king's destination square;
en passant uses the capturing pawn's destination. All four promotion choices
have distinct UCI move strings.

```cpp
#include "inez/chess.hpp"

inez::Position position;
position.play_uci("e2e4");
const auto legal = position.legal_moves();
position.unmake_move();
```

`make_move()` and `play_uci()` reject illegal moves without changing the
position. `unmake_move()` restores board state, clocks, raw en passant state,
castling rights, and known repetition history. Undoing beyond construction or
FEN import is an error. Queries and perft preserve the original position.

The core accepts six-field standard-chess FEN. It checks syntax and basic
position invariants, but does not solve historical reachability. FEN export
retains the raw en passant target after a double pawn move, even if no capture
is available. Chess960 and null moves are outside this milestone.

## History, draws, and cache identity

FEN contains no repetition history. An imported FEN starts with one known
position; earlier occurrences must not be inferred. Replay the preceding
moves when repetition-sensitive results are needed. The network's eventual
history window does not limit the rules core's known history.

`outcome()` reports forced outcomes, including checkmate, stalemate, supported
material draws, fivefold repetition, and the 75-move rule. Checkmate takes
precedence over move-count draws. `outcome(true)` also considers available
50-move and threefold claims. The corresponding `can_claim_*()` methods
consider both the current position and an eligible intended legal move. These
queries do not record a player actually making a claim.

Legal move generation is independent of draw adjudication, so callers must
check `outcome()` when running games. Perft counts legal move paths regardless
of draws. This permits standard reference counts and rule inspection after a
draw condition is reached.

`RepetitionKey` is an exact structural identity: pieces, side to move,
castling rights, and an en passant target only when a legal en passant capture
exists. A pinned pawn that cannot capture en passant does not distinguish two
otherwise identical repetition states.

`SearchKey` additionally includes the halfmove clock and all known repetition
states. Its conservative equality prevents cached draw results from being
reused for different known histories. `repetition_hash()` is a deterministic
64-bit routing/diagnostic hash; exact keys remain authoritative. A hash match
alone never proves equality. The current implementation favors inspectable
state and exact restoration over optimized search storage.

## Dead-position limitation

The material detector covers bare kings, a single bishop or knight against a
bare king, and bishops-only positions where every bishop occupies the same
square color. It deliberately does not declare two knights against a bare
king dead: checkmate is possible with cooperative play.

This is a sound, conservative material classifier, not a general proof that
no legal continuation can ever produce mate. Unusual dead positions involving
blocked pawns or other immobile pieces can remain unrecognized. Extend this
before relying on exhaustive dead-position adjudication in training games.

## Verification

The native test executable checks reference perft trees, castling, checks and
pins, en passant, promotions, FEN validation, mate and draws, claimable versus
automatic endings, history-sensitive cache identity, and make/unmake
restoration. Debug builds can enable address and undefined-behavior sanitizers.

The optional differential suite uses pinned `chess==1.11.2` as an independent
test oracle. It compares legal move sets, raw FEN, check and draw results over
special-rule fixtures and deterministic random games, then undoes every move.
CI runs both suites in release and sanitized debug builds on Linux.

For a longer local differential run:

```sh
.venv/bin/python tests/differential.py build/inezita-rules \
  --seed 42 --games 100 --plies 160
```

Rules reference: [FIDE Laws of Chess](https://handbook.fide.com/chapter/E012023).
Test oracle: [python-chess](https://python-chess.readthedocs.io/en/latest/core.html).
