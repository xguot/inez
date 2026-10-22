# SPDX-License-Identifier: GPL-3.0-only
"""Modern Chess Openings (MCO / ECO A-E) book lookup and training record generation."""

from collections import defaultdict
import csv
from dataclasses import dataclass
import functools
import hashlib
import json
from pathlib import Path

import chess

from .encoding import wdl_from_result
from .records import validate_record

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENINGS_PATH = ROOT / 'data/openings/mco_openings.tsv'


def position_key(board: chess.Board) -> str:
    return ' '.join(board.fen(en_passant='legal').split()[:4])


@dataclass(frozen=True)
class OpeningEntry:
    eco: str
    name: str
    pgn: str
    moves: tuple[str, ...]


def load_mco_entries(path=DEFAULT_OPENINGS_PATH) -> tuple[OpeningEntry, ...]:
    """Load and validate ECO/MCO opening lines from a tab-separated corpus."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'MCO openings file not found: {path}')
    entries = []
    with path.open(encoding='utf-8') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['eco', 'name', 'pgn', 'uci']:
            raise ValueError('MCO openings TSV must have columns: eco, name, pgn, uci')
        for row in reader:
            moves = tuple(row['uci'].split())
            if not moves:
                raise ValueError(f'empty opening line: {row}')
            entries.append(OpeningEntry(
                eco=row['eco'].strip(),
                name=row['name'].strip(),
                pgn=row['pgn'].strip(),
                moves=moves,
            ))
    if not entries:
        raise ValueError('MCO openings corpus is empty')
    return tuple(entries)


DUBIOUS_ROOT_MOVES = frozenset({
    'f2f3', 'g2g4', 'h2h4', 'a2a4', 'g1h3', 'b1a3', 'a2a3', 'h2h3', 'c2c3', 'd2d3', 'e2e3', 'f2f4',
})

PRINCIPLED_ROOT_MOVES = {
    'e2e4': 1.35,
    'd2d4': 1.35,
    'c2c4': 1.20,
    'g1f3': 1.20,
}


def _is_train_bucket(initial_fen: str, moves: list[str], seed: str, train_fraction: float = 0.8) -> bool:
    canonical = json.dumps(
        {'initial_fen': initial_fen, 'moves': moves},
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )
    fingerprint = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    digest = hashlib.sha256((str(seed) + '\0' + fingerprint).encode('utf-8')).digest()
    bucket = int.from_bytes(digest[:8], 'big') / 2**64
    return bucket < train_fraction


def _ensure_train_trajectory(board: chess.Board, prefix: list[str], seed: str,
                             initial_fen: str = chess.STARTING_FEN) -> tuple[list[str], str]:
    """Return a move list starting with prefix that hashes into the train split."""
    if _is_train_bucket(initial_fen, prefix, seed):
        return prefix, '1/2-1/2'
    queue: list[tuple[chess.Board, list[str]]] = [(board, prefix)]
    for _ in range(4):
        next_queue: list[tuple[chess.Board, list[str]]] = []
        for cursor, moves in queue:
            for move in cursor.legal_moves:
                child = cursor.copy(stack=False)
                child.push(move)
                candidate = moves + [move.uci()]
                outcome = child.outcome(claim_draw=False)
                result = outcome.result() if outcome is not None else '1/2-1/2'
                if _is_train_bucket(initial_fen, candidate, seed):
                    return candidate, result
                if outcome is None and len(next_queue) < 64:
                    next_queue.append((child, candidate))
        queue = next_queue
        if not queue:
            break
    return prefix, '1/2-1/2'


class OpeningBook:
    """Transposition-aware Modern Chess Openings lookup table."""

    def __init__(self, entries: tuple[OpeningEntry, ...]):
        self.entries = entries
        counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        names: dict[str, tuple[str, str]] = {}
        for entry in entries:
            end_board = chess.Board()
            for move_uci in entry.moves:
                end_board.push_uci(move_uci)
            outcome = end_board.outcome(claim_draw=False)
            loser = (not outcome.winner) if (outcome is not None and outcome.winner is not None) else None

            board = chess.Board()
            # Mainlines and shorter canonical MCO lines receive higher base weight.
            line_weight = 1.0 + 2.0 / max(1, len(entry.moves))
            if entry.eco == 'A00':
                line_weight *= 0.15
            first_move = entry.moves[0]
            if first_move in PRINCIPLED_ROOT_MOVES:
                line_weight *= PRINCIPLED_ROOT_MOVES[first_move]

            for ply, move_uci in enumerate(entry.moves):
                # Never teach the losing side's blunders in checkmate trap lines (e.g. Fool's Mate).
                if loser is None or board.turn != loser:
                    weight = line_weight
                    if ply == 0 and move_uci in DUBIOUS_ROOT_MOVES:
                        weight *= 0.05
                    counts[position_key(board)][move_uci] += weight
                board.push_uci(move_uci)
            names.setdefault(position_key(board), (entry.eco, entry.name))
        self._continuations = {
            key: dict(move_weights) for key, move_weights in counts.items()
        }
        self._names = names

    def priors(self, board: chess.Board) -> dict[str, float] | None:
        raw = self._continuations.get(position_key(board))
        if not raw:
            return None
        legal = {move.uci() for move in board.legal_moves}
        filtered = {move: weight for move, weight in raw.items() if move in legal and weight > 0}
        if not filtered:
            return None
        cutoff = 0.02 * max(filtered.values())
        pruned = {move: weight for move, weight in filtered.items() if weight >= cutoff}
        total = sum(pruned.values())
        return {move: weight / total for move, weight in pruned.items()}

    def blend_priors(self, board: chess.Board, neural_priors: dict[str, float],
                     *, weight: float = 0.65) -> dict[str, float]:
        if not 0.0 <= weight <= 1.0:
            raise ValueError('book blend weight must be in [0, 1]')
        book = self.priors(board)
        if not book or weight == 0.0:
            return neural_priors
        blended = {
            move: (1.0 - weight) * prior + weight * book.get(move, 0.0)
            for move, prior in neural_priors.items()
        }
        total = sum(blended.values())
        return {move: value / total for move, value in blended.items()} if total > 0 else neural_priors

    def identify(self, board: chess.Board) -> tuple[str, str] | None:
        return self._names.get(position_key(board))


@functools.lru_cache(maxsize=4)
def default_opening_book(path: str = str(DEFAULT_OPENINGS_PATH)) -> OpeningBook:
    return OpeningBook(load_mco_entries(Path(path)))


def opening_records(path=DEFAULT_OPENINGS_PATH, *, excluded=frozenset(), seed='20261019'):
    """Build deduplicated v1 training records covering all MCO opening positions."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source_id = 'mco-' + digest[:16]
    entries = load_mco_entries(path)
    book = OpeningBook(entries)
    shortest_prefix: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        board = chess.Board()
        for ply in range(len(entry.moves)):
            if board.is_game_over(claim_draw=False):
                break
            key = position_key(board)
            if key not in excluded:
                prefix = entry.moves[:ply]
                if key not in shortest_prefix or len(prefix) < len(shortest_prefix[key]):
                    shortest_prefix[key] = prefix
            board.push_uci(entry.moves[ply])
    records = []
    initial_fen = chess.Board(chess.STARTING_FEN).fen(en_passant='fen')
    for key, prefix in sorted(shortest_prefix.items(), key=lambda item: (len(item[1]), item[1])):
        ply = len(prefix)
        board = chess.Board(initial_fen)
        for move_uci in prefix:
            board.push_uci(move_uci)
        if board.is_game_over(claim_draw=False):
            continue
        policy = book.priors(board)
        if not policy:
            best_move = next(iter(board.legal_moves)).uci()
            policy = {best_move: 1.0}
        moves, result = _ensure_train_trajectory(board, list(prefix), str(seed), initial_fen=initial_fen)
        game_id = hashlib.sha256((' '.join(moves) + f':{ply}').encode('utf-8')).hexdigest()
        record = {
            'record_version': 1,
            'encoding_version': 1,
            'source_id': source_id,
            'game_id': game_id,
            'initial_fen': initial_fen,
            'moves': moves,
            'ply': ply,
            'result': result,
            'policy': policy,
            'wdl': list(wdl_from_result(result, board.turn)),
            'label_kind': 'game_result',
            'teacher': None,
        }
        validate_record(record)
        records.append(record)
    source_meta = {
        source_id: {
            'uri': 'https://github.com/lichess-org/chess-openings',
            'license': 'CC0-1.0',
            'sha256': digest,
        }
    }
    return records, source_meta
