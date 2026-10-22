"""Pure Python implementation of Inez's version 1 neural encoding contract.

No array or machine-learning runtime is needed. Features are a flat, row-major
64 by 119 float list. All eight history frames use the current player's board
orientation and ownership, rather than the player to move in each past frame.
"""

from collections import Counter
import math
import struct
from typing import Sequence

import chess

CONTRACT_VERSION = 1
HISTORY_FRAMES = 8
FRAME_FEATURES = 14
FEATURE_COUNT = HISTORY_FRAMES * FRAME_FEATURES + 7
POLICY_SIZE = 4288
SQUARE_COUNT = 64
UNDERPROMOTION_OFFSET = 4096
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def encoding_metadata():
    """Return JSON-compatible identifiers that a consumer must agree on."""
    return {
        "contract_version": CONTRACT_VERSION,
        "history_frames": HISTORY_FRAMES,
        "feature_count": FEATURE_COUNT,
        "policy_size": POLICY_SIZE,
        "orientation": "current_player_rank_flip",
        "layout": "square_major",
        "value_order": ["win", "draw", "loss"],
    }


def oriented_square(square: int, perspective: chess.Color) -> int:
    if not isinstance(square, int) or isinstance(square, bool) or not 0 <= square < 64:
        raise ValueError("square must be an integer from 0 to 63")
    if perspective not in (chess.WHITE, chess.BLACK):
        raise ValueError("perspective must be chess.WHITE or chess.BLACK")
    return square if perspective == chess.WHITE else square ^ 56


def _repetition_identity(board: chess.Board):
    # Piece origin (promoted or original) is irrelevant to FIDE repetition.
    # Raw FEN EP squares count only if an EP capture is actually legal.
    return (
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
        board.turn, board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def _require_standard_board(board: chess.Board):
    if not isinstance(board, chess.Board):
        raise TypeError("expected a chess.Board")
    if board.chess960 or type(board) is not chess.Board:
        raise ValueError("contract version 1 supports standard chess only")
    if not board.is_valid():
        raise ValueError("cannot encode an invalid chess position")


def encode_position(board: chess.Board) -> list[float]:
    """Encode the current position and up to seven known preceding positions.

    Missing frames are entirely zero, including their availability flag. A FEN
    import has one known frame; history must be retained to represent repetition
    faithfully. Neither the board nor its move stack is modified.
    """
    _require_standard_board(board)
    cursor = board.copy(stack=True)
    keys = []
    frames = []
    while True:
        keys.append(_repetition_identity(cursor))
        if len(frames) < HISTORY_FRAMES:
            frames.append(cursor.copy(stack=False))
        if not cursor.move_stack:
            break
        cursor.pop()
    counts = Counter(keys)
    features = [0.0] * (SQUARE_COUNT * FEATURE_COUNT)
    perspective = board.turn
    for frame_index, frame in enumerate(frames):
        base = frame_index * FRAME_FEATURES
        repeated = float(counts[keys[frame_index]] >= 2)
        for square in chess.SQUARES:
            output_square = oriented_square(square, perspective)
            offset = output_square * FEATURE_COUNT + base
            piece = frame.piece_at(square)
            if piece is not None:
                channel = piece.piece_type - 1 + (0 if piece.color == perspective else 6)
                features[offset + channel] = 1.0
            features[offset + 12] = repeated
            features[offset + 13] = 1.0
        # The next older frame must not see occurrences from its future.
        counts[keys[frame_index]] -= 1

    extras = (
        float(board.has_kingside_castling_rights(perspective)),
        float(board.has_queenside_castling_rights(perspective)),
        float(board.has_kingside_castling_rights(not perspective)),
        float(board.has_queenside_castling_rights(not perspective)),
    )
    halfmove = struct.unpack("f", struct.pack("f", min(board.halfmove_clock, 150) / 150.0))[0]
    absolute_black = float(perspective == chess.BLACK)
    for square in chess.SQUARES:
        offset = square * FEATURE_COUNT
        features[offset + 112:offset + 116] = extras
        features[offset + 117] = halfmove
        features[offset + 118] = absolute_black
    if board.has_legal_en_passant():
        ep = oriented_square(board.ep_square, perspective)
        features[ep * FEATURE_COUNT + 116] = 1.0
    return features


def move_to_policy_index(board: chess.Board, move: chess.Move) -> int:
    """Map a legal move to the v1 policy vocabulary; queen promotions share
    the ordinary from-to entry, while N/B/R promotions have dedicated entries.
    """
    _require_standard_board(board)
    if not isinstance(move, chess.Move) or not board.is_legal(move):
        raise ValueError("policy target must be a legal chess move")
    return _move_to_index(board, move)


def _move_to_index(board: chess.Board, move: chess.Move) -> int:
    source = oriented_square(move.from_square, board.turn)
    target = oriented_square(move.to_square, board.turn)
    if move.promotion in UNDERPROMOTIONS:
        if chess.square_rank(source) != 6 or chess.square_rank(target) != 7:
            raise ValueError("underpromotion must run from oriented rank 7 to rank 8")
        piece_index = UNDERPROMOTIONS.index(move.promotion)
        return (UNDERPROMOTION_OFFSET + piece_index * 64 +
                chess.square_file(source) * 8 + chess.square_file(target))
    return source * 64 + target


def legal_policy_indices(board: chess.Board) -> list[int]:
    _require_standard_board(board)
    indices = [_move_to_index(board, move) for move in board.legal_moves]
    if len(indices) != len(set(indices)):
        raise ValueError("legal move policy mapping is not one-to-one")
    return indices


def policy_index_to_move(board: chess.Board, index: int) -> chess.Move:
    """Resolve an index through the current legal moves, rejecting invalid or
    masked entries instead of inventing an illegal move from its coordinates.
    """
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < POLICY_SIZE:
        raise ValueError("policy index is outside the v1 vocabulary")
    _require_standard_board(board)
    for move in board.legal_moves:
        if _move_to_index(board, move) == index:
            return move
    raise ValueError("policy index is not legal in this position")


def _validated_wdl(wdl: Sequence[float]) -> tuple[float, float, float]:
    if len(wdl) != 3:
        raise ValueError("WDL must contain win, draw, loss probabilities")
    values = tuple(float(value) for value in wdl)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("WDL probabilities must be finite and between zero and one")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("WDL probabilities must sum to one")
    return values


def expected_value(wdl: Sequence[float]) -> float:
    win, _, loss = _validated_wdl(wdl)
    return win - loss


def convert_wdl_perspective(wdl: Sequence[float], source: chess.Color,
                            target: chess.Color) -> tuple[float, float, float]:
    values = _validated_wdl(wdl)
    if source not in (chess.WHITE, chess.BLACK) or target not in (chess.WHITE, chess.BLACK):
        raise ValueError("source and target must be chess colors")
    return values if source == target else (values[2], values[1], values[0])


def wdl_from_result(result: str, perspective: chess.Color) -> tuple[float, float, float]:
    if result == "1-0":
        white_wdl = (1.0, 0.0, 0.0)
    elif result == "0-1":
        white_wdl = (0.0, 0.0, 1.0)
    elif result == "1/2-1/2":
        white_wdl = (0.0, 1.0, 0.0)
    else:
        raise ValueError("result must be a completed standard chess result")
    return convert_wdl_perspective(white_wdl, chess.WHITE, perspective)


def legal_masked_softmax(logits: Sequence[float], legal_indices: Sequence[int]) -> list[float]:
    """Numerically stable policy probabilities; no legal moves yields zeros.

    These are geometric legal moves, including in automatic draw positions.
    Game adjudication remains the caller's responsibility.
    """
    if len(logits) != POLICY_SIZE:
        raise ValueError("policy logits must have 4288 entries")
    indices = list(legal_indices)
    if any(not isinstance(index, int) or isinstance(index, bool) or
           not 0 <= index < POLICY_SIZE for index in indices):
        raise ValueError("legal policy index is outside the v1 vocabulary")
    if len(set(indices)) != len(indices):
        raise ValueError("legal policy indices must be unique")
    if any(not math.isfinite(logits[index]) for index in indices):
        raise ValueError("legal policy logits must be finite")
    result = [0.0] * POLICY_SIZE
    if not indices:
        return result
    maximum = max(logits[index] for index in indices)
    exponentials = [math.exp(logits[index] - maximum) for index in indices]
    denominator = sum(exponentials)
    for index, numerator in zip(indices, exponentials):
        result[index] = numerator / denominator
    return result
