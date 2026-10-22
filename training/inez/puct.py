# SPDX-License-Identifier: GPL-3.0-only
"""Reference neural PUCT for curriculum self-play and paired checkpoint games.

This Python implementation uses the same versioned encodings as the native
engine. Values on edges are always from the parent's side-to-move perspective.
"""

from dataclasses import dataclass, field
import itertools
import math
from pathlib import Path
import random

import chess
import torch

from .encoding import encode_position, move_to_policy_index
from .model import legal_masked_policy, wdl_probabilities_and_value
from .openings import DEFAULT_OPENINGS_PATH, default_opening_book

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 20.0,
}


@dataclass
class Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)
    parent_value: float = 0.0
    terminal: float | None = None


@dataclass(frozen=True)
class SearchInfo:
    nodes: int
    depth: int
    pv: tuple
    value: float | None


def _edge_order(item):
    _, edge = item
    mean = edge.value_sum / edge.visits if edge.visits else -math.inf
    return (edge.visits, mean, edge.prior)


def search_info(root, nodes, depth):
    """Snapshot the most-visited line; values use the root side's perspective."""
    children, pv, value = root, [], None
    while children:
        move, edge = max(children.items(), key=_edge_order)
        if not edge.visits:
            break
        if not pv:
            value = edge.value_sum / edge.visits
        pv.append(move)
        children = edge.children
    return SearchInfo(nodes, depth, tuple(pv), value)


def _least_valuable_attacker(board: chess.Board, square: int, color: bool) -> tuple[ int | None, float ]:
    """Return (attacker_square, piece_value) for the cheapest legal-ray attacker of square."""
    target_bb = chess.BB_SQUARES[square]
    best_sq = None
    best_val = math.inf
    for sq in board.attackers(color, square):
        if not (board.pin_mask(color, sq) & target_bb):
            continue
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = PIECE_VALUES[piece.piece_type]
        if val < best_val:
            best_val = val
            best_sq = sq
    return best_sq, best_val


def _loss_on_square(board: chess.Board, square: int, defender_color: bool) -> float:
    """Compute net material that (not defender_color) can win by initiating captures on square."""
    target = board.piece_at(square)
    if target is None or target.color != defender_color or target.piece_type == chess.KING:
        return 0.0
    attacker_color = not defender_color
    first_sq, first_val = _least_valuable_attacker(board, square, attacker_color)
    if first_sq is None:
        return 0.0
    sim = board.copy(stack=False)
    gains: list[float] = []
    side = attacker_color
    occupant_val = PIECE_VALUES[target.piece_type]
    next_sq, next_val = first_sq, first_val

    while next_sq is not None:
        if next_val == PIECE_VALUES[chess.KING]:
            opp_sq, _ = _least_valuable_attacker(sim, square, not side)
            if opp_sq is not None:
                break
        promo_gain = 0.0
        if next_val == PIECE_VALUES[chess.PAWN] and chess.square_rank(square) in (0, 7):
            promo_gain = PIECE_VALUES[chess.QUEEN] - PIECE_VALUES[chess.PAWN]
            next_val = PIECE_VALUES[chess.QUEEN]
        gains.append(occupant_val + promo_gain)
        occupant_val = next_val
        moving_piece = sim.remove_piece_at(next_sq)
        if moving_piece is not None:
            if promo_gain > 0.0:
                moving_piece = chess.Piece(chess.QUEEN, side)
            sim.set_piece_at(square, moving_piece)
        side = not side
        next_sq, next_val = _least_valuable_attacker(sim, square, side)

    value = 0.0
    for gain in reversed(gains):
        value = max(0.0, gain - value)
    return value


def _max_hanging_loss(board: chess.Board, color: bool, *, exclude_square: int | None = None) -> float:
    """Return the largest SEE loss among color's minor/major pieces on board."""
    worst = 0.0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            if sq == exclude_square:
                continue
            loss = _loss_on_square(board, sq, color)
            if loss > worst:
                worst = loss
    return worst


def _tactical_priors(board: chess.Board, priors: dict[str, float]) -> dict[str, float]:
    """Adjust priors for 1-ply mates, stalemates, exact SEE, hanging pieces, and endgame conversion."""
    turn = board.turn
    our_mat = sum(len(board.pieces(pt, turn)) * val for pt, val in PIECE_VALUES.items() if pt != chess.KING)
    opp_mat = sum(len(board.pieces(pt, not turn)) * val for pt, val in PIECE_VALUES.items() if pt != chess.KING)
    mat_diff = our_mat - opp_mat
    winning_material = mat_diff >= 1.0
    hanging_before = _max_hanging_loss(board, turn)

    adjusted = {}
    for uci, prior in priors.items():
        move = chess.Move.from_uci(uci)
        mover = board.piece_at(move.from_square)
        if board.is_en_passant(move):
            captured_val = 1.0
        else:
            captured = board.piece_at(move.to_square)
            captured_val = PIECE_VALUES[captured.piece_type] if captured else 0.0
        promo_gain = (PIECE_VALUES[move.promotion] - 1.0) if move.promotion else 0.0

        child = board.copy(stack=8)
        child.push(move)

        if child.is_checkmate():
            adjusted[uci] = max(prior, 0.05) * 200.0
            continue
        if winning_material and (child.is_stalemate() or child.is_insufficient_material()):
            adjusted[uci] = prior * 0.002
            continue
        if mat_diff >= 2.0 and (child.is_repetition(2) or child.halfmove_clock >= 100):
            adjusted[uci] = prior * 0.015
            continue

        mult = 1.0
        loss_at_dest = _loss_on_square(child, move.to_square, turn)
        move_see = captured_val + promo_gain - loss_at_dest

        # Other friendly pieces left hanging after this move (if not giving check).
        other_hanging = 0.0 if child.is_check() else _max_hanging_loss(
            child, turn, exclude_square=move.to_square
        )

        if move_see <= -1.5:
            # Move itself loses material on destination square (bad capture or stepping en prise).
            mult *= 0.015 if move_see <= -2.5 else 0.03
        elif move_see >= 1.5 and (move_see - other_hanging) >= 1.0:
            # Winning capture or promotion that remains profitable even after opponent's best reply.
            net_gain = move_see - other_hanging
            mult *= 5.0 * (1.0 + net_gain / 3.0)

        if hanging_before >= 2.0:
            # A minor/major piece was en prise before the move.
            worst_after = max(max(0.0, -move_see), max(0.0, other_hanging - max(0.0, move_see)))
            if worst_after < hanging_before - 1.0:
                mult *= 5.5
            else:
                mult *= 0.12
        elif other_hanging >= 2.0 and (other_hanging - max(0.0, move_see)) >= 2.0:
            # Move abandoned defense of a friendly piece or walked into a discovered attack.
            mult *= 0.04

        if opp_mat == 0.0 and our_mat >= 4.0:
            opp_moves = child.legal_moves.count()
            our_king_before = board.king(turn)
            our_king_after = child.king(turn)
            opp_king = child.king(not turn)
            if our_king_before is not None and our_king_after is not None and opp_king is not None:
                dist_before = chess.square_distance(our_king_before, opp_king)
                dist_after = chess.square_distance(our_king_after, opp_king)
                prior = 0.5 * prior + 0.5 / max(1, len(priors))
                mult *= (1.0 + 0.45 * (8 - min(8, opp_moves))) * (1.0 + 0.30 * (7 - dist_after))
                if dist_after < dist_before:
                    mult *= 1.8
                elif dist_after > dist_before:
                    mult *= 0.25
            if mover is not None and mover.piece_type == chess.PAWN:
                mult *= 1.6

        adjusted[uci] = prior * mult

    total = sum(adjusted.values())
    return {uci: val / total for uci, val in adjusted.items()} if total > 0 else priors


def _leaf_value_adjustment(board: chess.Board, neural_value: float) -> float:
    """Anchor leaf value with 1-ply SEE material reality and bare-king mating progress."""
    turn = board.turn
    our_mat = sum(len(board.pieces(pt, turn)) * val for pt, val in PIECE_VALUES.items() if pt != chess.KING)
    opp_mat = sum(len(board.pieces(pt, not turn)) * val for pt, val in PIECE_VALUES.items() if pt != chess.KING)
    if (opp_mat == 0.0 and our_mat >= 4.0) or (our_mat == 0.0 and opp_mat >= 4.0):
        winner_turn = our_mat > opp_mat
        bare_king = board.king(not turn if winner_turn else turn)
        win_king = board.king(turn if winner_turn else not turn)
        if bare_king is not None and win_king is not None:
            file_dist = min(chess.square_file(bare_king), 7 - chess.square_file(bare_king))
            rank_dist = min(chess.square_rank(bare_king), 7 - chess.square_rank(bare_king))
            edge_progress = (6 - (file_dist + rank_dist)) / 6.0
            king_closeness = (7 - chess.square_distance(win_king, bare_king)) / 6.0
            if not winner_turn:
                mobility_progress = (8 - min(8, board.legal_moves.count())) / 7.0
                progress = 0.35 * edge_progress + 0.35 * king_closeness + 0.30 * mobility_progress
            else:
                progress = 0.50 * edge_progress + 0.50 * king_closeness
            target = (0.70 + 0.29 * progress) if winner_turn else -(0.70 + 0.29 * progress)
            return max(-1.0, min(1.0, 0.1 * neural_value + 0.9 * target))
    immediate_gain = _max_hanging_loss(board, not turn)
    effective_diff = (our_mat - opp_mat) + 0.85 * immediate_gain
    material_value = math.tanh(effective_diff / 6.0)
    return max(-1.0, min(1.0, 0.75 * neural_value + 0.25 * material_value))


class NeuralEvaluator:
    def __init__(self, model, *, use_book=True, book_path=DEFAULT_OPENINGS_PATH, tactical_guard=True):
        self.model = model
        self.device = next(model.parameters()).device
        self.use_book = use_book and Path(book_path).is_file()
        self.book_path = str(book_path)
        self.tactical_guard = tactical_guard

    @torch.no_grad()
    def __call__(self, board):
        self.model.eval()
        x = torch.tensor(encode_position(board), dtype=torch.float32,
                         device=self.device).reshape(1, 64, 119)
        policy, wdl = self.model(x)
        moves = list(board.legal_moves)
        mask = torch.zeros_like(policy, dtype=torch.bool)
        indices = [move_to_policy_index(board, move) for move in moves]
        mask[0, indices] = True
        probabilities = legal_masked_policy(policy, mask)[0]
        _, value = wdl_probabilities_and_value(wdl)
        priors = {move.uci(): float(probabilities[index].item())
                  for move, index in zip(moves, indices)}
        leaf_value = float(value.item())
        if self.use_book and len(board.move_stack) <= 24:
            priors = default_opening_book(self.book_path).blend_priors(board, priors, weight=0.65)
        if self.tactical_guard and priors:
            priors = _tactical_priors(board, priors)
            leaf_value = _leaf_value_adjustment(board, leaf_value)
        return priors, leaf_value


def terminal_value(board, *, claim_draw=False):
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is None:
        return None
    return 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)


def _normalize_priors(priors, searchmoves=None):
    if searchmoves is not None:
        priors = {move: p for move, p in priors.items() if move in searchmoves}
        if sum(priors.values()) == 0:
            priors = {move: 1.0 for move in priors}
    total = sum(priors.values())
    priors = {move: p / total for move, p in priors.items()}
    floor = 1e-3 / max(1, len(priors))
    if any(p < floor for p in priors.values()):
        priors = {move: max(p, floor) for move, p in priors.items()}
        total = sum(priors.values())
        priors = {move: p / total for move, p in priors.items()}
    return priors


def search(board, evaluate, simulations=64, cpuct=2.5, *, noise=False, rng=None,
           stop=None, on_update=None, update_interval=32, searchmoves=None, depth_limit=None):
    """Return root visit counts; model priors never substitute for visit targets."""
    if (simulations is not None and simulations < 1) or not math.isfinite(cpuct) or cpuct <= 0:
        raise ValueError("PUCT requires positive simulations and cpuct")
    if update_interval < 1 or depth_limit is not None and depth_limit < 1:
        raise ValueError('invalid search reporting/depth limit')
    if simulations is None and stop is None and depth_limit is None:
        raise ValueError('unbounded search requires a stop condition')
    if terminal_value(board) is not None:
        if on_update:
            on_update(SearchInfo(0, 0, (), None))
        return {}
    legal = {move.uci() for move in board.legal_moves}
    if searchmoves is not None and (not searchmoves or not set(searchmoves) <= legal):
        raise ValueError('searchmoves must contain legal root moves')
    if stop is not None and stop():
        if on_update:
            on_update(SearchInfo(0, 0, (), None))
        return {move.uci(): 0 for move in board.legal_moves
                if searchmoves is None or move.uci() in searchmoves}
    rng = rng or random.Random(0)
    priors, root_value = evaluate(board)
    if not priors or set(priors) != legal:
        raise ValueError("evaluator must return exactly the legal moves")
    if any(not math.isfinite(p) or p < 0 for p in priors.values()) or sum(priors.values()) <= 0:
        raise ValueError("invalid PUCT priors")
    if not math.isfinite(root_value) or not -1 <= root_value <= 1:
        raise ValueError("invalid leaf value")
    priors = _normalize_priors(priors, searchmoves)
    if noise:
        samples = [rng.gammavariate(0.3, 1.0) for _ in priors]
        denominator = sum(samples)
        priors = {move: 0.75 * p + 0.25 * sample / denominator
                  for (move, p), sample in zip(priors.items(), samples)}
    root = {move: Edge(p, parent_value=root_value) for move, p in priors.items()}
    completed, deepest, reported = 0, 0, -1

    def finish():
        if on_update and completed != reported:
            on_update(search_info(root, completed, deepest))
        return {move: edge.visits for move, edge in root.items()}

    def puct_score(item, visits, node_value, fpu_value):
        _, edge = item
        if edge.visits:
            q = edge.value_sum / edge.visits
            if edge.terminal == 0.0:
                return q
            if edge.terminal == -1.0:
                return 2.0 + edge.prior
        else:
            q = fpu_value if visits > 0 else node_value
        return q + cpuct * edge.prior * math.sqrt(visits + 1) / (1 + edge.visits)

    iterations = range(simulations) if simulations is not None else itertools.count()
    for _ in iterations:
        if stop is not None and stop():
            break
        cursor = board.copy(stack=True)
        children = root
        path = []
        while True:
            if stop is not None and stop():
                return finish()
            visits = sum(edge.visits for edge in children.values())
            first_edge = next(iter(children.values()))
            if visits > 0:
                total_value = sum(edge.value_sum for edge in children.values())
                visited_prior = sum(edge.prior for edge in children.values() if edge.visits > 0)
                node_value = (first_edge.parent_value + total_value) / (1 + visits)
                fpu_value = max(-1.0, node_value - 0.22 * math.sqrt(min(1.0, visited_prior)))
            else:
                node_value = first_edge.parent_value
                fpu_value = node_value
            move, edge = max(children.items(),
                             key=lambda item: puct_score(item, visits, node_value, fpu_value))
            cursor.push_uci(move)
            path.append(edge)
            value = terminal_value(cursor, claim_draw=True)
            if value is not None:
                edge.terminal = value
                break
            if not edge.children:
                priors, value = evaluate(cursor)
                if not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError("invalid leaf value")
                edge.children = {move: Edge(p, parent_value=value)
                                 for move, p in _normalize_priors(priors).items()}
                break
            children = edge.children
        for depth_from_leaf, edge in enumerate(reversed(path)):
            value = -value * (0.99 if depth_from_leaf > 0 else 1.0)
            edge.visits += 1
            edge.value_sum += value
        completed += 1
        deepest = max(deepest, len(path))
        if on_update and completed % update_interval == 0:
            on_update(search_info(root, completed, deepest))
            reported = completed
        if depth_limit is not None and deepest >= depth_limit:
            break
    return finish()


def choose_move(visits, *, temperature=0.0, rng=None):
    if not visits or sum(visits.values()) <= 0:
        raise ValueError("cannot select from empty visits")
    if temperature == 0:
        return max(visits, key=visits.get)
    rng = rng or random.Random(0)
    weights = [count ** (1 / temperature) for count in visits.values()]
    return rng.choices(list(visits), weights=weights, k=1)[0]
