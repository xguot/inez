# SPDX-License-Identifier: GPL-3.0-only
"""Reference neural PUCT for curriculum self-play and paired checkpoint games.

This Python implementation uses the same versioned encodings as the native
engine. Values on edges are always from the parent's side-to-move perspective.
"""

from dataclasses import dataclass, field
import itertools
import math
import random

import chess
import torch

from .encoding import encode_position, move_to_policy_index
from .model import legal_masked_policy, wdl_probabilities_and_value


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


class NeuralEvaluator:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

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
        return {move.uci(): float(probabilities[index].item())
                for move, index in zip(moves, indices)}, float(value.item())


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

    def puct_score(item, visits):
        _, edge = item
        if edge.visits:
            q = edge.value_sum / edge.visits
            if edge.terminal == 0.0:
                return q
            if edge.terminal == -1.0:
                return 2.0 + edge.prior
        else:
            q = max(0.0, edge.parent_value - 0.25) if visits > 0 else 0.0
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
            move, edge = max(children.items(), key=lambda item: puct_score(item, visits))
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
        for edge in reversed(path):
            value = -value
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
