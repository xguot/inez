# SPDX-License-Identifier: GPL-3.0-only
"""Reference neural PUCT for curriculum self-play and paired checkpoint games.

This Python implementation uses the same versioned encodings as the native
engine. Values on edges are always from the parent's side-to-move perspective.
"""

from dataclasses import dataclass, field
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


def terminal_value(board):
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    return 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)


def search(board, evaluate, simulations=64, cpuct=2.5, *, noise=False, rng=None):
    """Return root visit counts; model priors never substitute for visit targets."""
    if simulations < 1 or cpuct <= 0:
        raise ValueError("PUCT requires positive simulations and cpuct")
    if terminal_value(board) is not None:
        return {}
    rng = rng or random.Random(0)
    priors, _ = evaluate(board)
    if not priors or set(priors) != {move.uci() for move in board.legal_moves}:
        raise ValueError("evaluator must return exactly the legal moves")
    if any(not math.isfinite(p) or p < 0 for p in priors.values()) or sum(priors.values()) <= 0:
        raise ValueError("invalid PUCT priors")
    total = sum(priors.values())
    priors = {move: p / total for move, p in priors.items()}
    if noise:
        samples = [rng.gammavariate(0.3, 1.0) for _ in priors]
        denominator = sum(samples)
        priors = {move: 0.75 * p + 0.25 * sample / denominator
                  for (move, p), sample in zip(priors.items(), samples)}
    root = {move: Edge(p) for move, p in priors.items()}
    for _ in range(simulations):
        cursor = board.copy(stack=True)
        children = root
        path = []
        while True:
            visits = sum(edge.visits for edge in children.values())
            move, edge = max(children.items(), key=lambda item: (
                item[1].value_sum / item[1].visits if item[1].visits else 0.0
            ) + cpuct * item[1].prior * math.sqrt(visits + 1) / (1 + item[1].visits))
            cursor.push_uci(move)
            path.append(edge)
            value = terminal_value(cursor)
            if value is not None:
                break
            if not edge.children:
                priors, value = evaluate(cursor)
                if not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError("invalid leaf value")
                edge.children = {move: Edge(p) for move, p in priors.items()}
                break
            children = edge.children
        for edge in reversed(path):
            value = -value
            edge.visits += 1
            edge.value_sum += value
    return {move: edge.visits for move, edge in root.items()}


def choose_move(visits, *, temperature=0.0, rng=None):
    if not visits or sum(visits.values()) <= 0:
        raise ValueError("cannot select from empty visits")
    if temperature == 0:
        return max(visits, key=visits.get)
    rng = rng or random.Random(0)
    weights = [count ** (1 / temperature) for count in visits.values()]
    return rng.choices(list(visits), weights=weights, k=1)[0]
