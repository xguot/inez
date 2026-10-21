# SPDX-License-Identifier: GPL-3.0-only
"""Frozen development diagnostics for neural move choice and value sanity."""

import copy
import json
import math
from pathlib import Path

import chess
import torch

from .encoding import encode_position
from .model import wdl_probabilities_and_value


def mirror_move(notation):
    move = chess.Move.from_uci(notation)
    return chess.Move(move.from_square ^ 56, move.to_square ^ 56, promotion=move.promotion).uci()


def make_board(position):
    if set(position) != {'fen', 'moves'} or not isinstance(position['moves'], list):
        raise ValueError('a position requires fen and a move list')
    board = chess.Board(position['fen'])
    if not board.is_valid():
        raise ValueError('invalid initial FEN')
    for notation in position['moves']:
        move = chess.Move.from_uci(notation)
        if board.is_game_over(claim_draw=False) or move not in board.legal_moves:
            raise ValueError(f'illegal history move: {notation}')
        board.push(move)
    return board


def mirror_position(position):
    return {'fen': chess.Board(position['fen']).mirror().fen(),
            'moves': [mirror_move(move) for move in position['moves']]}


def move_passes(case, board, notation):
    move = chess.Move.from_uci(notation)
    if move not in board.legal_moves:
        return False
    if case.get('root_moves') and notation not in case['root_moves']:
        return False
    if 'best_moves' in case:
        return notation in case['best_moves']
    if 'avoid_moves' in case:
        return notation not in case['avoid_moves']
    child = board.copy(stack=True)
    child.push(move)
    return not child.is_stalemate()


def load_suite(path):
    suite = json.loads(Path(path).read_text())
    if suite['version'] != 1 or type(suite['mirror_colors']) is not bool:
        raise ValueError('unsupported diagnostic suite')
    expanded = {'moves': [], 'values': [], 'terminals': []}
    ids = set()
    for section in expanded:
        for original in suite[section]:
            for mirrored in (False, True) if suite['mirror_colors'] else (False,):
                case = copy.deepcopy(original)
                case['group'] = case['id']
                case['mirrored'] = mirrored
                case['id'] += '_mirrored' if mirrored else ''
                if case['id'] in ids:
                    raise ValueError(f'duplicate diagnostic id: {case["id"]}')
                ids.add(case['id'])
                if mirrored:
                    for key in ('position', 'better', 'worse'):
                        if key in case:
                            case[key] = mirror_position(case[key])
                    for key in ('best_moves', 'avoid_moves', 'root_moves'):
                        if key in case:
                            case[key] = [mirror_move(move) for move in case[key]]
                    if 'perspective' in case:
                        case['perspective'] = 'black' if case['perspective'] == 'white' else 'white'
                if section == 'moves':
                    board = make_board(case['position'])
                    if board.is_game_over(claim_draw=False):
                        raise ValueError(f'move case is terminal: {case["id"]}')
                    predicates = sum(key in case for key in ('best_moves', 'avoid_moves', 'avoid_stalemate'))
                    if predicates != 1 or 'avoid_stalemate' in case and case['avoid_stalemate'] is not True:
                        raise ValueError('a move case must have exactly one explicit criterion')
                    for key in ('best_moves', 'avoid_moves', 'root_moves'):
                        if key in case:
                            if not case[key] or len(case[key]) != len(set(case[key])):
                                raise ValueError(f'empty or duplicate {key}')
                            if any(chess.Move.from_uci(move) not in board.legal_moves for move in case[key]):
                                raise ValueError(f'illegal {key}: {case["id"]}')
                    if 'root_moves' in case and 'best_moves' in case:
                        if not set(case['best_moves']) <= set(case['root_moves']):
                            raise ValueError('expected move is outside the restricted root')
                    if 'avoid_stalemate' in case:
                        outcomes = []
                        for move in board.legal_moves:
                            child = board.copy(stack=True)
                            child.push(move)
                            outcomes.append(child.is_stalemate())
                        if not any(outcomes) or all(outcomes):
                            raise ValueError('stalemate test must contain both safe and stalemating moves')
                elif section == 'values':
                    for key in ('better', 'worse'):
                        if make_board(case[key]).is_game_over(claim_draw=False):
                            raise ValueError('value comparisons must use nonterminal positions')
                    if case['perspective'] not in ('white', 'black'):
                        raise ValueError('value perspective must be white or black')
                    gap = case['minimum_value_gap']
                    if not math.isfinite(gap) or not 0 < gap <= 2:
                        raise ValueError('value gap must be finite and in (0, 2]')
                    if type(case['require_decisive_signs']) is not bool:
                        raise ValueError('require_decisive_signs must be boolean')
                else:
                    board = make_board(case['position'])
                    outcome = board.outcome(claim_draw=False)
                    if outcome is None or outcome.termination.name.lower() != case['termination']:
                        raise ValueError(f'incorrect terminal label: {case["id"]}')
                    expected = 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)
                    if type(case['value']) not in (int, float) or case['value'] != expected:
                        raise ValueError(f'incorrect terminal value: {case["id"]}')
                expanded[section].append(case)
    if any(not cases for cases in expanded.values()):
        raise ValueError('all three diagnostic sections must be nonempty')
    return expanded


def oriented_prediction(probabilities, turn, perspective):
    if len(probabilities) != 3 or any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError('invalid WDL probabilities')
    if not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
        raise ValueError('WDL probabilities must sum to one')
    own = probabilities[0] - probabilities[2]
    value = own if turn == perspective else -own
    return {'wdl_side_to_move': dict(zip(('win', 'draw', 'loss'), probabilities)),
            'side_to_move': 'white' if turn else 'black', 'value_side_to_move': own,
            'perspective': 'white' if perspective else 'black', 'value_for_perspective': value}


@torch.inference_mode()
def predict(model, board, perspective):
    device = next(model.parameters()).device
    features = torch.tensor(encode_position(board), dtype=torch.float32, device=device).reshape(1, 64, 119)
    _, logits = model(features)
    probabilities, _ = wdl_probabilities_and_value(logits)
    return oriented_prediction(probabilities[0].tolist(), board.turn, perspective)


def value_result(case, better, worse):
    good, bad = better['value_for_perspective'], worse['value_for_perspective']
    checks = {'ordered_with_margin': good - bad >= case['minimum_value_gap']}
    if case['require_decisive_signs']:
        checks.update(advantage_positive=good > 0, disadvantage_negative=bad < 0)
    return {'id': case['id'], 'group': case['group'], 'mirrored': case['mirrored'],
            'better': better, 'worse': worse, 'actual_value_gap': good - bad,
            'minimum_value_gap': case['minimum_value_gap'],
            'checks': checks, 'passed': all(checks.values())}


def summarize(report):
    by_budget = {}
    for budget in report['context']['simulations']:
        cases = [result for result in report['moves'] if result['simulations'] == budget]
        categories = {}
        for case in cases:
            counts = categories.setdefault(case['category'], {'passed': 0, 'total': 0})
            counts['total'] += 1
            counts['passed'] += int(case['passed'])
        by_budget[str(budget)] = {'passed': sum(case['passed'] for case in cases),
                                  'total': len(cases), 'categories': categories}
    values = {'passed': sum(case['passed'] for case in report['values']), 'total': len(report['values'])}
    terminals = {'passed': sum(case['passed'] for case in report['terminals']), 'total': len(report['terminals'])}
    return {'moves_by_budget': by_budget, 'value_comparisons': values, 'terminal_controls': terminals,
            'all_checks_passed': all(item['total'] > 0 and item['passed'] == item['total']
                                     for item in [*by_budget.values(), values, terminals]),
            'rating_estimate': None,
            'limitations': 'Mirrored cases are correlated; this development suite includes a known failure. '
                           'WDL checks test direction/order, not calibrated win probabilities. '
                           'Terminal controls test engine rules, not learned playing strength.'}
