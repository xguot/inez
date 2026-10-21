#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run frozen neural diagnostics through UCI; exit 2 for model failures, 1 for runner errors."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

import chess
import chess.engine
import torch

from inez.curriculum import atomic_json, sha256
from inez.diagnostics import load_suite, make_board, move_passes, predict, summarize, value_result
from inez.model import load_checkpoint
from inez.puct import NeuralEvaluator, terminal_value
from inez.uci import DEFAULT_MODEL, resolve_artifact


def reference_checks(suite, executable, nodes):
    """Validate case labels independently before looking at the tested network."""
    path = shutil.which(executable) or str(Path(executable).expanduser().resolve())
    report = {'executable': path, 'binary_sha256': sha256(path), 'nodes': nodes,
              'move_labels': [], 'value_ordering': []}
    with chess.engine.SimpleEngine.popen_uci(path, timeout=30) as engine:
        options = {'Threads': 1, 'Hash': 64}
        if 'UCI_LimitStrength' in engine.options:
            options['UCI_LimitStrength'] = False
        engine.configure(options)
        report.update(id=engine.id, options=options)
        def analyze(position, roots=None):
            board = make_board(position)
            result = engine.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=roots)
            return board, result
        for case in suite['moves']:
            roots = [chess.Move.from_uci(move) for move in case['root_moves']] if 'root_moves' in case else None
            board, info = analyze(case['position'], roots)
            move = info['pv'][0].uci()
            passed = move_passes(case, board, move)
            report['move_labels'].append({'id': case['id'], 'move': move,
                'score_side_to_move': str(info['score'].pov(board.turn)), 'label_confirmed': passed})
            if not passed:
                raise ValueError(f'reference does not confirm move labels: {case["id"]} ({move})')
        for case in suite['values']:
            perspective = case['perspective'] == 'white'
            _, good = analyze(case['better'])
            _, bad = analyze(case['worse'])
            good_score, bad_score = good['score'].pov(perspective), bad['score'].pov(perspective)
            passed = good_score > bad_score
            if case['require_decisive_signs']:
                passed = passed and good_score > chess.engine.Cp(0) and bad_score < chess.engine.Cp(0)
            report['value_ordering'].append({'id': case['id'], 'better_score': str(good_score),
                'worse_score': str(bad_score), 'ordering_confirmed': passed})
            if not passed:
                raise ValueError(f'reference does not confirm value ordering: {case["id"]}')
    return report


def run(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError(f'report already exists: {output}; choose a new output path')
    suite = load_suite(args.suite)
    artifact = resolve_artifact(args.model)
    manifest = json.loads((artifact / 'manifest.json').read_text())
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    # The strict loader checks manifest, encoding, tensor shapes and the actual weight hash.
    model = load_checkpoint(artifact, device=args.device)
    evaluator = NeuralEvaluator(model)
    try:
        revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {'format_version': 1, 'status': 'running', 'context': {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'model': str(artifact),
        'weights_sha256': manifest['weights_sha256'], 'parameters': sum(p.numel() for p in model.parameters()),
        'suite': str(Path(args.suite).resolve()), 'suite_sha256': sha256(args.suite),
        'runner_sha256': sha256(__file__),
        'diagnostics_sha256': sha256(ROOT / 'training/inez/diagnostics.py'),
        'uci_sha256': sha256(ROOT / 'training/inez/uci.py'), 'puct_sha256': sha256(ROOT / 'training/inez/puct.py'),
        'git_revision': revision, 'python': platform.python_version(), 'torch': str(torch.__version__),
        'chess': chess.__version__, 'platform': platform.platform(), 'device': args.device,
        'threads': args.threads, 'cpuct': 2.5, 'simulations': args.simulations,
        'uci_launcher': str(ROOT / 'inez-neural')},
        'reference': None, 'moves': [], 'values': [], 'terminals': []}
    atomic_json(output, report)
    try:
        if args.stockfish:
            report['reference'] = reference_checks(suite, args.stockfish, args.reference_nodes)
            atomic_json(output, report)
        command = [sys.executable, str(ROOT / 'tools/neural_uci.py'), '--model', str(artifact),
                   '--device', args.device, '--threads', str(args.threads)]
        report['context']['uci_command'] = command
        with chess.engine.SimpleEngine.popen_uci(command, timeout=60) as engine:
            engine.configure({'Cpuct': '2.5'})
            engine.ping()
            report['context']['uci_id'] = engine.id
            for case in suite['moves']:
                board = make_board(case['position'])
                priors, _ = evaluator(board)
                raw = predict(model, board, board.turn)
                for budget in args.simulations:
                    start = time.monotonic()
                    roots = [chess.Move.from_uci(move) for move in case['root_moves']] if 'root_moves' in case else None
                    result = engine.play(board, chess.engine.Limit(nodes=budget), root_moves=roots,
                                         game=object(), info=chess.engine.INFO_ALL)
                    if result.move not in board.legal_moves:
                        raise RuntimeError(f'illegal UCI move in {case["id"]}')
                    if result.info.get('nodes') != budget:
                        raise RuntimeError(f'incomplete node budget in {case["id"]}')
                    cursor = board.copy(stack=True)
                    for move in result.info.get('pv', []):
                        if move not in cursor.legal_moves:
                            raise RuntimeError('illegal UCI principal variation')
                        cursor.push(move)
                    notation = result.move.uci()
                    item = {'id': case['id'], 'group': case['group'], 'mirrored': case['mirrored'],
                        'category': case['category'], 'description': case['description'],
                        'position': case['position'], 'simulations': budget, 'move': notation,
                        'san': board.san(result.move), 'passed': move_passes(case, board, notation),
                        'criterion': {key: case[key] for key in ('best_moves', 'avoid_moves', 'avoid_stalemate', 'root_moves') if key in case},
                        'policy_top': sorted(priors.items(), key=lambda pair: pair[1], reverse=True)[:5],
                        'raw_value': raw, 'depth': result.info.get('depth'),
                        'pv': [move.uci() for move in result.info.get('pv', [])],
                        'elapsed_seconds': time.monotonic() - start}
                    report['moves'].append(item)
                    atomic_json(output, report)
                    print(json.dumps({'case': case['id'], 'simulations': budget,
                                      'move': item['san'], 'passed': item['passed']}), flush=True)
            for case in suite['values']:
                perspective = case['perspective'] == 'white'
                good = predict(model, make_board(case['better']), perspective)
                bad = predict(model, make_board(case['worse']), perspective)
                item = value_result(case, good, bad)
                report['values'].append(item)
                atomic_json(output, report)
                print(json.dumps({'value_case': case['id'], 'passed': item['passed'],
                                  'checks': item['checks']}), flush=True)
            for case in suite['terminals']:
                board = make_board(case['position'])
                result = engine.play(board, chess.engine.Limit(nodes=1), game=object(), info=chess.engine.INFO_ALL)
                move = result.move.uci() if result.move is not None else '0000'
                actual = terminal_value(board)
                report['terminals'].append({'id': case['id'], 'termination': case['termination'],
                    'bestmove': move, 'value': actual, 'used_network': False,
                    'passed': move == '0000' and actual == case['value'] and result.info.get('nodes') == 0})
                atomic_json(output, report)
        report['status'] = 'complete'
        report['summary'] = summarize(report)
        atomic_json(output, report)
        print(json.dumps(report['summary'], indent=2), flush=True)
        return 0 if report['summary']['all_checks_passed'] else 2
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error'
        report['error'] = str(error)
        atomic_json(output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--suite', default=str(ROOT / 'tests/fixtures/neural_diagnostics.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--simulations', type=int, nargs='+', default=[32, 128])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--stockfish', help='optional independent label verification using this executable')
    parser.add_argument('--reference-nodes', type=int, default=100_000)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64 or any(value < 1 for value in args.simulations) or args.reference_nodes < 1:
        parser.error('invalid thread count, simulation budget or reference nodes')
    args.simulations = list(dict.fromkeys(args.simulations))
    try:
        return run(args)
    except Exception as error:
        print(f'Neural diagnostics failed: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
