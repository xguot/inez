#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Diagnostic labels, perspective handling, and a real UCI report round trip."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

import chess
import torch

from inez.curriculum import sha256
from inez.diagnostics import (load_suite, make_board, mirror_move, mirror_position,
                              move_passes, oriented_prediction, summarize, value_result)
from inez.model import ChessTransformer, ModelConfig, save_checkpoint

FIXTURE = ROOT / 'tests/fixtures/neural_diagnostics.json'


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_suite(FIXTURE)

    def case(self, section, name):
        return next(case for case in self.suite[section] if case['id'] == name)

    def test_mirroring_preserves_history_castling_and_repetition(self):
        self.assertEqual({key: len(value) for key, value in self.suite.items()},
                         {'moves': 16, 'values': 6, 'terminals': 8})
        for section in self.suite.values():
            for case in section:
                if case['mirrored']:
                    continue
                for key in ('position', 'better', 'worse'):
                    if key not in case:
                        continue
                    position = case[key]
                    board = make_board(position)
                    mirror = make_board(mirror_position(position))
                    # Replaying from Black increments the FEN move number on a
                    # different ply. Board/rule state and history still mirror.
                    self.assertEqual(mirror.fen().split()[:5], board.mirror().fen().split()[:5])
                    self.assertEqual(len(mirror.move_stack), len(board.move_stack))
                    self.assertEqual(make_board(mirror_position(mirror_position(position))).fen(), board.fen())
        history = self.case('moves', 'xiyuan_save_bishop')['position']
        self.assertEqual(len(make_board(history).move_stack), 14)
        for name in ('fivefold_repetition', 'fivefold_repetition_mirrored'):
            board = make_board(self.case('terminals', name)['position'])
            self.assertTrue(board.is_fivefold_repetition())
            self.assertFalse(chess.Board(board.fen()).is_fivefold_repetition())

    def test_game_regression_scores_only_the_saved_bishop(self):
        case = self.case('moves', 'xiyuan_save_bishop')
        board = make_board(case['position'])
        self.assertTrue(move_passes(case, board, 'a4b3'))
        self.assertFalse(move_passes(case, board, 'e1g1'))
        self.assertFalse(move_passes(case, board, 'a4a8'))
        self.assertEqual(len(board.move_stack), 14)
        mirrored = self.case('moves', 'xiyuan_save_bishop_mirrored')
        self.assertTrue(move_passes(mirrored, make_board(mirrored['position']), 'a5b6'))

    def test_mate_stalemate_and_promotion_labels_have_rule_evidence(self):
        for suffix in ('', '_mirrored'):
            case = self.case('moves', 'mate_in_one' + suffix)
            board = make_board(case['position'])
            mates = []
            for move in list(board.legal_moves):
                board.push(move)
                if board.is_checkmate():
                    mates.append(move.uci())
                board.pop()
            self.assertEqual(set(mates), set(case['best_moves']))
            case = self.case('moves', 'avoid_stalemate' + suffix)
            board = make_board(case['position'])
            for move in list(board.legal_moves):
                child = board.copy()
                child.push(move)
                self.assertEqual(move_passes(case, board, move.uci()), not child.is_stalemate())
            case = self.case('moves', 'rook_underpromotion' + suffix)
            board = make_board(case['position'])
            for notation in case['root_moves']:
                child = board.copy()
                child.push_uci(notation)
                if notation.endswith('q'):
                    self.assertTrue(child.is_stalemate())
                elif notation.endswith('r'):
                    self.assertIsNone(child.outcome())
                else:
                    self.assertTrue(child.is_insufficient_material())
                self.assertEqual(move_passes(case, board, notation), notation.endswith('r'))
            self.assertFalse(move_passes(case, board, next(move.uci() for move in board.legal_moves
                                                         if not move.promotion)))
        self.assertEqual(mirror_move('c7c8r'), 'c2c1r')

    def test_positions_reject_illegal_null_and_post_terminal_histories(self):
        for position in (
                {'fen': chess.STARTING_FEN, 'moves': ['e2e5']},
                {'fen': chess.STARTING_FEN, 'moves': ['0000']},
                {'fen': '8/8/8/8/8/8/8/8 w - - 0 1', 'moves': []},
                {'fen': chess.STARTING_FEN, 'moves': [], 'ignored': True},
                {**self.case('terminals', 'stalemate')['position'], 'moves': ['h8h7']}):
            with self.assertRaises(ValueError):
                make_board(position)

    def test_suite_rejects_bad_labels_criteria_and_terminal_values(self):
        original = json.loads(FIXTURE.read_text())
        bad_suites = []
        for mutate in (
                lambda suite: suite['moves'][0].update(best_moves=['a4a8']),
                lambda suite: suite['moves'][0].update(avoid_moves=['e1g1']),
                lambda suite: suite['moves'][1].update(id=suite['moves'][0]['id']),
                lambda suite: suite['values'][0].update(minimum_value_gap=float('nan')),
                lambda suite: suite['values'][0].update(perspective='side_to_move'),
                lambda suite: suite['terminals'][0].update(value=1),
                lambda suite: suite['terminals'][0].update(termination='stalemate')):
            suite = copy.deepcopy(original)
            mutate(suite)
            bad_suites.append(suite)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for suite in bad_suites:
                path.write_text(json.dumps(suite))
                with self.assertRaises(ValueError):
                    load_suite(path)


class ScoringTests(unittest.TestCase):
    def test_wdl_perspective_uses_side_to_move_and_keeps_raw_probabilities(self):
        for turn in (chess.WHITE, chess.BLACK):
            for perspective in (chess.WHITE, chess.BLACK):
                result = oriented_prediction([0.6, 0.3, 0.1], turn, perspective)
                self.assertAlmostEqual(result['value_side_to_move'], 0.5)
                self.assertAlmostEqual(result['value_for_perspective'], 0.5 if turn == perspective else -0.5)
                self.assertEqual(result['wdl_side_to_move'], {'win': 0.6, 'draw': 0.3, 'loss': 0.1})
        for probabilities in ([0.6, 0.3], [1, 1, 1], [float('nan'), 0, 0], [-0.1, 0.6, 0.5]):
            with self.assertRaises(ValueError):
                oriented_prediction(probabilities, True, True)

    def test_value_order_and_decisive_sign_are_independent(self):
        case = {'id': 'example', 'group': 'example', 'mirrored': False,
                'minimum_value_gap': 0.05, 'require_decisive_signs': True}
        def score(good, bad):
            return value_result(case, {'value_for_perspective': good}, {'value_for_perspective': bad})
        self.assertTrue(score(0.6, -0.4)['passed'])
        self.assertFalse(score(0.6, 0.1)['passed'])
        self.assertTrue(score(0.6, 0.1)['checks']['ordered_with_margin'])
        self.assertFalse(score(0.01, -0.01)['checks']['ordered_with_margin'])
        self.assertFalse(score(-0.4, 0.6)['passed'])
        case['require_decisive_signs'] = False
        self.assertTrue(score(0.6, 0.1)['passed'])

    def test_summary_separates_rule_controls_from_strength_and_has_no_rating(self):
        report = {'context': {'simulations': [32, 128]}, 'moves': [
            {'simulations': 32, 'category': 'piece_safety', 'passed': False},
            {'simulations': 128, 'category': 'piece_safety', 'passed': True}],
            'values': [{'passed': False}], 'terminals': [{'passed': True}]}
        result = summarize(report)
        self.assertEqual(result['moves_by_budget']['32']['passed'], 0)
        self.assertEqual(result['moves_by_budget']['128']['passed'], 1)
        self.assertEqual(result['terminal_controls'], {'passed': 1, 'total': 1})
        self.assertFalse(result['all_checks_passed'])
        self.assertIsNone(result['rating_estimate'])
        report.update(moves=[], values=[], terminals=[])
        self.assertFalse(summarize(report)['all_checks_passed'])


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(827)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.artifact = cls.root / 'tiny'
        save_checkpoint(ChessTransformer(ModelConfig(width=16, layers=1, heads=2,
                         dynamic_rank=2, policy_width=4)), cls.artifact)
        cls.pointer = cls.root / 'accepted.json'
        cls.pointer.write_text(json.dumps({'artifact': str(cls.artifact), 'history': [str(cls.artifact)]}))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def invoke(self, output, *extra):
        return subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_neural.py'),
                              '--model', str(self.pointer), '--threads', '1', '--simulations', '2',
                              '--output', str(output), *extra], capture_output=True, text=True, timeout=90)

    def test_real_cli_scores_full_suite_without_modifying_artifact_or_pointer(self):
        paths = [FIXTURE, self.pointer, self.artifact / 'manifest.json', self.artifact / 'weights.pt']
        before = [sha256(path) for path in paths]
        output = self.root / 'complete.json'
        result = self.invoke(output)
        self.assertIn(result.returncode, (0, 2), result.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(result.returncode, 0 if report['summary']['all_checks_passed'] else 2)
        self.assertEqual(len(report['moves']), 16)
        self.assertEqual(len(report['values']), 6)
        self.assertEqual(report['summary']['terminal_controls'], {'passed': 8, 'total': 8})
        self.assertEqual(report['context']['suite_sha256'], before[0])
        self.assertEqual(report['context']['weights_sha256'], before[3])
        for case in report['moves']:
            self.assertIn(chess.Move.from_uci(case['move']), make_board(case['position']).legal_moves)
            self.assertEqual(case['simulations'], 2)
        self.assertEqual([sha256(path) for path in paths], before)
        saved = output.read_bytes()
        refused = self.invoke(output)
        self.assertEqual(refused.returncode, 1)
        self.assertIn('report already exists', refused.stderr)
        self.assertEqual(output.read_bytes(), saved)

    def test_reference_failure_is_runner_error_with_no_pass_summary(self):
        output = self.root / 'reference-error.json'
        result = self.invoke(output, '--stockfish', str(self.root / 'missing-stockfish'))
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report['status'], 'error')
        self.assertNotIn('summary', report)
        self.assertIn('error', report)
        self.assertEqual(report['moves'], [])


if __name__ == '__main__':
    unittest.main()
