#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Curriculum labels, search signs, independent gates, and exact resume."""

import copy
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import chess
import chess.engine
import torch

from inez.curriculum import (atomic_json, import_games, position_key, promotion_decision,
                            selfplay_records, stockfish_labels)
from inez.model import ChessTransformer, ModelConfig, save_checkpoint
from inez.puct import choose_move, search
from inez.records import load_records, validate_record
from inez.trainer import train_stage

ROOT = Path(__file__).resolve().parents[1]
MATE = '6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1'


def example(ply=0):
    return {'record_version': 1, 'encoding_version': 1, 'source_id': 'fixture', 'game_id': 'fools',
            'initial_fen': chess.STARTING_FEN, 'moves': ['f2f3', 'e7e5', 'g2g4', 'd8h4'],
            'ply': ply, 'result': '0-1', 'policy': {'h2h3': 1.0} if ply == 0 else {'e7e5': 1.0},
            'wdl': [0, 0, 1] if ply % 2 == 0 else [1, 0, 0],
            'label_kind': 'game_result', 'teacher': None}


def uniform(board):
    moves = list(board.legal_moves)
    return {move.uci(): 1 / len(moves) for move in moves}, 0.0


class FakeStockfish:
    id = {'name': 'Stockfish fixture'}

    def analyse(self, board, limit, multipv):
        moves = ['e2e4', 'd2d4'] if board.turn else ['e7e5', 'c7c5']
        def wdl(wins, draws, losses):
            value = chess.engine.Wdl(wins, draws, losses)
            return chess.engine.PovWdl(value, board.turn)
        return [{'pv': [chess.Move.from_uci(moves[0])], 'wdl': wdl(900, 50, 50)},
                {'pv': [chess.Move.from_uci(moves[1])], 'wdl': wdl(600, 200, 200)}]


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_teacher_replaces_bad_human_policy_and_result_for_both_sides(self):
        for ply in (0, 1):
            original = example(ply)
            labeled = stockfish_labels(original, FakeStockfish(), nodes=10, multipv=2)
            self.assertEqual(labeled['label_kind'], 'teacher_wdl')
            self.assertEqual(labeled['wdl'], [0.9, 0.05, 0.05])
            self.assertEqual(original['label_kind'], 'game_result')
            self.assertEqual(labeled['teacher']['limits'], {'nodes': 10})
            self.assertAlmostEqual(sum(labeled['policy'].values()), 1)
            if ply == 0:
                self.assertNotIn('h2h3', labeled['policy'])
                self.assertGreater(labeled['policy']['e2e4'], labeled['policy']['d2d4'])
            validate_record(labeled)

    def test_teacher_requires_native_wdl_instead_of_fabricating_draw_mass(self):
        engine = FakeStockfish()
        engine.analyse = lambda *args, **kwargs: [{'pv': [chess.Move.from_uci('e2e4')]}]
        with self.assertRaisesRegex(ValueError, 'UCI_ShowWDL'):
            stockfish_labels(example(), engine)

    def test_puct_visits_are_legal_normalizable_and_mate_has_correct_sign(self):
        board = chess.Board(MATE)
        visits = search(board, uniform, simulations=128)
        self.assertEqual(sum(visits.values()), 128)
        self.assertEqual(set(visits), {move.uci() for move in board.legal_moves})
        self.assertEqual(choose_move(visits), 'd1d8')
        self.assertEqual(board.fen(), MATE)

    def test_selfplay_has_visit_targets_and_only_real_game_outcomes(self):
        with patch('inez.curriculum.NeuralEvaluator', return_value=uniform):
            records, report = selfplay_records(None, games=1, simulations=128, max_plies=1,
                    seed=3, excluded=set(), initial_fen=MATE)
            self.assertEqual(report['completed_games'], 1)
            self.assertTrue(records)
            self.assertEqual(records[0]['label_kind'], 'search')
            self.assertEqual(records[0]['wdl'], [1, 0, 0])
            self.assertAlmostEqual(sum(records[0]['policy'].values()), 1)
            self.assertGreater(len(records[0]['policy']), 1)
            validate_record(records[0])
            records, report = selfplay_records(None, games=1, simulations=1, max_plies=1,
                                               seed=3, excluded=set())
            self.assertEqual(records, [])
            self.assertEqual(report['unfinished_games_discarded'], 1)

    def test_promotion_rejects_regression_and_uncertain_or_equal_games(self):
        previous = [{'name': 'mate', 'solved': True}]
        worse = [{'name': 'mate', 'solved': False}]
        self.assertFalse(promotion_decision(16, 16, worse, previous)['passed'])
        self.assertFalse(promotion_decision(8, 16, previous, previous)['passed'])
        self.assertFalse(promotion_decision(1, 1, previous, previous)['passed'])
        self.assertTrue(promotion_decision(16, 16, previous, previous)['passed'])

    def test_importer_preserves_histories_and_independent_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'games.pgn'
            rng = random.Random(5)
            games = []
            for i in range(80):
                board = chess.Board()
                game = chess.pgn.Game()
                game.headers.update(WhiteElo='2500', BlackElo='2400', Result='1-0')
                node = game
                for ply in range(16):
                    if board.outcome(claim_draw=False) is not None:
                        break
                    move = rng.choice(list(board.legal_moves))
                    node = node.add_variation(move)
                    board.push(move)
                outcome = board.outcome(claim_draw=False)
                if outcome:
                    game.headers['Result'] = outcome.result()
                games.append(str(game))
            path.write_text('\n\n'.join(games))
            manifest = import_games(path, root / 'dataset', uri='fixture://games', license_name='fixture',
                tactics=ROOT / 'tests/fixtures/independent_tactics.json', max_games=80,
                samples_per_game=2, min_rating=2300, seed=5)
            self.assertEqual(manifest['audit']['cross_split_positions'], 0)
            self.assertEqual(manifest['audit']['duplicate_samples'], 0)
            owners = {}
            for split in ('train', 'validation', 'evaluation'):
                records = load_records([root / 'dataset' / (split + '.jsonl')])
                self.assertTrue(records)
                for record in records:
                    self.assertGreaterEqual(len(validate_record(record).move_stack), 8)
                    self.assertEqual(owners.setdefault(record['game_id'], split), split)

    def test_resume_restores_optimizer_batch_order_and_dropout_rng(self):
        config = ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4, dropout=0.2)
        settings = dict(steps=3, batch_size=2, lr=0.001, weight_decay=0.0001,
                        clip_grad=1.0, checkpoint_every=3, log_every=10, seed=7)
        records = [example(0), example(1)]
        def gate(candidate, report):
            torch.rand(5)  # Evaluation must leave the training RNG untouched.
            atomic_json(report, {'passed': False})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(11)
            uninterrupted = ChessTransformer(config)
            self.assertTrue(train_stage(uninterrupted, records, records, root / 'full',
                        config=settings, stage='human', gate=gate))
            torch.manual_seed(11)
            interrupted = ChessTransformer(config)
            class Stop:
                requested = True
            self.assertFalse(train_stage(interrupted, records, records, root / 'resume',
                        config=settings, stage='human', gate=gate, stop=Stop()))
            restored = ChessTransformer(config)
            self.assertTrue(train_stage(restored, records, records, root / 'resume',
                        config=settings, stage='human', gate=gate))
            for name, value in uninterrupted.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
            with self.assertRaisesRegex(ValueError, 'configuration or dataset changed'):
                train_stage(restored, records[:1], records, root / 'resume',
                            config=settings, stage='human', gate=gate)

    def test_later_stages_reject_human_imitation_labels(self):
        model = ChessTransformer(ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4))
        settings = dict(steps=1, batch_size=1, lr=0.001, weight_decay=0, clip_grad=1,
                        checkpoint_every=1, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            for stage in ('teacher', 'selfplay'):
                with self.assertRaises(ValueError):
                    train_stage(model, [example()], [example()], tmp, config=settings,
                                stage=stage, gate=lambda *args: None)

    def test_slurm_warning_reaches_child_and_preserves_checkpoint_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'python'
            fake.write_text(f'''#!{sys.executable}
import os, pathlib, signal, sys, time
if 'preflight' in sys.argv:
    raise SystemExit(0)
def stop(signum, frame):
    pathlib.Path(os.environ['INEZ_ROOT'], 'checkpoint-saved').write_text('saved')
    raise SystemExit(75)
signal.signal(signal.SIGUSR1, stop)
pathlib.Path(os.environ['INEZ_ROOT'], 'ready').write_text('ready')
while True:
    time.sleep(0.05)
''')
            fake.chmod(0o755)
            env = {**os.environ, 'INEZ_ROOT': str(root), 'INEZ_PYTHON': str(fake),
                   'RUN_DIR': str(root / 'run')}
            proc = subprocess.Popen(['bash', str(ROOT / 'hpc/train_29m.slurm')], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 10
                while not (root / 'ready').exists() and time.monotonic() < deadline:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.02)
                if not (root / 'ready').exists():
                    out, err = proc.communicate(timeout=5)
                    self.fail('training child did not start: ' + out + err)
                proc.send_signal(signal.SIGUSR1)
                out, err = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 75, out + err)
                self.assertTrue((root / 'checkpoint-saved').exists())
                self.assertIn('exit=75', (root / 'run/run.log').read_text())
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()

    def test_rejected_candidate_cannot_replace_accepted_model(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        from train_inez import ChampionGate
        model_config = ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4)
        class Stop:
            requested = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ChessTransformer(model_config)
            for name in ('initial', 'good', 'bad'):
                save_checkpoint(model, root / name)
            initial = str((root / 'initial').resolve())
            atomic_json(root / 'accepted.json', {'artifact': initial, 'history': [initial]})
            config = {'tactics': str(ROOT / 'tests/fixtures/independent_tactics.json'),
                      'gate': {'historical_opponents': 2, 'openings': [[]], 'pairs': 1,
                               'simulations': 1, 'max_plies': 8, 'minimum_score': 0.55}}
            gate = ChampionGate(root, config, model_config, torch.device('cpu'), Stop())
            with patch('train_inez.evaluate_candidate', return_value={'passed': True, 'games': []}):
                gate(root / 'good', root / 'good-report.json')
            good = json.loads((root / 'accepted.json').read_text())
            self.assertEqual(good['artifact'], str((root / 'good').resolve()))
            self.assertEqual(len(good['history']), 2)
            with patch('train_inez.evaluate_candidate', return_value={'passed': False, 'games': []}):
                gate(root / 'bad', root / 'bad-report.json')
            self.assertEqual(json.loads((root / 'accepted.json').read_text()), good)
            # Simulate a crash after the promotion report, before its pointer.
            atomic_json(root / 'accepted.json', {'artifact': initial, 'history': [initial]})
            gate(root / 'good', root / 'good-report.json')
            self.assertEqual(json.loads((root / 'accepted.json').read_text()), good)


if __name__ == '__main__':
    unittest.main()
