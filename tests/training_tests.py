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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import chess
import chess.engine
import torch

from inez.curriculum import (atomic_json, import_games, position_key, promotion_decision,
                            selfplay_records, sha256, stockfish_labels)
from inez.model import ChessTransformer, ModelConfig, save_checkpoint
from inez.puct import choose_move, search
from inez.records import load_records, validate_record
from inez.trainer import StopRequested, restore_checkpoint_rng, train_stage

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

    def test_portable_resume_preserves_progress_optimizer_and_sampler(self):
        config = ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4, dropout=0.2)
        settings = dict(steps=3, batch_size=2, lr=0.001, weight_decay=0.0001,
                        clip_grad=1.0, checkpoint_every=3, log_every=10, seed=7)
        records = [example(0), example(1)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(11)
            full = ChessTransformer(config)
            train_stage(full, records, records, root / 'full', config=settings,
                        stage='human', gate=lambda *args: None)
            torch.manual_seed(11)
            interrupted = ChessTransformer(config)
            class Stop:
                requested = True
            train_stage(interrupted, records, records, root / 'resume', config=settings,
                        stage='human', gate=lambda *args: None, stop=Stop())
            path = root / 'resume/last.pt'
            state = torch.load(path, weights_only=True)
            state['context']['torch'] = 'old-build'
            torch.save(state, path)
            restored = ChessTransformer(config)
            with self.assertRaisesRegex(ValueError, 'torch: checkpoint=.+old-build.+RESUME_MODE=portable'):
                train_stage(restored, records, records, root / 'resume', config=settings,
                            stage='human', gate=lambda *args: None)
            # Failed exact resume must not replace the existing checkpoint.
            self.assertEqual(torch.load(path, weights_only=True)['step'], 1)
            self.assertTrue(train_stage(restored, records, records, root / 'resume', config=settings,
                stage='human', gate=lambda *args: None, resume_mode='portable'))
            for name, value in full.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
            saved = torch.load(path, weights_only=True)
            baseline = torch.load(root / 'full/last.pt', weights_only=True)
            self.assertEqual(saved['step'], settings['steps'])
            self.assertTrue(torch.equal(saved['sampler_rng'], baseline['sampler_rng']))
            self.assertEqual(saved['resume_history'][0]['differences']['torch']['checkpoint'], 'old-build')
            self.assertEqual(saved['context']['torch'], str(torch.__version__))
            with self.assertRaisesRegex(ValueError, 'configuration or dataset changed'):
                train_stage(restored, records[:1], records, root / 'resume', config=settings,
                            stage='human', gate=lambda *args: None, resume_mode='portable')

    def test_cuda_migration_reseeds_instead_of_restoring_incompatible_rng(self):
        state = {'context': {'device': 'cuda:0', 'gpu_name': 'Original GPU',
                             'torch': str(torch.__version__)},
                 'cuda_rng': [torch.zeros(1, dtype=torch.uint8)] * 2,
                 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(), 'step': 13}
        with patch('torch.cuda.get_device_name', return_value='NVIDIA A100-SXM4-40GB'), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.set_rng_state_all') as restore, \
             patch('torch.cuda.manual_seed_all') as reseed:
            kwargs = dict(seed=7, checkpoint='selfplay-001/last.pt')
            with self.assertRaisesRegex(ValueError, 'gpu_name:.*cuda_device_count:'):
                restore_checkpoint_rng(state, torch.device('cuda:0'), resume_mode='exact', **kwargs)
            event = restore_checkpoint_rng(state, torch.device('cuda:0'), resume_mode='portable', **kwargs)
            reseed.assert_called_once_with(20)
            restore.assert_not_called()
            self.assertEqual(set(event['differences']), {'gpu_name', 'cuda_device_count'})
            self.assertEqual(event['cuda_reseed'], 20)
            # Matching environments, including legacy checkpoints, keep exact CUDA RNGs.
            state['context']['gpu_name'] = 'NVIDIA A100-SXM4-40GB'
            state['cuda_rng'] = state['cuda_rng'][:1]
            reseed.reset_mock()
            event = restore_checkpoint_rng(state, torch.device('cuda:0'), resume_mode='exact', **kwargs)
            restore.assert_called_once_with(state['cuda_rng'])
            reseed.assert_not_called()
            self.assertEqual(event['differences'], {})

    def test_pipeline_portable_resume_skips_completed_stages_and_reuses_selfplay(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        from train_inez import pipeline
        model_config = ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4)
        config = json.loads((ROOT / 'configs/train/inez-29m.json').read_text())
        config['training'].update(human_steps=1, teacher_steps=1, selfplay_steps=2,
                                  batch_size=1, checkpoint_every=2, selfplay_lr=0.00003)
        config['selfplay']['rounds'] = 2
        teacher = stockfish_labels(example(), FakeStockfish(), nodes=10, multipv=2)
        search_record = {**example(), 'label_kind': 'search',
                         'teacher': {'name': 'fixture PUCT', 'version': '1', 'limits': {'nodes': 2}}}
        class Stop:
            requested = True
        fresh_rounds = []
        def check_initialization(*args, **kwargs):
            output = Path(args[3])
            if kwargs['stage'] == 'selfplay' and not (output / 'last.pt').exists():
                # All gates in this fixture reject, so both rounds must start
                # from initial rather than the rejected teacher/self-play tail.
                baseline = torch.load(output.parent / 'initial/weights.pt', weights_only=True)
                actual = args[0].state_dict()
                for name, value in baseline.items():
                    self.assertTrue(torch.equal(value, actual[name]), name)
                previous = 'teacher' if output.name == 'selfplay-001' else 'selfplay-001'
                rejected = torch.load(output.parent / previous / 'last.pt', weights_only=True)['model']
                self.assertTrue(any(not torch.equal(value, rejected[name]) for name, value in baseline.items()))
                fresh_rounds.append(output.name)
            return train_stage(*args, **kwargs)
        def interrupt_selfplay(*args, **kwargs):
            if kwargs['stage'] == 'selfplay':
                kwargs['stop'] = Stop()
            return check_initialization(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run = root / 'data', root / 'run'
            (data / 'human').mkdir(parents=True)
            (data / 'teacher').mkdir()
            atomic_json(data / 'human/manifest.json', {})
            atomic_json(data / 'teacher/manifest.json',
                        {'teacher': {'human_manifest_sha256': sha256(data / 'human/manifest.json')}})
            args = SimpleNamespace(device='cpu', init_artifact=None, run_dir=str(run),
                                   data_dir=str(data), resume_mode='exact')
            def dataset(directory, split):
                return [teacher if Path(directory).name == 'teacher' else example()]
            with patch('train_inez.dataset_records', side_effect=dataset), \
                 patch('train_inez.ChampionGate', return_value=lambda *args: None), \
                 patch('train_inez.selfplay_records', return_value=([search_record], {})) as generate:
                with patch('train_inez.train_stage', side_effect=interrupt_selfplay):
                    with self.assertRaises(InterruptedError):
                        pipeline(args, config, model_config)
                self.assertEqual(json.loads((run / 'curriculum.json').read_text())['completed'],
                                 ['human', 'teacher'])
                replay_hash = sha256(run / 'selfplay-001.jsonl')
                for name in ('human', 'teacher', 'selfplay-001'):
                    path = run / name / 'last.pt'
                    state = torch.load(path, weights_only=True)
                    state['context']['torch'] = 'old-build'
                    torch.save(state, path)
                args.resume_mode = 'portable'
                with patch('train_inez.train_stage', side_effect=check_initialization) as stages:
                    pipeline(args, config, model_config)
                self.assertEqual(stages.call_count, 2)
                self.assertEqual(stages.call_args.kwargs['stage'], 'selfplay')
                self.assertEqual(stages.call_args.kwargs['resume_mode'], 'portable')
                self.assertEqual(stages.call_args.kwargs['config']['lr'], 0.00003)
                self.assertEqual(fresh_rounds, ['selfplay-001', 'selfplay-002'])
                self.assertEqual(generate.call_count, 2)
                self.assertEqual(sha256(run / 'selfplay-001.jsonl'), replay_hash)
                self.assertEqual(json.loads((run / 'curriculum.json').read_text())['completed'],
                                 ['human', 'teacher', 'selfplay-001', 'selfplay-002'])
                final = torch.load(run / 'selfplay-001/last.pt', weights_only=True)
                self.assertEqual(final['step'], 2)
                self.assertEqual(final['resume_history'][0]['step'], 1)
                with patch('train_inez.train_stage') as stages:
                    pipeline(args, config, model_config)
                stages.assert_not_called()
                self.assertEqual(generate.call_count, 2)

    def test_gate_summary_explains_rejection_from_existing_reports(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        from train_inez import gate_summary
        result = {'passed': False, 'opponents': [{'opponent': '/accepted', 'passed': False,
            'score': 0.5, 'score_lower_bound': 0.1, 'completed_games': 2,
            'games': [{'candidate_score': 1.0}, {'candidate_score': 0.0}, {'candidate_score': None}],
            'tactical_regressions': ['mate']}]}
        summary = gate_summary(result, '/gate.json', minimum_score=0.55, cached=True)
        opponent = summary['opponents'][0]
        self.assertFalse(summary['promoted'])
        self.assertTrue(summary['cached'])
        self.assertEqual(opponent['rejection_reasons'], ['unfinished_match_games',
            'score_below_minimum', 'insufficient_match_evidence', 'tactical_regression'])
        self.assertEqual((opponent['wins'], opponent['draws'], opponent['losses'],
                          opponent['unfinished_games']), (1, 0, 1, 1))
        result['opponents'][0].update(score=0.75, score_lower_bound=0.3,
            completed_games=4, tactical_regressions=[],
            games=[{'candidate_score': score} for score in (1.0, 1.0, 1.0, 0.0)])
        summary = gate_summary(result, '/gate.json', minimum_score=0.55)
        self.assertEqual(summary['opponents'][0]['rejection_reasons'], ['insufficient_match_evidence'])

    def test_status_cli_reports_completed_run_without_cuda_or_external_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {'selfplay': {'rounds': 1}, 'gate': {'minimum_score': 0.55}}
            atomic_json(root / 'request.json', {'config': config, 'initial_artifact': None})
            atomic_json(root / 'curriculum.json', {'completed': ['human', 'teacher', 'selfplay-001']})
            accepted = {'artifact': '/accepted/teacher/candidate-0001000', 'history': []}
            atomic_json(root / 'accepted.json', accepted)
            (root / 'selfplay-001').mkdir()
            atomic_json(root / 'selfplay-001/gate-0000500.json', {'passed': False, 'opponents': [
                {'opponent': accepted['artifact'], 'passed': False, 'score': 0.5,
                 'score_lower_bound': 0.25, 'completed_games': 2, 'tactical_regressions': [],
                 'games': [{'candidate_score': 0.5}, {'candidate_score': 0.5}]}]})
            before = {str(path): sha256(path) for path in root.rglob('*') if path.is_file()}
            proc = subprocess.run([sys.executable, str(ROOT / 'tools/train_inez.py'), 'status',
                '--run-dir', str(root), '--config', str(root / 'missing-config.json'), '--device', 'cuda'],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['pending_stages'], [])
            self.assertEqual(result['accepted'], accepted)
            self.assertIn('no further optimizer steps', result['next_action'])
            self.assertEqual(result['gates'][0]['opponents'][0]['rejection_reasons'],
                             ['score_below_minimum', 'insufficient_match_evidence'])
            after = {str(path): sha256(path) for path in root.rglob('*') if path.is_file()}
            self.assertEqual(before, after)

    def test_completed_checkpoint_only_finishes_pending_gate(self):
        model_config = ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4)
        settings = dict(steps=1, batch_size=1, lr=0.001, weight_decay=0,
                        clip_grad=1.0, checkpoint_every=1, seed=7)
        records = [example()]
        def pending_gate(candidate, report):
            raise InterruptedError('gate pending')
        with tempfile.TemporaryDirectory() as tmp:
            model = ChessTransformer(model_config)
            with self.assertRaises(InterruptedError):
                train_stage(model, records, records, tmp, config=settings, stage='human',
                            gate=pending_gate)
            saved = copy.deepcopy(model.state_dict())
            with patch('torch.optim.AdamW.step', side_effect=AssertionError('extra optimizer step')), \
                 patch('inez.trainer.save_checkpoint') as export:
                gates = []
                self.assertTrue(train_stage(ChessTransformer(model_config), records, records, tmp,
                    config=settings, stage='human', gate=lambda *args: gates.append(args)))
                self.assertEqual(len(gates), 1)
                export.assert_not_called()
            state = torch.load(Path(tmp) / 'last.pt', weights_only=True)
            self.assertEqual(state['step'], 1)
            for name, value in saved.items():
                self.assertTrue(torch.equal(value, state['model'][name]), name)

    def test_slurm_warning_reaches_child_and_preserves_checkpoint_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'python'
            fake.write_text(f'''#!{sys.executable}
import os, pathlib, signal, sys, time
if 'preflight' in sys.argv:
    raise SystemExit(0)
expected_mode = os.environ.get('RESUME_MODE', 'exact')
assert sys.argv[sys.argv.index('--resume-mode') + 1] == expected_mode
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
                   'RUN_DIR': str(root / 'run'), 'RESUME_MODE': 'portable'}
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

    def test_stop_requested_handles_c_level_signal_handlers(self):
        with patch('signal.getsignal', return_value=None):
            with self.assertRaises(InterruptedError):
                with StopRequested() as stop:
                    self.assertFalse(stop.requested)
                    stop.handle(signal.SIGUSR1, None)
                    self.assertTrue(stop.requested)
                    raise InterruptedError('checkpoint saved')

    def test_mco_openings_cover_all_volumes_and_hash_to_train_split(self):
        from inez.openings import default_opening_book, load_mco_entries, opening_records
        from inez.records import assign_splits
        entries = load_mco_entries()
        self.assertGreaterEqual(len(entries), 3800)
        volumes = {entry.eco[0] for entry in entries}
        self.assertEqual(volumes, {'A', 'B', 'C', 'D', 'E'})
        book = default_opening_book()
        root_priors = book.priors(chess.Board())
        self.assertIsNotNone(root_priors)
        self.assertGreater(root_priors.get('e2e4', 0.0) + root_priors.get('d2d4', 0.0), 0.85)
        self.assertNotIn('f2f3', root_priors)
        records, meta = opening_records(seed='20261019')
        self.assertGreaterEqual(len(records), 5500)
        self.assertEqual(len(meta), 1)
        splits = assign_splits(records, seed='20261019')
        self.assertEqual(len(splits['train']), len(records))
        self.assertEqual(len(splits['validation']), 0)
        self.assertEqual(len(splits['evaluation']), 0)

    def test_see_tactical_priors_prevent_game3_knight_blunder_and_convert_kqk(self):
        from inez.puct import _leaf_value_adjustment, _loss_on_square, _tactical_priors
        board = chess.Board()
        for move_uci in ('b1c3', 'd7d5', 'g1f3', 'd5d4', 'c3b5', 'b8c6'):
            board.push_uci(move_uci)
        after_blunder = board.copy()
        after_blunder.push_uci('b5d4')
        self.assertGreaterEqual(_loss_on_square(after_blunder, chess.D4, chess.WHITE), 3.0)
        uniform_priors = {m.uci(): 1.0 / board.legal_moves.count() for m in board.legal_moves}
        adjusted = _tactical_priors(board, uniform_priors)
        self.assertLess(adjusted['b5d4'], 0.002)
        self.assertGreater(adjusted['e2e3'], adjusted['b5d4'] * 20)

        kqk = chess.Board('1k3K2/8/8/8/8/8/6q1/8 b - - 0 1')
        def kqk_eval(pos):
            raw = {m.uci(): 1.0 / pos.legal_moves.count() for m in pos.legal_moves}
            return _tactical_priors(pos, raw), _leaf_value_adjustment(pos, 0.0)
        for _ in range(40):
            if kqk.is_game_over(claim_draw=True):
                break
            kqk.push_uci(choose_move(search(kqk, kqk_eval, simulations=16)))
        self.assertTrue(kqk.is_checkmate(), f'Expected checkmate in K+Q vs K, got {kqk.fen()}')

    def test_importer_supports_multi_archive_mco_and_blunder_counterfactuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1, p2 = root / 'a.pgn', root / 'b.pgn'
            rng = random.Random(19)
            for idx, path in enumerate((p1, p2)):
                games = []
                for _ in range(40):
                    board = chess.Board()
                    game = chess.pgn.Game()
                    game.headers.update(WhiteElo='2500', BlackElo='2450', Result='1-0')
                    node = game
                    for _ in range(20):
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
            mini_mco = root / 'mini_mco.tsv'
            mini_mco.write_text(
                'eco\tname\tpgn\tuci\n'
                'C20\tKing Pawn\t1. e4 e5\te2e4 e7e5\n'
                'D00\tQueen Pawn\t1. d4 d5\td2d4 d7d5\n',
                encoding='utf-8',
            )
            manifest = import_games(
                [p1, p2], root / 'dataset',
                uri=['fixture://a', 'fixture://b'],
                license_name='fixture',
                tactics=ROOT / 'tests/fixtures/independent_tactics.json',
                max_games=60, samples_per_game=2, min_ply=0, min_rating=2300,
                seed=20261019, openings_path=mini_mco, blunder_samples_per_game=1,
            )
            self.assertEqual(manifest['audit']['cross_split_positions'], 0)
            self.assertGreaterEqual(manifest['preparation']['mco_opening_records'], 3)
            self.assertGreater(manifest['preparation']['blunder_counterfactual_records'], 0)

    def test_paper_upgrades_surprise_sampling_history_masking_and_soft_wdl(self):
        from inez.curriculum import _soft_wdl_from_scalar
        from inez.trainer import (_apply_history_mask, _hl_gauss_wdl,
                                  _priority_sampling_table, _update_ema_state)
        w, d, l = _soft_wdl_from_scalar(0.6)
        self.assertAlmostEqual(w + d + l, 1.0)
        self.assertAlmostEqual(w - l, 0.6)

        raw_wdl = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        smoothed_wdl = _hl_gauss_wdl(raw_wdl, blend=0.20)
        self.assertTrue(torch.allclose(smoothed_wdl.sum(dim=-1), torch.ones(2)))
        self.assertGreater(float(smoothed_wdl[0, 1].item()), float(smoothed_wdl[0, 2].item()) * 10.0)
        self.assertGreater(float(smoothed_wdl[1, 1].item()), float(smoothed_wdl[1, 0].item()) * 10.0)

        model = ChessTransformer(ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4))
        ema = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        _update_ema_state(ema, model, step=16)
        self.assertFalse(torch.allclose(ema['input_projection.weight'],
                                        torch.zeros_like(model.input_projection.weight)))

        rec_routine = {**example(ply=0), 'policy': {'f2f3': 0.95, 'e2e4': 0.05},
                       'label_kind': 'teacher_wdl',
                       'teacher': {'name': 'Stockfish', 'version': '17.1', 'limits': {'nodes': 100}}}
        rec_surprise = {**example(ply=0), 'policy': {'f2f3': 0.05, 'e2e4': 0.95},
                        'label_kind': 'teacher_wdl',
                        'teacher': {'name': 'Stockfish', 'version': '17.1', 'limits': {'nodes': 100}}}
        table = _priority_sampling_table([rec_routine, rec_surprise])
        self.assertGreater(table.count(1), table.count(0))

        features = torch.ones(64, 64, 119)
        masked = _apply_history_mask(features, step=7, seed=20261019)
        self.assertTrue(torch.equal(features, torch.ones(64, 64, 119)))
        self.assertTrue(torch.equal(masked[:, :, :11], torch.ones(64, 64, 11)))
        self.assertLess(float(masked[:, :, 14:112].sum().item()), float(features[:, :, 14:112].sum().item()))


if __name__ == '__main__':
    unittest.main()
