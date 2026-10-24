#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Full-pool replay, artifact metrics, and exact optimizer/EMA resume."""
import copy
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import chess
import torch
from inez.model import ChessTransformer, ModelConfig, load_checkpoint
from inez.trainer import (sample_batch_indices, train_stage, validation_metrics)


def record(kind='search', source='selfplay'):
    return {'record_version': 1, 'encoding_version': 1, 'source_id': source,
            'game_id': source, 'initial_fen': chess.STARTING_FEN,
            'moves': ['e2e4', 'e7e5'], 'ply': 0, 'result': '1-0',
            'policy': {'e2e4': 1.0}, 'wdl': [1.0, 0.0, 0.0],
            'label_kind': kind, 'teacher': {'name': 'fixture', 'version': '1', 'limits': {'nodes': 1}}}


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_teacher_pool_is_reachable_and_priority_cannot_change_mix(self):
        rng = torch.Generator().manual_seed(7)
        seen, count = set(), 0
        for _ in range(200):
            indices, replay = sample_batch_indices(2, 10, rng, primary_table=[0] * 100 + [1],
                replay_size=100, replay_table=list(range(100)), replay_fraction=0.8)
            self.assertEqual(replay, 8)
            self.assertEqual(sum(index >= 2 for index in indices), 8)
            count += replay
            seen.update(index - 2 for index in indices if index >= 2)
        self.assertEqual(count, 1600)
        self.assertEqual(seen, set(range(100)))

    def test_saved_generator_reproduces_mixture_and_fraction_is_unbiased(self):
        rng = torch.Generator().manual_seed(11)
        kwargs = dict(replay_size=50, replay_fraction=0.8)
        state = rng.get_state()
        first = [sample_batch_indices(3, 1, rng, **kwargs) for _ in range(1000)]
        rng.set_state(state)
        self.assertEqual(first, [sample_batch_indices(3, 1, rng, **kwargs) for _ in range(1000)])
        self.assertLess(abs(sum(n for _, n in first) / 1000 - 0.8), 0.04)
        for kwargs in ({'replay_size': 0, 'replay_fraction': 0.8},
                       {'replay_size': 1, 'replay_fraction': float('nan')},
                       {'replay_size': 1}):
            with self.assertRaises(ValueError):
                sample_batch_indices(3, 1, rng, **kwargs)

    def test_metrics_have_independent_policy_and_value_denominators(self):
        class Constant(torch.nn.Module):
            def forward(self, features):
                return torch.zeros(len(features), 4288), torch.zeros(len(features), 3)
        with_policy, without = record(), record()
        without['policy'] = {}
        metrics = validation_metrics(Constant(), [with_policy, without], 2, 'cpu', limit=None)
        self.assertAlmostEqual(metrics['policy_loss'], math.log(20), places=5)
        self.assertAlmostEqual(metrics['wdl_cross_entropy'], math.log(3), places=5)
        self.assertAlmostEqual(metrics['expected_value_mse'], 1.0, places=5)
        self.assertAlmostEqual(metrics['expected_value_mae'], 1.0, places=5)
        self.assertAlmostEqual(metrics['wdl_squared_distance'], 2/3, places=5)
        self.assertEqual((metrics['policy_records'], metrics['records']), (1, 2))
        self.assertAlmostEqual(metrics['loss'], metrics['policy_loss'] + metrics['value_loss'])

    def test_continuation_settings_require_initialization_and_budget_candidates(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        from train_inezita import read_config, promotion_budget, pipeline
        from types import SimpleNamespace
        root = Path(__file__).resolve().parents[1]
        for name, candidates in [('inezita-0.1.3', 8), ('inezita-0.1.3-expanded', 10)]:
            config, model_config = read_config(root / 'configs/train' / (name + '.json'))
            budget = promotion_budget(config)
            self.assertEqual(budget['planned_candidates'], candidates)
            self.assertAlmostEqual(budget['candidate_alpha'] * candidates, 0.05)
            with self.assertRaisesRegex(ValueError, 'requires --init-artifact'):
                pipeline(SimpleNamespace(init_artifact=None), config, model_config)

    def test_exported_ema_metrics_and_raw_optimizer_state_survive_resume(self):
        config = ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4, dropout=0.2)
        settings = dict(steps=100, batch_size=2, lr=0.001, weight_decay=0,
                        clip_grad=1, checkpoint_every=50, log_every=1000,
                        seed=9, validation_limit=None)
        primary = [record()]
        teacher = [record('teacher_wdl', 'teacher-' + str(i)) for i in range(8)]
        def gate(candidate, report):
            torch.rand(3)
        class Stop:
            requested = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def new_model():
                torch.manual_seed(31)
                return ChessTransformer(config)
            full = new_model()
            train_stage(full, primary, teacher, root / 'full', config=settings,
                stage='selfplay', gate=gate, replay=teacher, replay_fraction=0.8)
            stop = Stop()
            from inez.trainer import _update_ema_state
            def update(*args, **kwargs):
                _update_ema_state(*args, **kwargs)
                if args[2] == 50:
                    stop.requested = True
            with patch('inez.trainer._update_ema_state', side_effect=update):
                self.assertFalse(train_stage(new_model(), primary, teacher, root / 'resumed',
                    config=settings, stage='selfplay', gate=gate, stop=stop,
                    replay=teacher, replay_fraction=0.8))
            resumed = new_model()
            train_stage(resumed, primary, teacher, root / 'resumed', config=settings,
                stage='selfplay', gate=gate, replay=teacher, replay_fraction=0.8)
            a = torch.load(root / 'full/last.pt', weights_only=True)
            b = torch.load(root / 'resumed/last.pt', weights_only=True)
            for key in ('model', 'ema_model'):
                for name, value in a[key].items():
                    self.assertTrue(torch.equal(value, b[key][name]), key + ':' + name)
            self.assertEqual(a['sampler_counts'], b['sampler_counts'])
            self.assertTrue(torch.equal(a['sampler_rng'], b['sampler_rng']))
            self.assertTrue(any(not torch.equal(a['model'][key], a['ema_model'][key]) for key in a['model']))
            for step in (50, 100):
                artifact = root / 'full' / f'candidate-{step:07d}'
                metrics = json.loads((root / 'full' / f'validation-{step:07d}.json').read_text())
                manifest = json.loads((artifact / 'manifest.json').read_text())
                self.assertEqual(metrics['weights_sha256'], manifest['weights_sha256'])
                self.assertEqual(metrics['weights_kind'], 'ema')
                actual = validation_metrics(load_checkpoint(artifact), teacher, 2, 'cpu', limit=None)
                self.assertAlmostEqual(metrics['loss'], actual['loss'], places=7)
            exported = load_checkpoint(root / 'full/candidate-0000100').state_dict()
            for name, value in a['ema_model'].items():
                self.assertTrue(torch.equal(value, exported[name]), name)
            changed = copy.deepcopy(teacher)
            changed[-1]['wdl'] = [0, 1, 0]
            with self.assertRaisesRegex(ValueError, 'configuration or dataset changed'):
                train_stage(resumed, primary, teacher, root / 'resumed', config=settings,
                    stage='selfplay', gate=gate, replay=changed, replay_fraction=0.8)


if __name__ == '__main__':
    unittest.main()
