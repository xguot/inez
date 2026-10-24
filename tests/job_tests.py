#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Submission contracts, frozen inputs, initialization and resume scheduling."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'training'))
import chess
import torch
import submit_training as jobs
from inez.curriculum import atomic_json, sha256
from inez.jobs import verify_job_spec
from inez.model import ChessTransformer, ModelConfig, save_checkpoint
from train_inezita import verify_prepared_data


class JobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, folder):
        config = json.loads((ROOT / 'configs/train/inezita-0.1.3.json').read_text())
        model_config = ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4)
        model_json = folder / 'model.json'
        atomic_json(model_json, vars(model_config))
        config['model_config'] = str(model_json)
        config_path = folder / 'recipe.json'
        atomic_json(config_path, config)
        accepted = folder / 'prior'
        artifact = accepted / 'teacher/candidate-0000001'
        save_checkpoint(ChessTransformer(model_config), artifact)
        atomic_json(accepted / 'accepted.json', {'artifact': str(artifact), 'history': [str(artifact)]})
        data = folder / 'data'
        human = data / 'human'
        teacher = data / 'teacher'
        human.mkdir(parents=True)
        teacher.mkdir()
        manifest = {'manifest_version': 1, 'record_version': 1, 'encoding_version': 1,
                    'audit': {'cross_split_positions': 0, 'duplicate_samples': 0}, 'splits': {},
                    'preparation': {key: config['preparation'][key] for key in
                        ('min_rating','min_ply','samples_per_game','max_games','blunder_samples_per_game')}}
        manifest['preparation'].update(tactics_sha256=sha256(ROOT / config['tactics']),
            openings_sha256=sha256(ROOT / config['preparation']['openings']))
        for split in ('train','validation','evaluation'):
            rec = {'record_version': 1, 'encoding_version': 1, 'source_id': 'fixture', 'game_id': split,
                'initial_fen': chess.STARTING_FEN, 'moves': ['e2e4','e7e5'], 'ply': 0,
                'result': '1-0', 'policy': {'e2e4': 1}, 'wdl': [1,0,0],
                'label_kind': 'game_result', 'teacher': None}
            path = human / (split + '.jsonl')
            path.write_text(json.dumps(rec) + '\n')
            manifest['splits'][split] = {'records': 1, 'sha256': sha256(path)}
        atomic_json(human / 'manifest.json', manifest)
        tm = copy.deepcopy(manifest)
        for split in ('train','validation','evaluation'):
            rec = json.loads((human / (split + '.jsonl')).read_text())
            rec.update(label_kind='teacher_wdl', teacher={'name':'Stockfish','version':'fixture','limits':{'nodes':10000}})
            path = teacher / (split + '.jsonl')
            path.write_text(json.dumps(rec) + '\n')
            tm['splits'][split] = {'records': 1, 'sha256': sha256(path)}
        tm['teacher'] = {'human_manifest_sha256': sha256(human / 'manifest.json'),
            'nodes':10000,'multipv':4,'temperature':0.15,'binary_sha256':'fixture'}
        atomic_json(teacher / 'manifest.json', tm)
        env = {'INIT_ARTIFACT':str(artifact),'RUN_DIR':str(folder/'run'),
               'DATA_DIR':str(data),'TRAIN_CONFIG':str(config_path)}
        return env, config_path, artifact

    def scheduler(self):
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs.get('env',{})))
            if command[0] == 'git':
                return SimpleNamespace(stdout=('a' * 40 if 'rev-parse' in command else ''), returncode=0)
            if command[0] == 'squeue':
                return SimpleNamespace(stdout='', returncode=0)
            self.assertEqual(command[0], 'sbatch')
            count = sum(command[0]=='sbatch' for command, _ in calls)
            return SimpleNamespace(stdout=str(1000+count)+';cluster\n', returncode=0)
        return calls, run

    def test_dry_run_validates_without_creating_output_or_submitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _, _ = self.fixture(Path(tmp))
            with patch.object(jobs.subprocess, 'run') as submit:
                plan = jobs.new_job('continue', dry_run=True, environment=env)
            submit.assert_not_called()
            self.assertEqual(plan['mode'],'continue')
            self.assertFalse(Path(env['RUN_DIR']).exists())
            env.pop('INIT_ARTIFACT')
            with self.assertRaisesRegex(ValueError,'requires INIT_ARTIFACT'):
                jobs.new_job('continue', dry_run=True, environment=env)

    def test_submission_uses_frozen_sources_and_resume_ignores_stale_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, config, _ = self.fixture(Path(tmp))
            calls, run = self.scheduler()
            with patch.object(jobs.subprocess, 'run', side_effect=run):
                spec = jobs.new_job('continue', environment=env)
                config.write_text('{}')  # Original recipe changes cannot affect the snapshot.
                resumed = jobs.dispatch(Path(env['RUN_DIR'])/'job.json', environment={'INIT_ARTIFACT':'/wrong',
                    'TRAIN_CONFIG':'/wrong','DATA_DIR':'/wrong','RUN_DIR':'/wrong','RESUME_MODE':'portable'})
            commands = [command for command, _ in calls if command[0]=='sbatch']
            self.assertEqual(len(commands),2)
            self.assertTrue(all(command[-1].startswith(spec['source_root']) for command in commands))
            exported = [e for command,e in calls if command[0]=='sbatch'][-1]
            self.assertEqual(exported['INIT_ARTIFACT'], spec['initial_artifact'])
            self.assertEqual(exported['TRAIN_CONFIG'], spec['config'])
            self.assertEqual(exported['RESUME_MODE'],'portable')
            self.assertEqual(resumed['submission']['train_job'],'1002')
            verify_job_spec(Path(env['RUN_DIR'])/'job.json')
            Path(spec['source_root'],'training/inez/trainer.py').write_text('# drift')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):
                jobs.dispatch(Path(env['RUN_DIR'])/'job.json', environment=env)

    def test_changed_data_or_initial_weights_cannot_resume(self):
        for change in ('data','weights','config'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                env, _, artifact = self.fixture(Path(tmp))
                calls, run = self.scheduler()
                with patch.object(jobs.subprocess,'run',side_effect=run):
                    spec = jobs.new_job('continue',environment=env)
                if change=='data':
                    Path(env['DATA_DIR'],'teacher/train.jsonl').write_text('{}\n')
                elif change=='weights':
                    (artifact/'weights.pt').write_bytes(b'changed')
                else:
                    Path(spec['config']).write_text('{}')
                with patch.object(jobs.subprocess,'run') as submit, self.assertRaises(ValueError):
                    jobs.dispatch(Path(env['RUN_DIR'])/'job.json',environment=env)
                submit.assert_not_called()

    def test_prepare_dependency_and_failed_submission_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _, _ = self.fixture(Path(tmp))
            calls, run = self.scheduler()
            with patch.object(jobs.subprocess,'run',side_effect=run):
                spec = jobs.new_job('continue',prepare=True,environment=env)
            submitted = [command for command,_ in calls if command[0]=='sbatch']
            self.assertEqual(len(submitted),2)
            self.assertIn('--dependency=afterok:1001',submitted[1])
            self.assertIn('--kill-on-invalid-dep=yes',submitted[1])
            with patch.object(jobs.subprocess,'run',side_effect=[SimpleNamespace(stdout=''),
                    subprocess.CalledProcessError(1,['sbatch'])]):
                with self.assertRaises(subprocess.CalledProcessError):
                    jobs.dispatch(Path(env['RUN_DIR'])/'job.json',environment=env)
            events=json.loads(Path(env['RUN_DIR'],'submissions.json').read_text())
            self.assertEqual(events[-1]['status'],'submission_failed')
            self.assertEqual(events[0]['prepare_job'],'1001')

    def test_active_job_blocks_duplicate_and_scratch_rejects_inherited_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _, _ = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'INIT_ARTIFACT to be unset'):
                jobs.new_job('scratch',environment=env)
            calls,run=self.scheduler()
            with patch.object(jobs.subprocess,'run',side_effect=run):
                spec=jobs.new_job('continue',environment=env)
            with patch.object(jobs.subprocess,'run',return_value=SimpleNamespace(
                    stdout='inezita_train_'+spec['run_id']+'\n')) as submit:
                with self.assertRaisesRegex(ValueError,'already has a queued'):
                    jobs.dispatch(Path(env['RUN_DIR'])/'job.json',environment=env)
            self.assertEqual(submit.call_count,1)
            with self.assertRaisesRegex(ValueError,'RUN_DIR already exists'):
                jobs.new_job('continue',environment=env)


if __name__=='__main__':
    unittest.main()
