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
        config = json.loads((ROOT / 'configs/train/inezita.json').read_text())
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
                        ('min_rating','min_ply','samples_per_game','max_games','blunder_samples_per_game',
                         'min_time_control_seconds','require_normal_termination')}}
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
            rec.update(label_kind='teacher_wdl', teacher={'name':'Stockfish','version':'fixture','limits':{'nodes':config['preparation']['teacher_nodes']}})
            path = teacher / (split + '.jsonl')
            path.write_text(json.dumps(rec) + '\n')
            tm['splits'][split] = {'records': 1, 'sha256': sha256(path)}
        tm['teacher'] = {'human_manifest_sha256': sha256(human / 'manifest.json'),
            'nodes':config['preparation']['teacher_nodes'],'multipv':4,'temperature':0.15,'label_contract':2,'binary_sha256':'fixture'}
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
            env['DATA_DIR'] = str(Path(tmp) / 'fresh-data')
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

    def test_auto_mode_submits_full_scratch_pipeline_and_resumes_existing_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _, _ = self.fixture(Path(tmp))
            scratch_cfg = json.loads((ROOT / 'configs/train/inezita.json').read_text())
            scratch_cfg['model_config'] = json.loads(Path(env['TRAIN_CONFIG']).read_text())['model_config']
            scratch_path = Path(tmp) / 'scratch.json'
            atomic_json(scratch_path, scratch_cfg)
            env['TRAIN_CONFIG'] = str(scratch_path)
            # An empty RUN_DIR is usable; auto mode starts scratch without inheriting old weights.
            Path(env['RUN_DIR']).mkdir(parents=True)
            calls, run = self.scheduler()
            with patch.dict(os.environ, env, clear=False), patch.object(jobs.subprocess, 'run', side_effect=run):
                self.assertEqual(jobs.main([]), 0)
                spec = verify_job_spec(Path(env['RUN_DIR']) / 'job.json')
                self.assertEqual(spec['mode'], 'scratch')
                self.assertTrue(spec['prepare_data'])
                self.assertIsNone(spec['initial_artifact'])
            # Re-running with no arguments automatically resumes the existing job.json.
            calls.clear()
            with patch.dict(os.environ, env, clear=False), patch.object(jobs.subprocess, 'run', side_effect=run):
                self.assertEqual(jobs.main([]), 0)
            submitted = [command for command, _ in calls if command[0] == 'sbatch']
            self.assertEqual(len(submitted), 1)  # Verified ready data needs only the GPU job.

    def test_nonempty_incomplete_and_overlapping_run_are_never_deleted(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run), tempfile.TemporaryDirectory() as tmp:
                env, _, _ = self.fixture(Path(tmp))
                run = Path(env['RUN_DIR'])
                run.mkdir()
                sentinel = run / 'valuable-checkpoint.bin'
                sentinel.write_bytes(b'keep-this')
                with patch.object(jobs.subprocess, 'run') as submit:
                    with self.assertRaisesRegex(ValueError, 'not empty'):
                        jobs.new_job('continue', dry_run=dry_run, environment=env)
                self.assertEqual(sentinel.read_bytes(), b'keep-this')
                submit.assert_not_called()
                env['RUN_DIR'] = env['DATA_DIR']
                before = Path(env['DATA_DIR'], 'human/train.jsonl').read_bytes()
                with self.assertRaisesRegex(ValueError, 'must be separate'):
                    jobs.new_job('continue', dry_run=dry_run, environment=env)
                self.assertEqual(Path(env['DATA_DIR'], 'human/train.jsonl').read_bytes(), before)

    def test_completed_experiment_never_submits_another_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, recipe, _ = self.fixture(Path(tmp))
            calls,run = self.scheduler()
            with patch.object(jobs.subprocess, 'run', side_effect=run):
                jobs.new_job('continue', environment=env)
            run_dir=Path(env['RUN_DIR'])
            atomic_json(run_dir/'request.json', {'config':json.loads(recipe.read_text()),
                'initial_artifact':env['INIT_ARTIFACT']})
            atomic_json(run_dir/'curriculum.json', {'completed':['teacher',
                'selfplay-001','selfplay-002','selfplay-003','selfplay-004']})
            with patch.object(jobs.subprocess, 'run') as submit:
                result=jobs.dispatch(run_dir/'job.json', environment=env)
            self.assertEqual(result['status'],'complete')
            self.assertIsNone(result['submission'])
            submit.assert_not_called()

    def test_frozen_preflight_executes_from_snapshot_and_accepts_portable_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _, _ = self.fixture(Path(tmp))
            calls, run = self.scheduler()
            with patch.object(jobs.subprocess, 'run', side_effect=run):
                spec = jobs.new_job('continue', environment=env)
            cmd=[sys.executable, str(Path(spec['source_root'])/'tools/train_inezita.py'),
                'preflight','--config',spec['config'],'--job-spec',str(Path(env['RUN_DIR'])/'job.json'),
                '--device','cpu','--threads','1']
            proc=subprocess.run(cmd, cwd=spec['source_root'],capture_output=True,text=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=30)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            self.assertIn('"parameters":',proc.stdout)
            job_path=Path(env['RUN_DIR'])/'job.json'
            changed=json.loads(job_path.read_text())
            changed['runtime']['torch']='different-build'
            atomic_json(job_path,changed)
            proc=subprocess.run(cmd+['--resume-mode','portable'],cwd=spec['source_root'],
                capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=30)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)


    def test_time_limits_are_explicit_ceiling_overrides_on_submissions_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _, _ = self.fixture(Path(tmp))
            env.update(DATA_DIR=str(Path(tmp)/'fresh-data'),
                       PREP_TIME_LIMIT='08:00:00', TRAIN_TIME_LIMIT='1-00:00:00')
            calls,run=self.scheduler()
            with patch.object(jobs.subprocess, 'run', side_effect=run):
                spec=jobs.new_job('continue',prepare=True,environment=env)
            submitted=[command for command,_ in calls if command[0]=='sbatch']
            self.assertIn('--time=08:00:00',submitted[0])
            self.assertIn('--time=1-00:00:00',submitted[1])
            self.assertEqual(spec['submission']['time_limits'],
                             {'prepare':'08:00:00','train':'1-00:00:00'})
            calls.clear()
            with patch.object(jobs.subprocess, 'run', side_effect=run):
                resumed=jobs.dispatch(Path(env['RUN_DIR'])/'job.json',
                    environment={'TRAIN_TIME_LIMIT':'12:00:00'})
            self.assertEqual(resumed['time_limits'],{'prepare':'08:00:00','train':'12:00:00'})
            self.assertIn('--time=12:00:00',[cmd for cmd,_ in calls if cmd[0]=='sbatch'][-1])

    def test_invalid_time_limits_fail_before_creating_or_submitting_a_run(self):
        for value in ('0:00:00','unlimited','1:99:00','-1:00:00','1-24:00:00','4h'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                env,_,_=self.fixture(Path(tmp))
                env['PREP_TIME_LIMIT']=value
                with patch.object(jobs.subprocess, 'run') as submit:
                    with self.assertRaises(ValueError):
                        jobs.new_job('continue',environment=env)
                submit.assert_not_called()
                self.assertFalse(Path(env['RUN_DIR']).exists())



if __name__=='__main__':
    unittest.main()
