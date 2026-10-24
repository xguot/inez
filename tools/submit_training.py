#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Submit or resume one explicit, frozen Inezita training experiment."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from inez.curriculum import atomic_json, sha256
from inez.experiment import create_snapshot, verify_accepted_artifact, verify_snapshot
from inez.jobs import verify_job_spec
from train_inezita import exclusive_run, read_config, verify_prepared_data
import chess
import torch


def absolute(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def new_job(mode='scratch', *, prepare=False, dry_run=False, environment=None):
    env = dict(os.environ if environment is None else environment)
    config_path = absolute(env.get('TRAIN_CONFIG') or 'configs/train/inezita.json')
    config, model_config = read_config(config_path)
    artifact = env.get('INIT_ARTIFACT')
    if mode == 'scratch' and artifact:
        raise ValueError('scratch training requires INIT_ARTIFACT to be unset')
    if mode == 'continue' and not artifact:
        raise ValueError('continuation requires INIT_ARTIFACT from a previous accepted run')
    if mode == 'scratch' and config.get('requires_init_artifact'):
        raise ValueError('this config requires continuation; use configs/train/inezita.json for scratch training')
    initial = verify_accepted_artifact(absolute(artifact), expected_config=model_config) if artifact else None
    run = absolute(env.get('RUN_DIR') or 'results/training/inezita-0.1.3')
    data = absolute(env.get('DATA_DIR') or 'data/processed/inezita-0.1.3')
    if run.exists():
        # Clean up an empty or incomplete run directory that never wrote job.json
        if not (run / 'job.json').exists() and not (run / 'request.json').exists():
            shutil.rmtree(run)
        else:
            raise ValueError(f'RUN_DIR already exists: {run}; use resume for its frozen experiment or choose a fresh directory')
    if run == data or run.is_relative_to(data) or data.is_relative_to(run):
        raise ValueError('data and run directories must be separate')
    if prepare:
        data_identity = None
    else:
        data_identity = verify_prepared_data(data, config)
    plan = {'mode': mode, 'config': str(config_path), 'data_dir': str(data),
            'run_dir': str(run), 'initial_artifact': str(initial) if initial else None,
            'prepare_data': prepare, 'data_identity': data_identity}
    if dry_run:
        return plan
    run.mkdir(parents=True, exist_ok=False)
    source = create_snapshot(ROOT, run / 'source')
    # Freeze the actual config, including explicit locations for shared raw
    # archives and snapshot-local model/tactical/opening resources.
    config = json.loads(json.dumps(config))
    resource_files = {}
    for key in ('model_config', 'tactics'):
        original = absolute(config[key])
        selected = source / original.relative_to(ROOT) if original.is_relative_to(ROOT) else original
        config[key] = str(selected)
        resource_files[str(selected)] = sha256(selected)
    if config['preparation'].get('openings'):
        original = absolute(config['preparation']['openings'])
        selected = source / original.relative_to(ROOT) if original.is_relative_to(ROOT) else original
        config['preparation']['openings'] = str(selected)
        resource_files[str(selected)] = sha256(selected)
    settings = config['preparation']
    for key in ('archive',):
        if settings.get(key):
            settings[key] = str(absolute(settings[key]))
    if settings.get('archives'):
        settings['archives'] = [str(absolute(value)) for value in settings['archives']]
    frozen_config = run / 'config.json'
    atomic_json(frozen_config, config)
    resource_files[str(frozen_config)] = sha256(frozen_config)
    spec = {'job_version': 1, **plan, 'config': str(frozen_config), 'source_root': str(source),
            'original_root': str(ROOT), 'source_manifest_sha256': sha256(source / '.inezita-source.json'),
            'input_files': resource_files,
            'initial_weights_sha256': sha256(initial / 'weights.pt') if initial else None,
            'initial_manifest_sha256': sha256(initial / 'manifest.json') if initial else None,
            'python': str(Path(sys.executable).absolute()),
            'run_id': hashlib.sha256(str(run).encode()).hexdigest()[:12],
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'chess': chess.__version__},
            'created_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(run / 'job.json', spec)
    return dispatch(run / 'job.json', environment=env, prepare=prepare)


def dispatch(path, *, environment=None, prepare=None):
    spec = verify_job_spec(path)
    with exclusive_run(spec['run_dir'], '.submit.lock'):
        return _dispatch(path, environment=environment, prepare=prepare)


def _dispatch(path, *, environment=None, prepare=None):
    spec = verify_job_spec(path)
    source = Path(spec['source_root'])
    config, _ = read_config(spec['config'])
    should_prepare = spec['prepare_data'] if prepare is None else prepare
    if not should_prepare:
        actual = verify_prepared_data(spec['data_dir'], config)
        if spec['data_identity'] is not None and actual != spec['data_identity']:
            raise ValueError('prepared data changed since job submission')
    env = dict(os.environ if environment is None else environment)
    # Explicit pinned values replace inherited settings from unrelated runs.
    env.update(INEZ_ROOT=str(source), INEZ_DATA_ROOT=spec['original_root'],
               INEZ_PYTHON=spec['python'], TRAIN_CONFIG=spec['config'],
               RUN_DIR=spec['run_dir'], DATA_DIR=spec['data_dir'],
               INEZ_JOB_SPEC=str(Path(path).resolve()))
    if spec['initial_artifact']:
        env['INIT_ARTIFACT'] = spec['initial_artifact']
    else:
        env.pop('INIT_ARTIFACT', None)
    if env.get('RESUME_MODE', 'exact') not in ('exact', 'portable'):
        raise ValueError('RESUME_MODE must be exact or portable')
    names = {kind: 'inezita_' + kind + '_' + spec['run_id'] for kind in ('prepare', 'train')}
    queued = subprocess.run(['squeue', '--me', '--noheader', '--format=%j'],
        env=env, capture_output=True, text=True, check=True)
    if set(queued.stdout.split()) & set(names.values()):
        raise ValueError('this run already has a queued/running job; wait for it to finish before resuming')
    account = env.get('INEZ_ACCOUNT', 'nssac_students')
    event = {'submitted_utc': datetime.now(timezone.utc).isoformat(),
             'prepare_job': None, 'train_job': None, 'status': 'submitting'}
    def submit(script, *extra):
        command = ['sbatch', '--parsable', '--export=ALL', '--chdir=' + str(source),
                   '--account=' + account, '--job-name=' + names['prepare' if script.startswith('prepare') else 'train'],
                   *extra, str(source / 'hpc' / script)]
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        value = result.stdout.strip().split(';')[0]
        if not re.fullmatch(r'[0-9]+', value):
            raise ValueError(f'sbatch did not return a numeric job ID: {result.stdout!r}')
        return value
    history = Path(spec['run_dir']) / 'submissions.json'
    events = json.loads(history.read_text()) if history.exists() else []
    events.append(event)
    atomic_json(history, events)
    try:
        dependency = []
        if should_prepare:
            event['prepare_job'] = submit('prepare_inezita.slurm')
            atomic_json(history, events)
            dependency = ['--dependency=afterok:' + event['prepare_job'], '--kill-on-invalid-dep=yes']
        event['train_job'] = submit('train_inezita.slurm', *dependency)
        event['status'] = 'submitted'
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        event['status'] = 'submission_failed'
        event['error'] = str(error)
        raise
    finally:
        atomic_json(history, events)
    return {**spec, 'submission': event}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='auto', choices=('auto', 'continue', 'scratch', 'resume'))
    parser.add_argument('--prepare', action='store_true', help='prepare/verify data in a CPU job before training')
    parser.add_argument('--dry-run', action='store_true', help='validate inputs and print the plan without creating a run or submitting jobs')
    args = parser.parse_args(argv)
    try:
        mode = args.mode
        prepare = args.prepare
        env = dict(os.environ)
        if mode == 'auto':
            run_dir = absolute(env.get('RUN_DIR') or 'results/training/inezita-0.1.3')
            if (run_dir / 'job.json').is_file():
                mode = 'resume'
            else:
                mode = 'scratch'
                prepare = True
                env.pop('INIT_ARTIFACT', None)
        if mode == 'resume':
            run_dir = absolute(env.get('RUN_DIR') or 'results/training/inezita-0.1.3')
            path = run_dir / 'job.json'
            if args.dry_run:
                result = verify_job_spec(path)
            else:
                result = dispatch(path, environment=env, prepare=True if prepare else None)
        else:
            result = new_job(mode, prepare=prepare, dry_run=args.dry_run, environment=env)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Inezita submission failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
