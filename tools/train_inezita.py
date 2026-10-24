#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare Elite/Stockfish data and run gated 29M Inezita learning stages."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import fcntl
import os
import platform
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

import chess
import torch

from inez.curriculum import (atomic_json, evaluate_candidate, import_games,
    position_key, promotion_decision, read_tactics, relabel_dataset, selfplay_records, sha256, verify_dataset)
from inez.model import ChessTransformer, ModelConfig, load_checkpoint, save_checkpoint
from inez.device import validate_cuda_runtime
from inez.experiment import training_sources, verify_accepted_artifact, verify_snapshot, SNAPSHOT_MANIFEST
from inez.jobs import verify_job_spec
from inez.records import load_records, validate_record, read_json
from inez.trainer import StopRequested, dataset_records, restore_checkpoint_rng, train_stage, training_signature


def rooted(path):
    path = Path(path)
    if not path.is_absolute() and path.parts[:2] == ('data', 'raw'):
        return Path(os.environ.get('INEZ_DATA_ROOT', ROOT)) / path
    return path if path.is_absolute() else ROOT / path


@contextmanager
def exclusive_run(directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'another job is using {directory}; use a different directory or wait for it to finish') from None
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()) + '\n')
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_config(path):
    config = json.loads(Path(path).read_text())
    preparation = config['preparation']
    for key, minimum in (('max_games', 1), ('samples_per_game', 1), ('min_ply', 0),
                         ('min_rating', 0), ('blunder_samples_per_game', 0),
                         ('min_time_control_seconds', 0),
                         ('teacher_nodes', 1), ('teacher_multipv', 1)):
        default = 0 if key in ('blunder_samples_per_game', 'min_time_control_seconds') else None
        value = preparation.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f'preparation {key} must be an integer >= {minimum}')
    if not isinstance(preparation.get('require_normal_termination', False), bool):
        raise ValueError('preparation require_normal_termination must be a boolean')
    temperature = preparation.get('teacher_temperature')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('preparation teacher_temperature must be finite and positive')
    archives, urls = preparation.get('archives'), preparation.get('urls')
    if archives is not None or urls is not None:
        if (not isinstance(archives, list) or not isinstance(urls, list) or not archives
                or len(archives) != len(urls)
                or any(not isinstance(item, str) or not item.strip() for item in archives + urls)):
            raise ValueError('preparation archives and urls must be nonempty aligned lists')
    elif not preparation.get('archive') or not preparation.get('url'):
        raise ValueError('preparation requires an archive and URL')
    if preparation.get('openings') and not rooted(preparation['openings']).is_file():
        raise ValueError('configured preparation openings file does not exist')
    # Reject invalid settings before any lengthy allocation or download.
    model_config = ModelConfig.from_dict(json.loads(rooted(config['model_config']).read_text()))
    read_tactics(rooted(config['tactics']))
    if not config['gate']['openings']:
        raise ValueError('paired gate requires opening positions')
    unique_openings = {tuple(moves) for moves in config['gate']['openings']}
    if (len(unique_openings) != len(config['gate']['openings'])
            or not 1 <= config['gate']['pairs'] <= len(unique_openings)):
        raise ValueError('match gate needs distinct paired openings; do not count repeated deterministic games as new evidence')
    positions = set()
    for moves in config['gate']['openings']:
        board = chess.Board()
        for move in moves:
            board.push_uci(move)
        key = position_key(board)
        if key in positions or board.outcome(claim_draw=False) is not None:
            raise ValueError('match gate needs distinct nonterminal opening positions')
        positions.add(key)
    settings = config['training']
    for key in ('human_steps', 'teacher_steps', 'selfplay_steps', 'batch_size', 'checkpoint_every'):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'training {key} must be a positive integer')
    for key in ('lr', 'teacher_lr', 'selfplay_lr'):
        value = settings.get(key, settings['teacher_lr'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'training {key} must be finite and positive')
    if settings.get('validation_limit') is not None and settings.get('validation_limit', 2048) < 1:
        raise ValueError('validation_limit must be positive or null for the full split')
    promotion_budget(config)
    if not 0 < config['selfplay']['teacher_replay_fraction'] < 1:
        raise ValueError('self-play requires both search targets and teacher replay')
    if config['selfplay']['rounds'] < 1 or config['gate']['historical_opponents'] < 1:
        raise ValueError('at least one self-play round and historical opponent required')
    if config['training'].get('selfplay_lr', config['training']['teacher_lr']) <= 0:
        raise ValueError('self-play learning rate must be positive')
    return config, model_config


def _validate_archive(path):
    path = Path(path)
    if path.name.endswith('.zst') or path.name.endswith('.zst.part'):
        try:
            with path.open('rb') as raw:
                magic = raw.read(4)
            magic_int = int.from_bytes(magic, 'little') if len(magic) == 4 else 0
            if magic != b'\x28\xb5\x2f\xfd' and not (0x184D2A50 <= magic_int <= 0x184D2A5F):
                raise ValueError(f'corrupt zstd archive: {path}')
        except OSError as error:
            raise ValueError(f'corrupt archive: {path}') from error
        return
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith('.pgn')]
            if len(names) != 1 or archive.testzip() is not None:
                raise ValueError('archive must contain exactly one intact PGN')
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError(f'corrupt archive: {path}') from error


def download_archive(path, url):
    path = Path(path)
    provenance_path = path.with_suffix('.source.json')
    if path.exists():
        if not path.is_file():
            raise ValueError(f'archive path is not a file: {path}')
        _validate_archive(path)
        digest = sha256(path)
        if provenance_path.exists():
            provenance = read_json(provenance_path.read_text())
            if provenance.get('uri') != url or provenance.get('sha256') != digest:
                raise ValueError(f'cached archive provenance or checksum mismatch: {path}')
        else:
            atomic_json(provenance_path, {'uri': url, 'sha256': digest,
                        'checksum_kind': 'observed existing archive hash; publisher supplies no checksum here'})
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'InezitaTraining/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 2 * 1024**3:
                    raise ValueError('Elite archive exceeds the 2 GiB download bound')
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_archive(temporary)
        os.replace(temporary, path)
        atomic_json(provenance_path, {'uri': url, 'sha256': sha256(path),
                    'checksum_kind': 'observed download hash; publisher supplies no checksum here'})
    finally:
        temporary.unlink(missing_ok=True)


def resolve_stockfish(value):
    """Use a provided engine or cache the pinned official release in data/raw."""
    executable = shutil.which(value) if value != 'auto' else None
    if executable:
        return executable
    if value not in ('auto', 'stockfish'):
        raise ValueError(f'Stockfish executable not found: {value}')
    system, machine = platform.system(), platform.machine().lower()
    if system == 'Darwin' and machine in ('arm64', 'aarch64'):
        asset = 'stockfish-macos-m1-apple-silicon'
    elif system == 'Linux' and machine in ('x86_64', 'amd64'):
        flags = Path('/proc/cpuinfo').read_text().lower()
        asset = 'stockfish-ubuntu-x86-64-avx2' if 'avx2' in flags else 'stockfish-ubuntu-x86-64'
    else:
        raise ValueError('Set INEZ_STOCKFISH to a compatible installed Stockfish executable on this platform')
    destination = Path(os.environ.get('INEZ_DATA_ROOT', ROOT)) / 'data/raw/stockfish-17.1'
    destination.mkdir(parents=True, exist_ok=True)
    binary = destination / asset
    provenance = destination / (asset + '.source.json')
    if binary.exists() and provenance.exists():
        if sha256(binary) != json.loads(provenance.read_text())['binary_sha256']:
            raise ValueError('cached Stockfish binary checksum changed')
        return str(binary)
    url = 'https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/' + asset + '.tar'
    archive = destination / (asset + '.tar')
    temporary = archive.with_suffix('.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'InezitaTraining/1.0'})
    if not archive.exists():
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        os.replace(temporary, archive)
    # Read a named member, never extract paths or symlinks from a remote tar.
    with tarfile.open(archive) as bundle:
        matches = [m for m in bundle.getmembers() if m.isfile() and Path(m.name).name == asset]
        if len(matches) != 1:
            raise ValueError('official Stockfish bundle does not contain the expected binary')
        with bundle.extractfile(matches[0]) as stream:
            binary.with_suffix('.tmp').write_bytes(stream.read())
        os.chmod(binary.with_suffix('.tmp'), 0o755)
        os.replace(binary.with_suffix('.tmp'), binary)
    try:
        probe = subprocess.run([str(binary)], input='uci\nquit\n', capture_output=True,
                               text=True, timeout=30)
        portable = probe.returncode == 0 and 'uciok' in probe.stdout
    except (OSError, subprocess.TimeoutExpired):
        portable = False
    build = 'official prebuilt binary'
    if not portable:
        if system != 'Linux' or shutil.which('make') is None:
            raise ValueError('official Stockfish binary cannot run here; set INEZ_STOCKFISH to a locally built engine')
        # Older cluster libc versions may not run Ubuntu release executables.
        # The official bundle contains the same release's source; build it
        # against this node's libraries instead of repeating a loader failure.
        source = destination / 'native-build'
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.startswith(('stockfish/src/', 'stockfish/scripts/')):
                    continue
                relative = Path(member.name).relative_to('stockfish')
                target = (source / relative).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise ValueError('unsafe source path in Stockfish release')
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as stream:
                    target.write_bytes(stream.read())
        arch = 'x86-64-avx2' if 'avx2' in Path('/proc/cpuinfo').read_text().lower() else 'x86-64'
        subprocess.run(['make', '-C', str(source / 'src'), '-j',
                        str(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))),
                        'build', 'ARCH=' + arch, 'COMP=gcc', 'CXX=' + os.environ.get('CXX', 'g++'),
                        'EXE=' + str(binary.resolve())], check=True)
        build = 'rebuilt release source for local libc'
    atomic_json(provenance, {'release': 'sf_17.1', 'uri': url, 'archive_sha256': sha256(archive),
        'binary_sha256': sha256(binary), 'checksum_kind': 'observed download hashes',
        'source': 'https://github.com/official-stockfish/Stockfish/tree/sf_17.1', 'license': 'GPL-3.0', 'build': build})
    return str(binary)


def _verify_preparation_manifest(manifest, config):
    preparation = manifest.get('preparation')
    if not isinstance(preparation, dict):
        raise ValueError('incomplete human preparation metadata; choose a fresh DATA_DIR')
    settings = config['preparation']
    for key in ('min_rating', 'min_ply', 'samples_per_game', 'max_games',
                'blunder_samples_per_game', 'min_time_control_seconds', 'require_normal_termination'):
        default = False if key == 'require_normal_termination' else (
            0 if key in ('blunder_samples_per_game', 'min_time_control_seconds') else None)
        if preparation.get(key, default) != settings.get(key, default):
            raise ValueError(f'prepared data {key} differs from the requested configuration; choose a fresh DATA_DIR')
    if preparation.get('tactics_sha256') != sha256(rooted(config['tactics'])):
        raise ValueError('prepared data uses a different held-out tactics suite')
    openings = rooted(settings['openings']) if settings.get('openings') else None
    if openings is not None and not openings.is_file():
        raise ValueError('configured preparation openings file does not exist')
    if preparation.get('openings_sha256') != (sha256(openings) if openings else None):
        raise ValueError('prepared data uses a different opening corpus')
    audit = manifest.get('audit', {})
    if audit.get('cross_split_positions') != 0 or audit.get('duplicate_samples') != 0:
        raise ValueError('prepared data does not declare independent, deduplicated splits')


def verify_prepared_data(directory, config, *, full=False):
    """Verify completed corpus identity and label budgets; training seed may change.

    full=True additionally validates every legal game/target. The default checks
    all split hashes and streamed record counts against their completed manifests.
    """
    directory = Path(directory)
    human = verify_dataset(directory / 'human', full=full)
    _verify_preparation_manifest(human, config)
    teacher = verify_dataset(directory / 'teacher', full=full)
    human_hash = sha256(directory / 'human/manifest.json')
    label = teacher.get('teacher', {})
    if label.get('human_manifest_sha256') != human_hash:
        raise ValueError('teacher and human splits have different provenance')
    settings = config['preparation']
    for key in ('nodes', 'multipv', 'temperature'):
        if label.get(key) != settings['teacher_' + key]:
            raise ValueError(f'prepared teacher {key} differs from the requested label budget; choose a fresh DATA_DIR')
    if settings.get('teacher_label_contract') is not None and label.get('label_contract') != settings['teacher_label_contract']:
        raise ValueError('teacher label contract differs from the requested configuration; choose a fresh DATA_DIR')
    for split in ('train', 'validation', 'evaluation'):
        if human['splits'][split]['records'] != teacher['splits'][split]['records']:
            raise ValueError('teacher and human split record counts differ')
    request_path = directory / 'request.json'
    if request_path.is_file():
        request = read_json(request_path.read_text())
        # Data preparation seed stays recorded in request/manifest; a new training
        # experiment may deliberately use a different optimizer/sampling seed.
        recorded = request.get('preparation', {})
        for key in ('min_rating', 'min_ply', 'samples_per_game', 'max_games',
                    'blunder_samples_per_game', 'min_time_control_seconds', 'require_normal_termination'):
            default = False if key == 'require_normal_termination' else (
                0 if key in ('blunder_samples_per_game', 'min_time_control_seconds') else None)
            if recorded.get(key, default) != settings.get(key, default):
                raise ValueError(f'prepared data request {key} differs from the requested preparation')
        if request.get('tactics_sha256') != human['preparation']['tactics_sha256']:
            raise ValueError('prepared data request uses a different tactics suite')
        if request.get('openings_sha256') != human['preparation'].get('openings_sha256'):
            raise ValueError('prepared data request uses a different opening corpus')
        if request.get('teacher_binary_sha256') != label.get('binary_sha256'):
            raise ValueError('prepared teacher binary differs from its preparation request')
    return {'human_manifest_sha256': human_hash,
            'teacher_manifest_sha256': sha256(directory / 'teacher/manifest.json'),
            'splits': {split: teacher['splits'][split] for split in ('train', 'validation', 'evaluation')}}


def prepare(args, config):
    settings = config['preparation']
    if type(args.threads) is not int or args.threads < 1:
        raise ValueError('preparation threads must be a positive integer')
    if args.archive:
        archives = [Path(args.archive)]
        urls = [settings.get('url') or settings['urls'][0]]
    elif settings.get('archives'):
        archives = [rooted(item) for item in settings['archives']]
        urls = list(settings['urls'])
    else:
        archives = [rooted(settings['archive'])]
        urls = [settings['url']]
    if len(archives) != len(urls) or not archives:
        raise ValueError('preparation archives and urls must be nonempty aligned lists')
    openings_path = rooted(settings['openings']) if settings.get('openings') else None
    if openings_path is not None and not openings_path.is_file():
        raise ValueError('configured preparation openings file does not exist')
    output = Path(args.data_dir)
    # Validate already published data before downloading or scheduling more work.
    if (output / 'human').exists():
        _verify_preparation_manifest(verify_dataset(output / 'human'), config)
    if (output / 'teacher').exists():
        identity = verify_prepared_data(output, config)
        print(json.dumps({'data_ready': str(output.resolve()), **identity}), flush=True)
        return
    executable = resolve_stockfish(args.stockfish)
    for archive_path, archive_url in zip(archives, urls):
        download_archive(archive_path, archive_url)
    output.mkdir(parents=True, exist_ok=True)
    request = {'preparation': settings, 'seed': config['seed'],
               'archive_sha256': [sha256(a) for a in archives] if len(archives) > 1 else sha256(archives[0]),
               'openings_sha256': sha256(openings_path) if openings_path else None,
               'teacher_binary_sha256': sha256(executable), 'teacher_threads': args.threads,
               'tactics_sha256': sha256(rooted(config['tactics']))}
    request_path = output / 'request.json'
    if request_path.exists() and read_json(request_path.read_text()) != request:
        raise ValueError('preparation request changed; choose a new data directory')
    atomic_json(request_path, request)
    if not (output / 'human').exists():
        manifest = import_games(archives, output / 'human', uri=urls,
            license_name=settings['license'], tactics=rooted(config['tactics']),
            max_games=settings['max_games'], samples_per_game=settings['samples_per_game'],
            min_ply=settings['min_ply'], min_rating=settings['min_rating'], seed=config['seed'],
            openings_path=openings_path,
            blunder_samples_per_game=settings.get('blunder_samples_per_game', 0),
            min_time_control_seconds=settings.get('min_time_control_seconds', 0),
            require_normal_termination=settings.get('require_normal_termination', False))
        print(json.dumps({'human_prepared': manifest['preparation']}, indent=2), flush=True)
    relabel_dataset(output / 'human', output / 'teacher', executable,
        nodes=settings['teacher_nodes'], multipv=settings['teacher_multipv'],
        temperature=settings['teacher_temperature'], threads=args.threads)
    identity = verify_prepared_data(output, config)
    print(json.dumps({'data_ready': str(output.resolve()), **identity}), flush=True)


def gate_summary(result, report_path, *, minimum_score, cached=False):
    """Explain existing gate reports too, without replaying matches or changing acceptance."""
    opponents = []
    for report in result['opponents']:
        games = report.get('games', [])
        scores = [game['candidate_score'] for game in games if game['candidate_score'] is not None]
        completed = report.get('completed_games', len(scores))
        score, lower = report.get('score'), report.get('score_lower_bound')
        regressions = report.get('tactical_regressions', [])
        reasons = list(report.get('rejection_reasons', []))
        if not report['passed'] and not reasons:
            if completed == 0:
                reasons.append('no_completed_games')
            if completed < len(games):
                reasons.append('unfinished_match_games')
            if score is not None and score < minimum_score:
                reasons.append('score_below_minimum')
            if lower is not None and lower <= 0.5:
                reasons.append('insufficient_match_evidence')
            if regressions:
                reasons.append('tactical_regression')
            if not reasons:
                reasons.append('gate_rejected; inspect the full report')
        opponents.append({'opponent': report['opponent'], 'passed': report['passed'],
            'score': score, 'minimum_score': minimum_score, 'score_lower_bound': lower,
            'required_lower_bound_exclusive': 0.5 if lower is not None else None,
            'paired_test': report.get('paired_test'), 'completed_games': completed,
            'total_games': len(games), 'wins': scores.count(1.0), 'draws': scores.count(0.5),
            'losses': scores.count(0.0), 'unfinished_games': len(games) - completed,
            'tactical_regressions': regressions, 'rejection_reasons': reasons})
    return {'gate': str(report_path), 'promoted': result['passed'], 'cached': cached,
            'opponents': opponents}


def run_status(directory):
    """Inspect durable run/gate metadata without loading weights or requiring CUDA."""
    root = Path(directory)
    request_path = root / 'request.json'
    if not request_path.exists():
        raise ValueError(f'no training run request found at {request_path}')
    request = json.loads(request_path.read_text())
    config = request['config']
    expected = ([] if request['initial_artifact'] else ['human']) + ['teacher']
    expected += [f'selfplay-{number + 1:03d}' for number in range(config['selfplay']['rounds'])]
    state_path = root / 'curriculum.json'
    completed = json.loads(state_path.read_text())['completed'] if state_path.exists() else []
    pending = [name for name in expected if name not in completed]
    accepted_path = root / 'accepted.json'
    accepted = json.loads(accepted_path.read_text()) if accepted_path.exists() else None
    gates = [gate_summary(json.loads(path.read_text()), path,
                         minimum_score=config['gate']['minimum_score'], cached=True)
             for path in sorted(root.glob('*/gate-*.json'))]
    return {'status': 'in_progress' if pending else 'complete',
            'curriculum_completed': completed, 'pending_stages': pending,
            'accepted': accepted, 'gates': gates,
            'next_action': ('Resume this RUN_DIR to finish the pending stages.' if pending else
                'Training finished. Reusing this RUN_DIR performs no further optimizer steps. '
                'For more training, choose a new RUN_DIR and set INIT_ARTIFACT to accepted.artifact.'),
            'note': 'Gate rejection keeps the earlier accepted model and is a normal training outcome.'}


def promotion_budget(config, *, warm_start=True):
    """Bonferroni allocation across the predeclared candidate checkpoints.

    Beating every historical opponent is an intersection requirement, so no
    extra opponent factor is needed. A new run is a new experiment, not another
    chance to keep testing an unchanged candidate until it passes.
    """
    settings = config.get('training', {})
    interval = settings.get('checkpoint_every', 1)
    stages = [settings.get('teacher_steps', 1)]
    if not warm_start:
        stages.append(settings.get('human_steps', 1))
    stages += [settings.get('selfplay_steps', 1)] * config.get('selfplay', {}).get('rounds', 1)
    candidates = sum(math.ceil(steps / interval) for steps in stages)
    familywise_alpha = config['gate'].get('familywise_alpha', 0.05)
    if not math.isfinite(familywise_alpha) or not 0 < familywise_alpha < 0.5:
        raise ValueError('gate familywise_alpha must be between zero and 0.5')
    return {'method': 'bonferroni-candidates-v1', 'planned_candidates': candidates,
            'familywise_alpha': familywise_alpha, 'candidate_alpha': familywise_alpha / candidates}


class ChampionGate:
    def __init__(self, root, config, model_config, device, stop, *, warm_start=True):
        self.root, self.config, self.model_config = Path(root), config, model_config
        self.device, self.stop = device, stop
        self.path = self.root / 'accepted.json'
        self.budget = promotion_budget(config, warm_start=warm_start)
        self.tactics = read_tactics(rooted(config['tactics']))

    def _evidence_identity(self, current, opponents):
        artifacts = {}
        for name in [current, *opponents]:
            path = Path(name)
            manifest = json.loads((path / 'manifest.json').read_text())
            actual = sha256(path / 'weights.pt')
            if actual != manifest['weights_sha256']:
                raise ValueError('promotion artifact weights changed or do not match their manifest')
            artifacts[name] = {'weights_sha256': actual,
                               'manifest_sha256': sha256(path / 'manifest.json')}
        sources = {name: sha256(ROOT / name) for name in (
            'tools/train_inezita.py', 'training/inez/curriculum.py',
            'training/inez/puct.py', 'training/inez/encoding.py', 'training/inez/model.py')}
        return {'artifacts': artifacts, 'sources': sources}

    def _validate_promotion_report(self, result, current, history):
        """A saved pass is authority only for this candidate and experiment."""
        expected_opponents = list(reversed(history[-self.config['gate']['historical_opponents']:]))
        if (result.get('candidate') != current or result.get('parent_history') != history
                or result.get('promotion_budget') != self.budget
                or result.get('evidence_identity') != self._evidence_identity(current, expected_opponents)
                or result.get('tactics_sha256') != sha256(rooted(self.config['tactics']))
                or not expected_opponents
                or [report.get('opponent') for report in result.get('opponents', [])] != expected_opponents):
            raise ValueError('promotion report does not match this candidate, champion history, or paired experiment; use a fresh run directory')
        expected_tactics = [item['name'] for item in self.tactics]
        expected_openings = []
        for moves in self.config['gate']['openings'][:self.config['gate']['pairs']]:
            board = chess.Board()
            for move in moves:
                board.push_uci(move)
            expected_openings.append((list(moves), position_key(board)))
        if len(expected_openings) != self.config['gate']['pairs']:
            raise ValueError('promotion report has insufficient configured opening pairs')
        for report in result['opponents']:
            candidate_tactics = report.get('candidate_tactics', [])
            incumbent_tactics = report.get('incumbent_tactics', [])
            if [item.get('name') for item in candidate_tactics] != expected_tactics:
                raise ValueError('promotion report has missing or different tactical evidence')
            games = report.get('games', [])
            for game in games:
                pair = game.get('pair')
                if (isinstance(pair, bool) or not isinstance(pair, int)
                        or not 0 <= pair < len(expected_openings)
                        or (game.get('opening'), game.get('opening_position')) != expected_openings[pair]):
                    raise ValueError('promotion report does not match the configured opening pairs')
            points = sum(game['candidate_score'] for game in games if game.get('candidate_score') is not None)
            decision = promotion_decision(points, report.get('completed_games'),
                candidate_tactics, incumbent_tactics, match_games=games,
                minimum_score=self.config['gate']['minimum_score'],
                alpha=self.budget['candidate_alpha'], expected_pairs=self.config['gate']['pairs'])
            if not decision['passed'] or any(report.get(key) != value for key, value in decision.items()):
                raise ValueError('promotion report lacks valid paired evidence for this experiment')

    def __call__(self, candidate_path, report_path):
        if self.stop.requested:
            raise InterruptedError('checkpoint saved; candidate gate deferred until resume')
        # Model loading initializes temporary tensors. Evaluation must not change
        # dropout/training RNG or make an interrupted resume follow another path.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if self.device.type == 'cuda' else []):
            history = json.loads(self.path.read_text())['history']
            current = str(candidate_path.resolve())
            if Path(report_path).exists():
                result = json.loads(Path(report_path).read_text())
                # The report is durable before the pointer. Recover that narrow
                # interruption without rerunning games or losing their evidence.
                if result['passed'] and current not in history:
                    self._validate_promotion_report(result, current, history)
                    atomic_json(self.path, {'artifact': current, 'history': history + [current]})
                print(json.dumps(gate_summary(result, report_path,
                    minimum_score=self.config['gate']['minimum_score'], cached=True)), flush=True)
                return
            candidate = load_checkpoint(candidate_path, device=self.device, expected_config=self.model_config)
            opponents = history[-self.config['gate']['historical_opponents']:]
            reports = []
            identity = self._evidence_identity(current, list(reversed(opponents)))
            for opponent in reversed(opponents):
                if self.stop.requested:
                    raise InterruptedError('gate deferred until resume')
                incumbent = load_checkpoint(opponent, device=self.device, expected_config=self.model_config)
                report = evaluate_candidate(candidate, incumbent, tactics=self.tactics,
                    **{key: self.config['gate'][key] for key in
                       ('openings', 'pairs', 'simulations', 'max_plies', 'minimum_score')},
                    alpha=self.budget['candidate_alpha'], stop=self.stop,
                    progress_path=Path(str(report_path) + '.progress') / ('opponent-' + sha256(Path(opponent) / 'weights.pt')[:16]),
                    progress_identity={'candidate': identity['artifacts'][current],
                                       'opponent': identity['artifacts'][opponent],
                                       'sources': identity['sources']})
                reports.append({'opponent': opponent, **report})
                del incumbent
                if not report['passed']:
                    break
            passed = len(reports) == len(opponents) and all(item['passed'] for item in reports)
            result = {'passed': passed, 'candidate': current, 'opponents': reports,
                      'parent_history': history,
                      'tactics_sha256': sha256(rooted(self.config['tactics'])),
                      'promotion_budget': self.budget, 'evidence_identity': identity}
            if passed:
                self._validate_promotion_report(result, current, history)
            # Only accepted.json determines deployment and self-play generation.
            # A rejected candidate remains inspectable without replacing champion.
            atomic_json(report_path, result)
            if passed:
                atomic_json(self.path, {'artifact': current, 'history': history + [current]})
            print(json.dumps(gate_summary(result, report_path,
                minimum_score=self.config['gate']['minimum_score'])), flush=True)


def pipeline(args, config, model_config):
    if config.get('requires_init_artifact') and not args.init_artifact:
        raise ValueError('this continuation requires --init-artifact; use a fresh RUN_DIR')
    device = torch.device(args.device)
    validate_cuda_runtime(device)
    if args.init_artifact:
        args.init_artifact = str(verify_accepted_artifact(args.init_artifact, expected_config=model_config))
    if hasattr(args, 'job_spec') and args.job_spec:
        verify_snapshot(ROOT)
    # Data selection and teacher budgets are contracts, not path-name hints.
    if not (getattr(args, '_verified_data', None)):
        verify_prepared_data(args.data_dir, config)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    root, data = Path(args.run_dir), Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    request = {'config': config, 'model_config': asdict(model_config),
        'training_sources': training_sources(ROOT),
        'human_manifest_sha256': sha256(data / 'human/manifest.json'),
        'teacher_manifest_sha256': sha256(data / 'teacher/manifest.json'),
        'tactics_sha256': sha256(rooted(config['tactics'])),
        'initial_artifact': str(Path(args.init_artifact).resolve()) if args.init_artifact else None,
        'initial_weights_sha256': sha256(Path(args.init_artifact) / 'weights.pt') if args.init_artifact else None}
    request_path = root / 'request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('run configuration/data changed; resume unchanged or use a new run directory')
    atomic_json(request_path, request)
    human_train = dataset_records(data / 'human', 'train')
    human_validation = dataset_records(data / 'human', 'validation')
    teacher_train = dataset_records(data / 'teacher', 'train')
    teacher_validation = dataset_records(data / 'teacher', 'validation')
    teacher_manifest = json.loads((data / 'teacher/manifest.json').read_text())
    if teacher_manifest['teacher']['human_manifest_sha256'] != sha256(data / 'human/manifest.json'):
        raise ValueError('teacher and human splits have different provenance')
    # Hold out tactical positions and both validation/evaluation corpora from all
    # future self-play updates as well as the initial human import.
    tactics = read_tactics(rooted(config['tactics']))
    excluded = {position_key(chess.Board(item['fen'])) for item in tactics}
    for split in ('validation', 'evaluation'):
        for record in dataset_records(data / 'human', split):
            excluded.add(position_key(validate_record(record)))
    state_path = root / 'curriculum.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'completed': []}
    if args.init_artifact:
        model = load_checkpoint(args.init_artifact, device=device, expected_config=model_config)
    else:
        model = ChessTransformer(model_config).to(device)
    if not (root / 'accepted.json').exists():
        baseline = root / 'initial'
        if not baseline.exists():
            save_checkpoint(model, baseline)
        atomic_json(root / 'accepted.json', {'artifact': str(baseline.resolve()), 'history': [str(baseline.resolve())]})
    settings = config['training']
    with StopRequested() as stop:
        gate = ChampionGate(root, config, model_config, device, stop, warm_start=bool(args.init_artifact))

        def stage(name, kind, train, validation, steps, *, replay=None, replay_fraction=None):
            nonlocal model
            options = {key: settings[key] for key in ('batch_size', 'weight_decay', 'clip_grad', 'checkpoint_every', 'log_every')}
            learning_rate = settings['lr'] if kind == 'human' else settings['teacher_lr']
            if kind == 'selfplay':
                learning_rate = settings.get('selfplay_lr', learning_rate)
            options.update(steps=steps, seed=config['seed'], lr=learning_rate)
            for key in ('validation_limit', 'validation_seed'):
                if key in settings:
                    options[key] = settings[key]
            if name in state['completed']:
                checkpoint = root / name / 'last.pt'
                previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
                expected = training_signature(model, train, validation, options, kind,
                    replay=replay, replay_fraction=replay_fraction)
                if previous['signature'] != expected or previous['step'] != steps:
                    raise ValueError('completed checkpoint configuration, dataset or step changed; use a new run directory')
                restore_checkpoint_rng(previous, next(model.parameters()).device,
                    resume_mode=args.resume_mode, seed=config['seed'], checkpoint=checkpoint)
                model.load_state_dict(previous['model'])
                print(json.dumps({'stage': name, 'status': 'already_complete',
                                  'completed_steps': previous['step'], 'remaining_steps': 0}), flush=True)
                return
            if not train_stage(model, train, validation, root / name,
                               config=options, stage=kind, gate=gate, stop=stop,
                               resume_mode=args.resume_mode, replay=replay, replay_fraction=replay_fraction):
                raise InterruptedError('training checkpoint saved')
            state['completed'].append(name)
            atomic_json(state_path, state)

        # Warm-starting a later run explicitly skips human imitation forever.
        if not args.init_artifact:
            stage('human', 'human', human_train, human_validation, settings['human_steps'])
        if 'teacher' not in state['completed'] and not (root / 'teacher' / 'last.pt').exists():
            accepted_state = json.loads((root / 'accepted.json').read_text())
            if len(accepted_state.get('history', [])) > 1:
                champion_path = Path(accepted_state['artifact'])
                model = load_checkpoint(champion_path, device=device, expected_config=model_config)
                print(json.dumps({'teacher_initialization': 'teacher',
                                  'artifact': str(champion_path)}), flush=True)
        stage('teacher', 'teacher', teacher_train, teacher_validation, settings['teacher_steps'])
        for number in range(config['selfplay']['rounds']):
            if stop.requested:
                raise InterruptedError('stopped between curriculum stages')
            name = f'selfplay-{number + 1:03d}'
            samples_path = root / (name + '.jsonl')
            if not samples_path.exists():
                accepted = json.loads((root / 'accepted.json').read_text())['artifact']
                champion = load_checkpoint(accepted, device=device, expected_config=model_config)
                records, report = selfplay_records(champion,
                    **{key: config['selfplay'][key] for key in ('games', 'simulations', 'max_plies')},
                    seed=config['seed'] + number, excluded=excluded,
                    initial_fen=config['selfplay'].get('initial_fen', chess.STARTING_FEN), stop=stop,
                    progress_path=root / (name + '.progress'),
                    progress_identity={'weights_sha256': sha256(Path(accepted) / 'weights.pt'),
                                       'sources': training_sources(ROOT)})
                del champion
                if not records:
                    raise ValueError('self-play produced no completed games; no fabricated draw targets were written')
                temporary = samples_path.with_suffix('.tmp')
                temporary.write_text(''.join(json.dumps(record) + '\n' for record in records))
                atomic_json(root / (name + '-source.json'), {**report, 'champion': accepted,
                    'champion_weights_sha256': sha256(Path(accepted) / 'weights.pt'),
                    'records_sha256': sha256(temporary)})
                os.replace(temporary, samples_path)
            source = json.loads((root / (name + '-source.json')).read_text())
            if source['records_sha256'] != sha256(samples_path):
                raise ValueError('self-play replay checksum mismatch')
            records = load_records([samples_path])
            fraction = config['selfplay']['teacher_replay_fraction']
            if name not in state['completed'] and not (root / name / 'last.pt').exists():
                # Each round proposes an update to the model that generated its
                # frozen targets. The preceding stage/round may have been rejected.
                champion_path = Path(source['champion'])
                if sha256(champion_path / 'weights.pt') != source['champion_weights_sha256']:
                    raise ValueError('self-play source champion checksum mismatch')
                model = load_checkpoint(champion_path, device=device, expected_config=model_config)
                print(json.dumps({'selfplay_initialization': name,
                                  'artifact': str(champion_path), 'learning_rate':
                                  settings.get('selfplay_lr', settings['teacher_lr'])}), flush=True)
            stage(name, 'selfplay', records, teacher_validation, settings['selfplay_steps'],
                  replay=teacher_train, replay_fraction=fraction)
    print(json.dumps(run_status(root), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'preflight', 'status'))
    parser.add_argument('--config', default=str(ROOT / 'configs/train/inezita.json'))
    parser.add_argument('--data-dir', default=str(ROOT / 'data/processed/inezita'))
    parser.add_argument('--run-dir', default=str(ROOT / 'results/training/inezita'))
    parser.add_argument('--job-spec', type=Path, help='frozen submission specification; validates job inputs')
    parser.add_argument('--archive')
    parser.add_argument('--stockfish', default=os.environ.get('INEZ_STOCKFISH', 'stockfish'))
    parser.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--init-artifact', help='accepted model to improve; skips human initialization')
    parser.add_argument('--resume-mode', choices=('exact', 'portable'), default='exact',
                        help='portable keeps training progress across hardware/PyTorch changes; '
                             'reproducibility is no longer bitwise exact')
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(run_status(args.run_dir), indent=2), flush=True)
        return
    config, model_config = read_config(args.config)
    if args.job_spec:
        spec = verify_job_spec(args.job_spec, source_root=ROOT, runtime=True, resume_mode=args.resume_mode)
        if Path(args.config).resolve() != Path(spec['config']).resolve():
            raise ValueError('configuration differs from the frozen job')
        if args.command in ('train', 'prepare') and Path(args.data_dir).resolve() != Path(spec['data_dir']).resolve():
            raise ValueError('data directory differs from the frozen job')
        if args.command == 'train':
            expected_initial = spec['initial_artifact']
            actual_initial = str(Path(args.init_artifact).resolve()) if args.init_artifact else None
            if (Path(args.run_dir).resolve() != Path(spec['run_dir']).resolve()
                    or actual_initial != expected_initial):
                raise ValueError('run directory or initialization differs from the frozen job')
        if args.command != 'prepare':
            actual_data = verify_prepared_data(spec['data_dir'], config)
            if spec['data_identity'] is not None and actual_data != spec['data_identity']:
                raise ValueError('prepared dataset changed since submission')
            args._verified_data = actual_data
    torch.set_num_threads(args.threads)
    if args.command == 'prepare':
        with exclusive_run(args.data_dir, '.prepare.lock'):
            prepare(args, config)
    elif args.command == 'train':
        with exclusive_run(args.run_dir, '.train.lock'):
            pipeline(args, config, model_config)
    else:
        runtime = validate_cuda_runtime(args.device)
        if torch.device(args.device).type == 'cuda':
            from inez.trainer import supervised_loss
            model = ChessTransformer(model_config).to(args.device)
            features = torch.zeros((1, 64, 119), device=args.device)
            mask = torch.ones((1, 4288), device=args.device, dtype=torch.bool)
            policy = torch.zeros((1, 4288), device=args.device)
            policy[0, 0] = 1
            wdl = torch.tensor([[0.0, 1.0, 0.0]], device=args.device)
            loss, _, _ = supervised_loss(model, (features, mask, policy, wdl))
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('model forward/backward preflight produced nonfinite values')
            torch.cuda.synchronize()
            runtime['model_forward_backward'] = 'passed'
        else:
            with torch.device('meta'):
                model = ChessTransformer(model_config)
        print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()),
                          'device': args.device, 'torch': str(torch.__version__), 'runtime': runtime}))



if __name__ == '__main__':
    try:
        main()
    except InterruptedError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(75)
