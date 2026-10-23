#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Pinned two-checkpoint or external reference Fastchess benchmark, with audited PGNs and artifacts."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'tools'))

import chess
import chess.engine
import chess.pgn
import torch

from inez.curriculum import atomic_json, sha256
from inez.device import validate_cuda_runtime
from inez.uci import DEFAULT_MODEL, resolve_artifact
from train_inezita import resolve_stockfish

BOOK_REVISION = 'a804a97e709f40aa75a00f9d19e84915b7a3a45c'
BOOK_URL = f'https://raw.githubusercontent.com/official-stockfish/books/{BOOK_REVISION}/8moves_v3.pgn.zip'
BOOK_SHA256 = '7e1e9dd118b4bb97d8a8b5b8a790c86e21f8509d59a27d2883767d94477be02e'
STAGES = {'pilot': (10, 0, '60+0.6'), 'baseline': (1000, 10, '60+0.6'),
          'gauntlet': (500, 10, '60+0.6'), 'confirmation': (1000, 1010, '180+1.8')}
REFERENCE_PRESETS = {
    'stockfish-skill-0-n128': {
        'engine': 'stockfish',
        'name': 'stockfish-17.1-skill0-n128',
        'args': [],
        'options': {'Threads': 1, 'Hash': 16, 'Skill Level': 0, 'Move Overhead': 50},
        'nodes': 128,
    },
    'stockfish-elo-1320': {
        'engine': 'stockfish',
        'name': 'stockfish-17.1-elo1320',
        'args': [],
        'options': {'Threads': 1, 'Hash': 64, 'UCI_LimitStrength': 'true', 'UCI_Elo': 1320, 'Move Overhead': 50},
        'nodes': None,
    },
    'stockfish-elo-1600': {
        'engine': 'stockfish',
        'name': 'stockfish-17.1-elo1600',
        'args': [],
        'options': {'Threads': 1, 'Hash': 64, 'UCI_LimitStrength': 'true', 'UCI_Elo': 1600, 'Move Overhead': 50},
        'nodes': None,
    },
    'inezita-material': {
        'engine': 'inezita-baseline',
        'name': 'inezita-material-n128',
        'args': ['--baseline'],
        'options': {'Evaluator': 'material', 'BatchSize': 4},
        'nodes': 128,
    },
    'lc0': {
        'engine': 'lc0',
        'name': 'lc0-reference',
        'args': [],
        'options': {'Threads': 2, 'MinibatchSize': 1, 'MaxPrefetch': 0, 'NNCacheSize': 200000},
        'nodes': None,
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def previous_artifact(run):
    run = Path(run).resolve()
    record = json.loads((run / 'accepted.json').read_text())
    index = record['history'].index(record['artifact'])
    if index == 0:
        raise ValueError('no earlier accepted opponent; supply --opponent')
    saved = Path(record['history'][index - 1])
    # Same copied-cluster relocation rule as the playing-model loader.
    for index in reversed(range(len(saved.parts))):
        if saved.parts[index] == run.name:
            local = run.joinpath(*saved.parts[index + 1:]).resolve()
            if local.is_relative_to(run) and (local / 'manifest.json').is_file():
                return local
            break
    return resolve_artifact(saved if saved.is_absolute() else run / saved)


def artifact_record(path):
    path = resolve_artifact(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('weights') != 'weights.pt':
        raise ValueError('unsupported artifact weight filename')
    digest = sha256(path / 'weights.pt')
    if digest != manifest['weights_sha256']:
        raise ValueError(f'weight checksum mismatch: {path}')
    return {'path': str(path), 'weights_sha256': digest,
            'manifest_sha256': sha256(path / 'manifest.json'), 'config': manifest['config']}


def parse_uci_options(items):
    options = {}
    for item in items or ():
        if '=' not in item:
            raise ValueError(f'UCI option must be NAME=VALUE: {item}')
        name, value = item.split('=', 1)
        name = name.strip()
        if not name:
            raise ValueError(f'empty UCI option name: {item}')
        options[name] = value
    return options


def _resolve_executable(spec):
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which(str(spec))
    if found:
        return Path(found).resolve()
    raise ValueError(f'external engine executable not found: {spec}')


def external_opponent_record(args):
    if getattr(args, 'opponent', None) is not None:
        raise ValueError('cannot combine checkpoint --opponent with --reference or --opponent-engine')
    reference = getattr(args, 'reference', None)
    preset = REFERENCE_PRESETS.get(reference, {}) if reference else {}
    engine_kind = preset.get('engine')
    opponent_engine = getattr(args, 'opponent_engine', None)
    if opponent_engine is not None:
        binary = _resolve_executable(opponent_engine)
    elif engine_kind == 'stockfish':
        binary = Path(resolve_stockfish(os.environ.get('INEZ_STOCKFISH', 'stockfish'))).resolve()
    elif engine_kind == 'inezita-baseline':
        binary = Path(args.binary).expanduser().resolve()
    elif engine_kind == 'lc0':
        env_lc0 = os.environ.get('LC0_BINARY')
        if not env_lc0 and shutil.which('lc0') is None:
            raise ValueError('supply --opponent-engine or set LC0_BINARY for the lc0 reference')
        binary = _resolve_executable(env_lc0 or 'lc0')
    else:
        raise ValueError('external opponent requires --reference or --opponent-engine')

    options = {str(k): str(v) for k, v in preset.get('options', {}).items()}
    options.update(parse_uci_options(getattr(args, 'opponent_option', ())))
    opponent_arg = getattr(args, 'opponent_arg', None)
    extra_args = list(opponent_arg) if opponent_arg else list(preset.get('args', ()))

    weights_spec = getattr(args, 'lc0_weights', None) or (os.environ.get('LC0_WEIGHTS') if engine_kind == 'lc0' else None)
    if weights_spec:
        weights_path = Path(weights_spec).expanduser().resolve()
        if not weights_path.is_file():
            raise ValueError(f'external opponent weights file not found: {weights_path}')
        options['WeightsFile'] = str(weights_path)
    elif 'WeightsFile' in options:
        weights_path = Path(options['WeightsFile']).expanduser().resolve()
        if not weights_path.is_file():
            raise ValueError(f'external opponent weights file not found: {weights_path}')
        options['WeightsFile'] = str(weights_path)
    else:
        weights_path = None
    if engine_kind == 'lc0' and weights_path is None:
        raise ValueError('lc0 reference requires --lc0-weights, LC0_WEIGHTS, or --opponent-option WeightsFile=...')

    opponent_nodes = getattr(args, 'opponent_nodes', None)
    if opponent_nodes is None:
        opponent_nodes = preset.get('nodes')
    opponent_name = getattr(args, 'opponent_name', None) or preset.get('name') or 'inezita-reference'
    if opponent_name == 'inezita-candidate':
        raise ValueError('opponent name must be distinct from inezita-candidate')

    record = {
        'kind': 'external_uci',
        'reference': reference,
        'name': opponent_name,
        'binary': str(binary),
        'binary_sha256': sha256(binary),
        'args': extra_args,
        'options': options,
        'nodes': opponent_nodes,
    }
    if weights_path is not None:
        record['weights_path'] = str(weights_path)
        record['weights_sha256'] = sha256(weights_path)
    return record


def verify_external_opponent_unchanged(opponent):
    if sha256(opponent['binary']) != opponent['binary_sha256']:
        raise ValueError('external opponent binary changed during the tournament')
    if 'weights_path' in opponent and sha256(opponent['weights_path']) != opponent['weights_sha256']:
        raise ValueError('external opponent weights changed during the tournament')


def _configure_value(option, raw_value):
    if option.type == 'check':
        if isinstance(raw_value, bool):
            return raw_value
        lowered = str(raw_value).strip().lower()
        if lowered in ('true', '1', 'yes', 'on'):
            return True
        if lowered in ('false', '0', 'no', 'off'):
            return False
        raise ValueError(f'invalid boolean value for UCI option {option.name}: {raw_value}')
    if option.type == 'spin':
        return int(raw_value)
    return str(raw_value)


def source_record():
    files = [ROOT / 'CMakeLists.txt']
    for directory in ('src', 'include', 'training', 'tools', 'hpc'):
        files += [p for p in (ROOT / directory).rglob('*')
                  if p.suffix in ('.py', '.cpp', '.hpp', '.slurm', '.sh')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def default_book(cache):
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / '8moves_v3.pgn.zip'
    if not archive.exists():
        temporary = cache / f'.book-{os.getpid()}.download'
        try:
            request = urllib.request.Request(BOOK_URL, headers={'User-Agent': 'InezitaBenchmark/0.1.2'})
            with urllib.request.urlopen(request, timeout=120) as remote, temporary.open('wb') as out:
                while block := remote.read(1024 * 1024):
                    out.write(block)
            if sha256(temporary) != BOOK_SHA256:
                raise ValueError('downloaded opening archive checksum mismatch')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(archive) != BOOK_SHA256:
        raise ValueError('cached opening archive checksum mismatch')
    with zipfile.ZipFile(archive) as zipped:
        data = zipped.read('8moves_v3.pgn')
    # Read the named member only; do not extract paths supplied by the archive.
    return data.decode('utf-8-sig'), {'url': BOOK_URL, 'revision': BOOK_REVISION,
        'archive_sha256': BOOK_SHA256, 'training_overlap': 'not fully audited'}


def position_key(board):
    return ' '.join(board.fen().split()[:4])


def excluded_positions():
    excluded = set()
    sources = [ROOT / 'data/openings/mco_openings.tsv', ROOT / 'configs/train/inezita.json']
    with sources[0].open() as stream:
        lines = [row['uci'].split() for row in csv.DictReader(stream, delimiter='\t')]
    lines += json.loads(sources[1].read_text())['gate']['openings']
    for moves in lines:
        board = chess.Board()
        excluded.add(position_key(board))
        for move in moves:
            board.push_uci(move)
            excluded.add(position_key(board))
    return excluded, {str(path.relative_to(ROOT)): sha256(path) for path in sources}


def select_openings(text, start, pairs, excluded):
    stream, selected, seen = io.StringIO(text), [], set()
    eligible = 0
    while (game := chess.pgn.read_game(stream)) is not None:
        if game.errors:
            raise ValueError(f'invalid opening PGN: {game.errors}')
        moves = list(game.mainline_moves())
        board = game.end().board()
        if len(moves) > 24:
            raise ValueError('opening book contains a full game or a line longer than 24 plies')
        if not board.is_valid() or board.outcome(claim_draw=True) is not None:
            continue
        key = position_key(board)
        if key in seen or key in excluded:
            continue
        seen.add(key)
        eligible += 1
        if eligible <= start:
            continue
        clean = chess.pgn.Game.from_board(board)
        clean.headers['Result'] = '*'
        selected.append(clean)
        if len(selected) == pairs:
            return selected
    raise ValueError(f'opening book lacks {start + pairs} unique eligible positions')


def audit_games(path, openings, opponent_name='inezita-incumbent'):
    summary = {'games': 0, 'wins': 0, 'draws': 0, 'losses': 0,
               'unfinished': 0, 'faults': {}, 'pairs': 0, 'pentanomial': [0] * 5}
    groups, faults = defaultdict(list), Counter()
    if not path.exists():
        return summary
    with path.open() as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            if game.errors:
                raise ValueError(f'invalid output PGN: {game.errors}')
            summary['games'] += 1
            round_id = int(game.headers['Round']) - 1
            if not 0 <= round_id < len(openings):
                raise ValueError('output PGN has an unexpected opening round')
            opening = openings[round_id]
            prefix = list(opening.mainline_moves())
            moves = list(game.mainline_moves())
            if game.board().fen() != opening.board().fen() or moves[:len(prefix)] != prefix:
                raise ValueError('output PGN opening/history differs from the scheduled pair')
            white, black = game.headers['White'], game.headers['Black']
            if {white, black} != {'inezita-candidate', opponent_name}:
                raise ValueError('unexpected player identities in output PGN')
            termination = game.headers.get('Termination', 'missing')
            if termination != 'normal':
                faults[termination] += 1
            result = game.headers['Result']
            if result == '*':
                summary['unfinished'] += 1
                continue
            score = {'1-0': 1, '1/2-1/2': 0.5, '0-1': 0}[result]
            score = score if white == 'inezita-candidate' else 1 - score
            summary[{1: 'wins', 0.5: 'draws', 0: 'losses'}[score]] += 1
            groups[round_id].append((white, score))
    for games in groups.values():
        if len(games) > 2 or len({white for white, _ in games}) != len(games):
            raise ValueError('duplicate game or missing color reversal in an opening pair')
        if len(games) == 2:
            summary['pairs'] += 1
            summary['pentanomial'][round(sum(score for _, score in games) * 2)] += 1
    summary['faults'] = dict(faults)
    return summary


def fastchess_command(args, output, candidate, opponent):
    command = [str(args.fastchess), '-engine', f'cmd={args.binary}', 'name=inezita-candidate',
        f'option.ModelPath={candidate["path"]}', f'option.Device={args.device}',
        f'option.Threads={args.threads}']
    if opponent.get('kind') == 'external_uci':
        if args.nodes:
            command.append(f'nodes={args.nodes}')
        command += ['-engine', f'cmd={opponent["binary"]}', f'name={opponent["name"]}']
        if opponent.get('args'):
            command.append(f'args={" ".join(opponent["args"])}')
        for key, value in opponent.get('options', {}).items():
            command.append(f'option.{key}={value}')
        if opponent.get('nodes'):
            command.append(f'nodes={opponent["nodes"]}')
        command += ['-each', 'proto=uci', f'tc={args.time_control}']
    else:
        command += ['-engine', f'cmd={args.binary}', 'name=inezita-incumbent',
            f'option.ModelPath={opponent["path"]}', f'option.Device={args.device}',
            f'option.Threads={args.threads}', '-each', 'proto=uci', f'tc={args.time_control}']
        if args.nodes:
            command.append(f'nodes={args.nodes}')
    command += ['-openings', f'file={output / "openings.pgn"}', 'format=pgn', 'order=sequential',
        '-rounds', str(args.pairs), '-repeat', '-concurrency', '1', '-report', 'penta=true',
        '-pgnout', f'file={output / "games.pgn"}', 'append=false', 'nodes=true', 'timeleft=true',
        '-config', f'outname={output / "fastchess-state.json"}',
        '-log', f'file={output / "fastchess.log"}', 'level=info', 'engine=true']
    return command


def wait_for_tournament(child):
    """Forward interrupts and always reap the child before restoring handlers."""
    previous = {}

    def interrupted(signum, frame):
        if child.poll() is None:
            try:
                child.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        raise InterruptedError(f'benchmark interrupted by signal {signum}')

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        return child.wait()
    finally:
        try:
            if child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        finally:
            for signum, handler in previous.items():
                # Native libraries may install C handlers represented by None.
                # None is a valid getsignal result but not a signal() argument.
                signal.signal(signum, signal.SIG_DFL if handler is None else handler)


def execute(args, report):
    report['cuda_preflight'] = validate_cuda_runtime(args.device)
    candidate = artifact_record(args.model)
    external_mode = bool(getattr(args, 'reference', None) or getattr(args, 'opponent_engine', None))
    if external_mode:
        opponent = external_opponent_record(args)
        opponent_name = opponent['name']
    else:
        opponent = artifact_record(args.opponent or previous_artifact(args.model))
        if candidate['weights_sha256'] == opponent['weights_sha256']:
            raise ValueError('candidate and opponent have identical weights')
        opponent_name = 'inezita-incumbent'
    git_env = {**os.environ, 'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_NOSYSTEM': '1'}
    report.update(candidate=candidate, opponent=opponent, sources=source_record(),
        binary_sha256=sha256(args.binary), fastchess_sha256=sha256(args.fastchess),
        fastchess_version=subprocess.check_output([str(args.fastchess), '-version'], text=True).strip(),
        git_revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, env=git_env).strip(),
        git_status=subprocess.check_output(['git', '-C', str(ROOT), 'status', '--short'], text=True, env=git_env),
        python=platform.python_version(), torch=str(torch.__version__), chess=chess.__version__,
        cuda_runtime=torch.version.cuda, precision='float32',
        host=platform.node(), platform=platform.platform(), device=args.device, threads=args.threads,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    report['device_name'] = torch.cuda.get_device_name(0) if args.device == 'cuda' else platform.processor()
    if args.openings:
        text = args.openings.read_text(encoding='utf-8-sig')
        book = {'path': str(args.openings), 'training_overlap': 'not fully audited'}
    else:
        text, book = default_book(args.cache)
    book['source_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    excluded, exclusions = excluded_positions()
    openings = select_openings(text, args.start, args.pairs, excluded)
    (args.output / 'openings.pgn').write_text('\n\n'.join(str(game) for game in openings) + '\n')
    report.update(book={**book, 'excluded_sources': exclusions, 'eligible_start': args.start,
                       'selected_sha256': sha256(args.output / 'openings.pgn')},
                  stage=args.stage, pairs=args.pairs, time_control=args.time_control, nodes=args.nodes)
    env = {**os.environ, 'INEZ_PYTHON': sys.executable, 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('INEZ_BASELINE', None)
    with chess.engine.SimpleEngine.popen_uci(str(args.binary), env=env, timeout=120) as engine:
        if engine.id.get('name') != 'Inezita 0.1.2' or 'ModelPath' not in engine.options:
            raise ValueError('benchmark requires the Inezita 0.1.2 neural UCI launcher')
        engine.configure({'ModelPath': candidate['path'], 'Device': args.device, 'Threads': args.threads})
        engine.ping()
        # Loading weights alone did not expose the V100/cu130 mismatch.
        # Run the actual neural forward/search path before a tournament.
        result = engine.play(chess.Board(), chess.engine.Limit(nodes=1))
        if result.move not in chess.Board().legal_moves:
            raise RuntimeError('neural preflight returned an illegal move')
    if external_mode:
        cmd = [opponent['binary'], *opponent.get('args', ())]
        with chess.engine.SimpleEngine.popen_uci(cmd, env=env, timeout=120) as engine:
            opponent['uci_id'] = dict(engine.id)
            configured = {}
            for key, raw_value in opponent.get('options', {}).items():
                if key not in engine.options:
                    raise ValueError(f'external opponent {opponent["name"]} does not support UCI option: {key}')
                configured[key] = _configure_value(engine.options[key], raw_value)
            if configured:
                engine.configure(configured)
            engine.ping()
            result = engine.play(chess.Board(), chess.engine.Limit(nodes=1))
            if result.move not in chess.Board().legal_moves:
                raise RuntimeError('external opponent preflight returned an illegal move')
    else:
        with chess.engine.SimpleEngine.popen_uci(str(args.binary), env=env, timeout=120) as engine:
            if engine.id.get('name') != 'Inezita 0.1.2' or 'ModelPath' not in engine.options:
                raise ValueError('benchmark requires the Inezita 0.1.2 neural UCI launcher')
            engine.configure({'ModelPath': opponent['path'], 'Device': args.device, 'Threads': args.threads})
            engine.ping()
            result = engine.play(chess.Board(), chess.engine.Limit(nodes=1))
            if result.move not in chess.Board().legal_moves:
                raise RuntimeError('neural preflight returned an illegal move')
    if args.stage != 'pilot':
        if args.pilot_dir is None:
            raise ValueError('supply --pilot-dir from a completed 20-game pilot before a larger run')
        pilot = json.loads((args.pilot_dir / 'manifest.json').read_text())
        if pilot['stage'] != 'pilot' or pilot['status'] != 'complete' or pilot['pairs'] < 10:
            raise ValueError('pilot must be a complete, fault-free run of at least 20 games')
        for key in ('candidate', 'opponent', 'sources', 'binary_sha256', 'fastchess_sha256',
                    'device', 'device_name', 'threads', 'nodes', 'python', 'torch', 'chess', 'cuda_runtime', 'precision'):
            if report[key] != pilot[key]:
                raise ValueError(f'pilot context changed: {key}; run a fresh pilot')
        if book['source_sha256'] != pilot['book']['source_sha256']:
            raise ValueError('opening source differs from the pilot')
        if args.start < pilot['book']['eligible_start'] + pilot['pairs']:
            raise ValueError('larger run overlaps the pilot opening slice')
        if args.stage in ('baseline', 'gauntlet') and args.time_control != pilot['time_control']:
            raise ValueError(f'{args.stage} clock differs from the pilot')
        report['pilot_manifest_sha256'] = sha256(args.pilot_dir / 'manifest.json')
    command = fastchess_command(args, args.output, candidate, opponent)
    report.update(command=command, status='running', match_started_utc=utc_now())
    atomic_json(args.output / 'manifest.json', report)
    print('Starting:', ' '.join(command), flush=True)
    print('Output:', args.output, flush=True)
    with (args.output / 'tournament.out').open('w') as out:
        child = subprocess.Popen(command, cwd=args.output, env=env, stdout=out, stderr=subprocess.STDOUT)
        try:
            report['returncode'] = wait_for_tournament(child)
        finally:
            report['summary'] = audit_games(args.output / 'games.pgn', openings, opponent_name=opponent_name)
    if artifact_record(candidate['path']) != candidate:
        raise ValueError('model artifacts changed during the tournament')
    if external_mode:
        verify_external_opponent_unchanged(opponent)
    elif artifact_record(opponent['path']) != opponent:
        raise ValueError('model artifacts changed during the tournament')
    if source_record() != report['sources'] or sha256(args.binary) != report['binary_sha256']:
        raise ValueError('engine sources or launcher changed during the tournament')
    if sha256(args.fastchess) != report['fastchess_sha256']:
        raise ValueError('Fastchess changed during the tournament')
    summary = report['summary']
    good = (report['returncode'] == 0 and summary['games'] == args.pairs * 2
            and summary['pairs'] == args.pairs and not summary['faults'] and not summary['unfinished'])
    report['status'] = 'complete' if good else 'incomplete_or_faulted'
    report['elapsed_seconds'] = time.monotonic() - args.started
    report['projected_1000_game_hours'] = report['elapsed_seconds'] / max(1, summary['games']) * 1000 / 3600
    report['projected_2000_game_hours'] = report['elapsed_seconds'] / max(1, summary['games']) * 2000 / 3600
    print(json.dumps(report['summary'], indent=2), flush=True)
    return 0 if good else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, default='pilot')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--opponent', type=Path)
    parser.add_argument('--reference', choices=tuple(REFERENCE_PRESETS),
                        help='named external reference preset (e.g. stockfish-elo-1320, lc0)')
    parser.add_argument('--opponent-engine', type=Path,
                        help='external UCI engine binary path when not using a checkpoint opponent')
    parser.add_argument('--opponent-name', help='player name in PGN/Fastchess for external opponent')
    parser.add_argument('--opponent-arg', action='append', default=[],
                        help='CLI argument passed to external opponent binary (repeatable)')
    parser.add_argument('--opponent-option', action='append', default=[],
                        help='UCI option NAME=VALUE for external opponent (repeatable)')
    parser.add_argument('--opponent-nodes', type=int,
                        help='optional per-move node/simulation cap for external opponent only')
    parser.add_argument('--lc0-weights', type=Path,
                        help='network weights file for lc0 reference opponent')
    parser.add_argument('--fastchess', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/inezita')
    parser.add_argument('--openings', type=Path)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/raw/benchmarks')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-dir', type=Path)
    parser.add_argument('--pairs', type=int)
    parser.add_argument('--start', type=int)
    parser.add_argument('--time-control')
    parser.add_argument('--nodes', type=int, help='optional simulations cap; the clock can also bind, so audit actual nodes')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    pairs, start, clock = STAGES[args.stage]
    args.pairs = pairs if args.pairs is None else args.pairs
    args.start = start if args.start is None else args.start
    args.time_control = args.time_control or clock
    if (args.pairs < 1 or args.start < 0 or not 1 <= args.threads <= 64
            or args.nodes is not None and args.nodes < 1
            or args.opponent_nodes is not None and args.opponent_nodes < 1):
        parser.error('invalid pair count, opening offset, threads or simulation budget')
    if args.opponent is not None and (args.reference is not None or args.opponent_engine is not None):
        parser.error('cannot combine checkpoint --opponent with --reference or --opponent-engine')
    for name in ('model', 'opponent', 'opponent_engine', 'lc0_weights', 'fastchess', 'binary', 'openings', 'cache', 'output', 'pilot_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'format_version': 1, 'status': 'preparing', 'started_utc': utc_now()}
    try:
        return execute(args, report)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'error',
                      error=str(error))
        print(f'Inezita benchmark failed: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        report['ended_utc'] = utc_now()
        atomic_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    raise SystemExit(main())
