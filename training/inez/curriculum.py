# SPDX-License-Identifier: GPL-3.0-only
"""Game-safe human initialization, Stockfish relabeling and independent gates."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import tempfile
import threading
import zipfile

import chess
import chess.engine
import chess.pgn

from .encoding import wdl_from_result
from .openings import _is_train_bucket, default_opening_book, opening_records
from .puct import PIECE_VALUES, NeuralEvaluator, _loss_on_square, _max_hanging_loss, choose_move, search
from .records import (assign_splits, audit_splits, game_fingerprint, load_records,
                      validate_record, write_dataset, read_json)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def position_key(board):
    return ' '.join(board.fen(en_passant='legal').split()[:4])


def read_tactics(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError('tactical suite must be a nonempty list')
    names = set()
    for item in data:
        board = chess.Board(item['fen'])
        if item['name'] in names or not board.is_valid() or not item['best_moves']:
            raise ValueError('invalid or duplicate tactical position')
        names.add(item['name'])
        if any(chess.Move.from_uci(move) not in board.legal_moves for move in item['best_moves']):
            raise ValueError('tactical solution contains an illegal first move')
    return data


@contextmanager
def pgn_stream(path):
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist() if name.endswith('.pgn'))
            if len(names) != 1:
                raise ValueError('PGN zip must contain exactly one PGN file')
            with archive.open(names[0]) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig') as stream:
                yield stream
    elif path.suffix == '.zst':
        try:
            from compression import zstd as _py_zstd
        except ImportError:
            _py_zstd = None
        if _py_zstd is not None:
            with _py_zstd.open(path, 'rt', encoding='utf-8-sig') as stream:
                yield stream
        else:
            with subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL) as process:
                try:
                    with io.TextIOWrapper(process.stdout, encoding='utf-8-sig') as stream:
                        yield stream
                finally:
                    process.terminate() if process.poll() is None else None
                    process.wait()
    else:
        with path.open(encoding='utf-8-sig') as stream:
            yield stream


def make_record(board, moves, ply, result, source, game_id, policy, *, kind='game_result', wdl=None, teacher=None):
    sample = board.copy(stack=True)
    for move in moves[:ply]:
        sample.push_uci(move)
    return {'record_version': 1, 'encoding_version': 1, 'source_id': source,
            'game_id': game_id, 'initial_fen': board.fen(en_passant='fen'),
            'moves': moves, 'ply': ply, 'result': result, 'policy': policy,
            'wdl': list(wdl_from_result(result, sample.turn) if wdl is None else wdl),
            'label_kind': kind, 'teacher': teacher}


def _ensure_nonterminal_train_moves(board, prefix, seed, initial_fen):
    """Extend prefix with non-terminal legal moves if needed so it hashes to the train split."""
    if _is_train_bucket(initial_fen, prefix, str(seed)):
        return prefix
    queue = [(board, prefix)]
    for _ in range(4):
        next_queue = []
        for cursor, moves in queue:
            for move in cursor.legal_moves:
                child = cursor.copy(stack=False)
                child.push(move)
                if child.outcome(claim_draw=False) is not None:
                    continue
                candidate = moves + [move.uci()]
                if _is_train_bucket(initial_fen, candidate, str(seed)):
                    return candidate
                if len(next_queue) < 48:
                    next_queue.append((child, candidate))
        queue = next_queue
        if not queue:
            break
    return None


def _blunder_counterfactuals(initial, moves, source, seed, rng, excluded, max_branches):
    """Generate off-policy piece-blunder branches teaching recapture policy and material-loss value."""
    if max_branches < 1 or len(moves) < 8:
        return []
    initial_fen = initial.fen(en_passant='fen')
    candidate_plies = list(range(4, min(len(moves), 56)))
    rng.shuffle(candidate_plies)
    selected_plies = set(candidate_plies[:max_branches * 4])
    if not selected_plies:
        return []

    generated = []
    board = initial.copy(stack=True)
    for ply, game_move_uci in enumerate(moves):
        if ply > max(selected_plies) or len(generated) >= max_branches * 2:
            break
        if ply in selected_plies and not board.is_game_over(claim_draw=False):
            turn = board.turn
            if _max_hanging_loss(board, turn) < 1.5 and _max_hanging_loss(board, not turn) < 1.5:
                legal_moves = list(board.legal_moves)
                rng.shuffle(legal_moves)
                for blunder in legal_moves:
                    if blunder.uci() == game_move_uci or blunder.promotion:
                        continue
                    mover = board.piece_at(blunder.from_square)
                    if mover is None or mover.piece_type not in (
                        chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN
                    ):
                        continue
                    captured = board.piece_at(blunder.to_square)
                    captured_val = PIECE_VALUES[captured.piece_type] if captured else 0.0
                    mover_val = PIECE_VALUES[mover.piece_type]
                    if captured_val > mover_val - 2.0:
                        continue
                    after_blunder = board.copy(stack=True)
                    after_blunder.push(blunder)
                    if after_blunder.is_game_over(claim_draw=False) or after_blunder.is_check():
                        continue
                    loss_at_dest = _loss_on_square(after_blunder, blunder.to_square, turn)
                    if captured_val - loss_at_dest > -2.0:
                        continue
                    best_recap = None
                    best_recap_val = math.inf
                    for recap in after_blunder.legal_moves:
                        if recap.to_square != blunder.to_square:
                            continue
                        recap_piece = after_blunder.piece_at(recap.from_square)
                        if recap_piece is None:
                            continue
                        after_recap = after_blunder.copy(stack=True)
                        after_recap.push(recap)
                        if after_recap.is_game_over(claim_draw=False):
                            continue
                        recap_loss = _loss_on_square(after_recap, recap.to_square, after_blunder.turn)
                        if mover_val - recap_loss >= 2.0 and PIECE_VALUES[recap_piece.piece_type] < best_recap_val:
                            best_recap = (recap, after_recap)
                            best_recap_val = PIECE_VALUES[recap_piece.piece_type]
                    if best_recap is None:
                        continue
                    recap_move, after_recap = best_recap
                    key_blunder = position_key(after_blunder)
                    key_recap = position_key(after_recap)
                    if key_blunder in excluded or key_recap in excluded or key_blunder == key_recap:
                        continue
                    prefix = moves[:ply] + [blunder.uci(), recap_move.uci()]
                    branch_moves = _ensure_nonterminal_train_moves(after_recap, prefix, seed, initial_fen)
                    if branch_moves is None:
                        continue
                    reply_move = next(iter(after_recap.legal_moves)).uci()
                    winner_result = '1-0' if after_blunder.turn == chess.WHITE else '0-1'
                    branch_id = hashlib.sha256((initial_fen + ' '.join(branch_moves)).encode()).hexdigest()
                    rec1 = make_record(
                        initial, branch_moves, ply + 1, winner_result, source, branch_id,
                        {recap_move.uci(): 1.0},
                    )
                    rec2 = make_record(
                        initial, branch_moves, ply + 2, winner_result, source, branch_id,
                        {reply_move: 1.0},
                    )
                    validate_record(rec1)
                    validate_record(rec2)
                    generated.extend((rec1, rec2))
                    break
        board.push_uci(game_move_uci)
    return generated


def _is_normal_termination(header):
    """Verify a game ended by standard chess rules or tablebase/engine adjudication, not time forfeit or abandonment."""
    if not isinstance(header, str):
        return False
    low = header.strip().lower()
    if not low:
        return True
    if any(bad in low for bad in ('forfeit', 'abandoned', 'unplayed', 'stalled',
                                  'infraction', 'unterminated', 'emergency', 'disconnect', 'timeout')):
        return False
    return low in ('normal', 'adjudication', 'gui adjudication', 'gui tb adjudication',
                   'adjudication - tcec draw rule', 'white mates', 'black mates',
                   '3-fold rep', '3-fold repetition', '50-move rule', 'fifty moves rule',
                   'insufficient material')


def _estimated_game_seconds(header, game_headers=None):
    """Estimate initial + 40-move increment time budget in seconds across PGN, FIDE OTB broadcast, and TCEC formats."""
    headers = game_headers or {}
    event_text = ' '.join(str(headers.get(k, '')) for k in ('Event', 'BroadcastName', 'Site')).lower()
    if any(tag in event_text for tag in ('blitz', 'bullet', 'rapid', 'armageddon',
                                         'hyperbullet', 'ultrabullet', 'titled tuesday')):
        return 0
    if not isinstance(header, str):
        return 0
    tc = header.strip().lower()
    if not tc or tc == '-':
        if 'tcec' in event_text or 'lichess.org/broadcast/' in event_text:
            return 1800
        return 0
    match_moves = re.match(r'^(\d+)/(\d+)', tc)
    if match_moves:
        return int(match_moves.group(2))
    match_desc = re.search(r'(\d+)\s*(?:minutes|minute|min|m\b)', tc)
    if match_desc:
        return int(match_desc.group(1)) * 60
    match_plus = re.match(r'^(\d+)\+(\d+(?:\.\d+)?)$', tc)
    if match_plus:
        base = int(match_plus.group(1))
        increment = float(match_plus.group(2))
        if base < 0 or increment < 0:
            return 0
        # Official FIDE OTB broadcasts record classical controls as minutes+seconds (e.g. 90+30, 60+30, 45+30).
        if 40 <= base <= 180 and increment >= 10:
            return base * 60 + int(40 * increment)
        return int(base + 40 * increment)
    return 0


def import_games(path, output, *, uri, license_name, tactics, max_games=1000,
                 samples_per_game=16, min_ply=8, min_rating=2300, seed=20261019,
                 openings_path=None, blunder_samples_per_game=0,
                 min_time_control_seconds=0, require_normal_termination=False):
    """Import a bounded corpus; remove held-out tactical boards and split overlap."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if (max_games < 1 or samples_per_game < 1 or min_ply < 0 or blunder_samples_per_game < 0
            or type(min_time_control_seconds) is not int or min_time_control_seconds < 0
            or not isinstance(require_normal_termination, bool)):
        raise ValueError('invalid importer bounds')
    paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
    uris = list(uri) if isinstance(uri, (list, tuple)) else [uri] * len(paths)
    if not paths or len(paths) != len(uris):
        raise ValueError('archive paths and URIs must be nonempty and aligned')

    if openings_path is not None and not Path(openings_path).is_file():
        raise ValueError('configured preparation openings file does not exist')
    excluded = {position_key(chess.Board(item['fen'])) for item in read_tactics(tactics)}
    mco_records, mco_sources, mco_keys = [], {}, set()
    if openings_path is not None and Path(openings_path).is_file():
        mco_records, mco_sources = opening_records(openings_path, excluded=excluded, seed=str(seed))
        mco_keys = {position_key(validate_record(r)) for r in mco_records}

    sources = dict(mco_sources)
    archive_sources = []
    for archive_path, archive_uri in zip(paths, uris):
        digest = sha256(archive_path)
        source_id = 'elite-' + digest[:16]
        sources[source_id] = {'uri': archive_uri, 'license': license_name, 'sha256': digest}
        archive_sources.append((archive_path, source_id))

    rng = random.Random(seed)
    records, seen = [], set()
    stats = {
        'read_games': 0,
        'accepted_games': 0,
        'invalid_or_filtered_games': 0,
        'mco_opening_records': len(mco_records),
        'blunder_counterfactual_records': 0,
    }
    logging.getLogger('chess.pgn').setLevel(logging.CRITICAL)
    with ExitStack() as stack:
        streams = [
            (stack.enter_context(pgn_stream(archive_path)), source_id)
            for archive_path, source_id in archive_sources
        ]
        active = list(streams)
        cursor = 0
        while stats['accepted_games'] < max_games and active:
            stream, source = active[cursor % len(active)]
            game = chess.pgn.read_game(stream)
            if game is None:
                active.pop(cursor % len(active))
                continue
            cursor += 1
            stats['read_games'] += 1
            try:
                ratings = [int(game.headers.get(key, '0')) for key in ('WhiteElo', 'BlackElo')]
                initial = game.board()
                termination = game.headers.get('Termination', 'Normal')
                if (game.errors or initial.chess960 or not initial.is_valid()
                        or game.headers.get('Variant', 'Standard') != 'Standard'
                        or min(ratings) < min_rating
                        or (require_normal_termination and not _is_normal_termination(termination))
                        or (min_time_control_seconds > 0
                            and _estimated_game_seconds(game.headers.get('TimeControl', ''), game.headers) < min_time_control_seconds)):
                    raise ValueError('filtered game')
                result = game.headers.get('Result')
                if result not in ('1-0', '0-1', '1/2-1/2'):
                    raise ValueError('unfinished game')
                moves = [move.uci() for move in game.mainline_moves()]
                initial_fen = initial.fen(en_passant='fen')
                is_train_game = _is_train_bucket(initial_fen, moves, str(seed))
                effective_min_ply = min_ply if is_train_game else max(min_ply, 12 if mco_keys or min_ply < 8 else min_ply)
                if len(moves) <= effective_min_ply:
                    raise ValueError('short game')
                identity = hashlib.sha256((initial.fen() + ' '.join(moves)).encode()).hexdigest()
                if identity in seen:
                    continue
                seen.add(identity)
                candidates = rng.sample(
                    range(effective_min_ply, len(moves)),
                    min(samples_per_game * (2 if mco_keys else 1), len(moves) - effective_min_ply),
                )
                if mco_keys and len(candidates) > samples_per_game:
                    endgame_cut = max(effective_min_ply, int(len(moves) * 0.65))
                    late = [p for p in candidates if p >= endgame_cut]
                    early = [p for p in candidates if p < endgame_cut]
                    ordered_candidates = late[:max(1, samples_per_game // 3)] + early + late[max(1, samples_per_game // 3):]
                else:
                    ordered_candidates = sorted(candidates)
                accepted = []
                for ply in ordered_candidates:
                    if len(accepted) >= samples_per_game:
                        break
                    record = make_record(initial, moves, ply, result, source, identity, {moves[ply]: 1.0})
                    board = validate_record(record)
                    key = position_key(board)
                    if key in excluded or key in mco_keys:
                        continue
                    accepted.append(record)
                if not accepted:
                    continue
                accepted.sort(key=lambda r: r['ply'])
                records.extend(accepted)
                if is_train_game and blunder_samples_per_game > 0:
                    blunders = _blunder_counterfactuals(
                        initial, moves, source, seed, rng, excluded | mco_keys, blunder_samples_per_game
                    )
                    records.extend(blunders)
                    stats['blunder_counterfactual_records'] += len(blunders)
                stats['accepted_games'] += 1
            except (ValueError, TypeError, OverflowError):
                stats['invalid_or_filtered_games'] += 1
    if not records:
        raise ValueError('no games survived import filters')
    records = mco_records + records
    splits = assign_splits(records, seed=str(seed))
    # Reserve held-out board identities before training, even for transpositions
    # from otherwise distinct games. Evaluation has priority over validation.
    # MCO opening records are always prioritized in train and never placed in validation/evaluation.
    reserved, filtered = set(excluded), {}
    for split in ('evaluation', 'validation'):
        items = []
        for r in splits[split]:
            if r['source_id'].startswith('mco-'):
                continue
            key = position_key(validate_record(r))
            if key in reserved or key in mco_keys:
                continue
            items.append(r)
            reserved.add(key)
        filtered[split] = items

    train_items = []
    seen_train_samples = set()
    for r in splits['train']:
        board = validate_record(r)
        key = position_key(board)
        if key in reserved:
            continue
        sample_id = (game_fingerprint(r), r['ply'])
        if sample_id in seen_train_samples:
            continue
        seen_train_samples.add(sample_id)
        train_items.append(r)
    filtered['train'] = train_items

    records = [r for split in ('train', 'validation', 'evaluation') for r in filtered[split]]
    if any(not filtered[split] for split in filtered):
        raise ValueError('need more imported games to populate all three independent splits')
    active_sources = {r['source_id'] for r in records}
    sources = {k: v for k, v in sources.items() if k in active_sources}
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        raw = Path(temporary) / 'records.jsonl'
        raw.write_text(''.join(json.dumps(r) + '\n' for r in records))
        staging = Path(temporary) / 'human'
        manifest = write_dataset([raw], staging, sources, seed=str(seed), require_disjoint=True)
        manifest['preparation'] = {**stats, 'min_rating': min_rating, 'min_ply': min_ply,
                                   'samples_per_game': samples_per_game, 'max_games': max_games,
                                   'blunder_samples_per_game': blunder_samples_per_game,
                                   'min_time_control_seconds': min_time_control_seconds,
                                   'require_normal_termination': require_normal_termination,
                                   'openings_sha256': sha256(openings_path) if openings_path and Path(openings_path).is_file() else None,
                                   'tactics_sha256': sha256(tactics), 'labels': 'human_moves_and_game_result'}
        atomic_json(staging / 'manifest.json', manifest)
        os.rename(staging, output)
    return manifest


def stockfish_labels(record, engine, *, nodes=10000, multipv=4, temperature=0.15):
    """Replace both human targets with independent Stockfish analysis.

    MultiPV scores become a documented Boltzmann distribution over expected
    outcomes, not search visit counts. The best line's native UCI WDL supplies
    all three value probabilities; missing WDL fails rather than inventing it.
    """
    if nodes < 1 or multipv < 1 or temperature <= 0:
        raise ValueError('invalid teacher settings')
    board = validate_record(record)
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        raise ValueError('teacher corpus must contain nonterminal positions')
    requested_multipv = max(multipv, 8) if any(m.promotion for m in board.legal_moves) else multipv
    # A fresh game token sends ucinewgame before every independent label,
    # clearing Stockfish's search history/hash across workers and restarts.
    infos = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=object(),
                           multipv=min(requested_multipv, board.legal_moves.count()))
    if not isinstance(infos, list):
        infos = [infos]
    if not infos or any('wdl' not in info or not info.get('pv') for info in infos):
        raise ValueError('Stockfish must provide UCI_ShowWDL and principal variations')
    outcomes = [info['wdl'].pov(board.turn) for info in infos]
    values = []
    for info, wdl in zip(infos, outcomes):
        value = (wdl.wins - wdl.losses) / 1000
        score = info['score'].pov(board.turn) if 'score' in info else None
        if score is not None:
            if score.is_mate() and score.mate() != 0:
                value += 0.5 / score.mate()
            elif score.score() is not None:
                value += 0.12 * math.tanh(score.score() / 600.0)
        values.append(value)
    weights = [math.exp((value - max(values)) / temperature) for value in values]
    total_weight = sum(weights)
    policy = {info['pv'][0].uci(): weight / total_weight for info, weight in zip(infos, weights)}
    if str(record.get('source_id', '')).startswith('mco-') and record.get('policy'):
        # Preserve MCO opening repertoire coverage while anchoring weights to Stockfish.
        mco_policy = record['policy']
        merged = {move: 0.55 * policy.get(move, 0.0) + 0.45 * mco_policy.get(move, 0.0)
                  for move in set(policy) | set(mco_policy)}
        merged_total = sum(merged.values())
        if merged_total > 0:
            policy = {move: val / merged_total for move, val in merged.items() if val > 0}
    best = max(range(len(values)), key=values.__getitem__)
    wdl = outcomes[best]
    labeled = copy.deepcopy(record)
    labeled.update(policy=policy, wdl=[wdl.wins / 1000, wdl.draws / 1000, wdl.losses / 1000],
                   label_kind='teacher_wdl', teacher={'name': 'Stockfish',
                   'version': engine.id.get('name', 'unknown'), 'limits': {'nodes': nodes}})
    validate_record(labeled)
    return labeled


def verify_dataset(directory, *, full=False):
    """Check a completed dataset using bounded memory before reuse or relabeling."""
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'incomplete dataset: missing {manifest_path}')
    manifest = read_json(manifest_path.read_text())
    if (not isinstance(manifest, dict)
            or any(manifest.get(key) != 1 for key in ('manifest_version', 'record_version', 'encoding_version'))
            or not isinstance(manifest.get('splits'), dict)):
        raise ValueError(f'unsupported dataset manifest: {manifest_path}')
    for split in ('train', 'validation', 'evaluation'):
        expected = manifest.get('splits', {}).get(split, {})
        count = expected.get('records') if isinstance(expected, dict) else None
        if type(count) is not int or count < 1:
            raise ValueError(f'dataset requires a nonempty {split} split: {directory}')
        path = directory / (split + '.jsonl')
        if not path.is_file() or sha256(path) != expected.get('sha256'):
            raise ValueError(f'dataset checksum mismatch: {path}')
        actual = 0
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    if full:
                        try:
                            validate_record(read_json(line))
                        except (ValueError, TypeError, OverflowError) as error:
                            raise ValueError(f'{path}:{number}: {error}') from error
                    actual += 1
        if actual != count:
            raise ValueError(f'dataset manifest record count mismatch: {path}')
    return manifest


def _write_records_atomic(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _teacher_cache_key(record, board):
    # Stockfish receives the move stack: fifty-move and repetition states must
    # not share labels merely because piece placement and castling rights match.
    return (board.root().fen(en_passant='fen'),
            tuple(move.uci() for move in board.move_stack),
            tuple(sorted(record['policy'].items())) if str(record.get('source_id', '')).startswith('mco-') else None)


def relabel_dataset(source, output, stockfish, *, nodes=10000, multipv=4, temperature=0.15, threads=8):
    """Relabel fixed chunks durably; interruption retains verified completed work."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (type(nodes) is not int or nodes < 1 or type(multipv) is not int or multipv < 1
            or type(threads) is not int or threads < 1
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('invalid teacher settings')
    source_manifest = verify_dataset(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    num_workers = min(threads, 32)
    hash_per_worker = max(32, 1024 // num_workers)
    chunk_size = 32
    work = output.with_name(output.name + '.work')
    work.mkdir(exist_ok=True)
    request = {'format': 'inez-teacher-chunks-v1', 'label_contract': 2,
               'analysis_game': 'fresh-per-position',
               'label_sources': {name: sha256(Path(__file__).parent / name) for name in
                   ('curriculum.py', 'records.py', 'encoding.py', 'openings.py')},
               'source_manifest_sha256': sha256(source / 'manifest.json'),
               'source_splits': source_manifest['splits'], 'binary_sha256': sha256(stockfish),
               'nodes': nodes, 'multipv': multipv, 'temperature': temperature,
               'chunk_size': chunk_size, 'threads': threads, 'hash_per_worker': hash_per_worker}
    request_path = work / 'request.json'
    if request_path.exists() and read_json(request_path.read_text()) != request:
        raise ValueError('teacher preparation request changed; choose a new data directory')
    atomic_json(request_path, request)
    with ExitStack() as stack:
        engine_queue = queue.Queue()
        engine_name = None
        for _ in range(num_workers):
            eng = stack.enter_context(chess.engine.SimpleEngine.popen_uci(str(stockfish), timeout=120))
            if 'stockfish' not in eng.id.get('name', '').lower() or 'UCI_ShowWDL' not in eng.options:
                raise ValueError('teacher must be Stockfish with UCI_ShowWDL support')
            eng.configure({'Threads': 1, 'Hash': hash_per_worker, 'UCI_ShowWDL': True})
            if engine_name is None:
                engine_name = eng.id['name']
            engine_queue.put(eng)
        analysis_cache = {}
        cache_lock = threading.Lock()

        def process_chunk(item):
            split, index, records = item
            destination = work / split / (f'{index:07d}.jsonl')
            marker_path = destination.with_suffix('.complete.json')
            if destination.is_file() and marker_path.is_file():
                try:
                    marker = read_json(marker_path.read_text())
                    if (isinstance(marker, dict) and marker.get('records') == len(records)
                            and marker.get('sha256') == sha256(destination)):
                        return destination, len(records), True
                except (ValueError, OSError):
                    pass  # Uncommitted or damaged work is recomputed, never published.
            worker_engine = engine_queue.get()
            try:
                labeled_records = []
                for record in records:
                    board = validate_record(record)
                    cache_key = _teacher_cache_key(record, board)
                    with cache_lock:
                        cached = analysis_cache.get(cache_key)
                    if cached is not None:
                        labeled = copy.deepcopy(record)
                        labeled.update(copy.deepcopy(cached))
                        validate_record(labeled)
                    else:
                        labeled = stockfish_labels(record, worker_engine, nodes=nodes,
                                                  multipv=multipv, temperature=temperature)
                        snapshot = {key: labeled[key] for key in ('policy', 'wdl', 'label_kind', 'teacher')}
                        with cache_lock:
                            if len(analysis_cache) < 65536:
                                analysis_cache[cache_key] = snapshot
                    labeled_records.append(labeled)
                _write_records_atomic(destination, labeled_records)
                atomic_json(marker_path, {'records': len(records), 'sha256': sha256(destination)})
                return destination, len(records), False
            finally:
                engine_queue.put(worker_engine)

        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            staging = Path(temporary) / 'teacher'
            staging.mkdir()
            manifest = copy.deepcopy(source_manifest)
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for split in ('train', 'validation', 'evaluation'):
                    destination = staging / (split + '.jsonl')
                    completed, reused = 0, 0
                    index, records, pending = 0, [], []
                    with (source / (split + '.jsonl')).open() as source_stream, destination.open('wb') as target:
                        def finish_window():
                            nonlocal completed, reused
                            for chunk_path, count, cached in pool.map(process_chunk, pending):
                                with chunk_path.open('rb') as chunk_stream:
                                    for block in iter(lambda: chunk_stream.read(1024 * 1024), b''):
                                        target.write(block)
                                completed += count
                                reused += count if cached else 0
                            print(f'Stockfish {split}: {completed}/{source_manifest["splits"][split]["records"]}'
                                  f' ({reused} reused)', flush=True)
                            pending.clear()
                        for line in source_stream:
                            if not line.strip():
                                continue
                            records.append(read_json(line))
                            if len(records) == chunk_size:
                                pending.append((split, index, records))
                                index, records = index + 1, []
                            if len(pending) >= num_workers:
                                finish_window()
                        if records:
                            pending.append((split, index, records))
                        if pending:
                            finish_window()
                        target.flush()
                        os.fsync(target.fileno())
                    if completed != source_manifest['splits'][split]['records']:
                        raise ValueError('teacher source count changed during preparation')
                    manifest['splits'][split] = {'records': completed, 'sha256': sha256(destination)}
            if sha256(source / 'manifest.json') != request['source_manifest_sha256']:
                raise ValueError('teacher source manifest changed during preparation')
            verify_dataset(source)
            if sha256(stockfish) != request['binary_sha256']:
                raise ValueError('teacher binary changed during preparation')
            manifest['teacher'] = {'name': engine_name, 'binary_sha256': request['binary_sha256'],
                'nodes': nodes, 'multipv': multipv, 'temperature': temperature, 'threads': threads,
                'hash_mb': hash_per_worker * num_workers, 'uci_show_wdl': True,
                'policy_target': 'softmax((win-loss)/temperature) over MultiPV first moves',
                'value_target': 'native UCI WDL of best returned line',
                'human_manifest_sha256': sha256(source / 'manifest.json'),
                'label_contract': 2, 'analysis_game': request['analysis_game'],
                'resume_format': request['format']}
            atomic_json(staging / 'manifest.json', manifest)
            verify_dataset(staging)
            os.rename(staging, output)
    return manifest


def _soft_wdl_from_scalar(value: float) -> tuple[float, float, float]:
    """Convert scalar search value in [-1, 1] to soft WDL (Braun 2026 Appendix D.3)."""
    v = max(-1.0, min(1.0, float(value)))
    rem = 1.0 - abs(v)
    return (max(v, 0.0) + rem / 3.0, rem / 3.0, max(-v, 0.0) + rem / 3.0)


def selfplay_records(model, *, games, simulations, max_plies, seed, excluded,
                     initial_fen=chess.STARTING_FEN, stop=None,
                     progress_path=None, progress_identity=None):
    from .progress import GameProgress, content_digest, restore_random_state

    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (games, simulations, max_plies)):
        raise ValueError('self-play game, simulation and ply limits must be positive integers')
    progress = GameProgress(progress_path, {
        'kind': 'selfplay-v1', 'identity': progress_identity, 'games': games,
        'simulations': simulations, 'max_plies': max_plies, 'seed': seed,
        'initial_fen': initial_fen, 'excluded_sha256': content_digest(sorted(excluded))})
    if len(progress.chunks) > games:
        raise ValueError('too many saved self-play games')

    def discarded(board):
        outcome = board.outcome(claim_draw=False)
        if outcome is None:
            return True
        if outcome.winner is not None:
            return False
        material = [sum(len(board.pieces(pt, color)) * value
                        for pt, value in PIECE_VALUES.items() if pt != chess.KING)
                    for color in (chess.WHITE, chess.BLACK)]
        return abs(material[0] - material[1]) >= 5.0

    records, unfinished = [], 0
    rng = random.Random(seed)
    for saved in progress.chunks:
        if (not isinstance(saved, dict)
                or set(saved) != {'moves', 'result', 'discarded', 'records', 'rng_state'}
                or not isinstance(saved['moves'], list) or len(saved['moves']) > max_plies
                or not isinstance(saved['records'], list) or not isinstance(saved['discarded'], bool)):
            raise ValueError('invalid saved self-play game')
        board, target_plies = chess.Board(initial_fen), []
        try:
            for ply, move in enumerate(saved['moves']):
                if board.outcome(claim_draw=False) is not None:
                    raise ValueError('saved game continues after its outcome')
                if position_key(board) not in excluded:
                    target_plies.append(ply)
                board.push_uci(move)
        except (TypeError, ValueError) as error:
            raise ValueError('invalid saved self-play moves') from error
        outcome = board.outcome(claim_draw=False)
        if (saved['result'] != (outcome.result() if outcome else '*')
                or saved['discarded'] != discarded(board)
                or (outcome is None and len(saved['moves']) != max_plies)):
            raise ValueError('saved self-play outcome is inconsistent')
        identity = hashlib.sha256(' '.join(saved['moves']).encode()).hexdigest()
        if saved['discarded']:
            if saved['records']:
                raise ValueError('discarded self-play game contains training records')
            unfinished += 1
        else:
            for record in saved['records']:
                validate_record(record)
                if (record['moves'] != saved['moves'] or record['result'] != saved['result']
                        or record['game_id'] != identity or record['label_kind'] != 'search'
                        or record['initial_fen'] != chess.Board(initial_fen).fen(en_passant='fen')):
                    raise ValueError('saved self-play records disagree with their game')
            if [record['ply'] for record in saved['records']] != target_plies:
                raise ValueError('saved self-play targets are missing or reordered')
            records.extend(saved['records'])
        restore_random_state(rng, saved['rng_state'])

    evaluator = NeuralEvaluator(model)
    for game_index in range(len(progress.chunks), games):
        if stop is not None and stop.requested:
            raise InterruptedError('self-play generation stopped; completed games are saved')
        board = chess.Board(initial_fen)
        targets, moves = [], []
        while board.outcome(claim_draw=False) is None and len(moves) < max_plies:
            if stop is not None and stop.requested:
                raise InterruptedError('self-play generation stopped; completed games are saved')
            infos = []
            visits = search(board, evaluator, simulations, noise=True, rng=rng, on_update=infos.append)
            if position_key(board) not in excluded:
                total = sum(visits.values())
                root_val = infos[-1].value if infos else None
                targets.append((len(moves), {move: count / total for move, count in visits.items() if count},
                                root_val))
            temp = 0.85 if len(moves) < 10 else (0.25 if len(moves) < 18 else 0.0)
            move = choose_move(visits, temperature=temp, rng=rng)
            board.push_uci(move)
            moves.append(move)
        outcome = board.outcome(claim_draw=False)
        game_records = []
        is_discarded = discarded(board)
        if is_discarded:
            unfinished += 1  # A move cap or botched material win is not a draw target.
        else:
            identity = hashlib.sha256(' '.join(moves).encode()).hexdigest()
            total_plies = len(moves)
            for ply, policy, root_val in targets:
                record = make_record(chess.Board(initial_fen), moves, ply, outcome.result(), 'inez-selfplay',
                                     identity, policy, kind='search',
                                     teacher={'name': 'Inez PUCT', 'version': 'v1', 'limits': {'nodes': simulations}})
                remaining = total_plies - 1 - ply
                if remaining > 0 and root_val is not None and math.isfinite(root_val):
                    gamma = 0.998 ** remaining
                    disc_wdl = [gamma * w + (1.0 - gamma) / 3.0 for w in record['wdl']]
                    search_wdl = _soft_wdl_from_scalar(root_val)
                    blended = [0.50 * d + 0.50 * s for d, s in zip(disc_wdl, search_wdl)]
                    norm = sum(blended)
                    record['wdl'] = [b / norm for b in blended]
                validate_record(record)
                game_records.append(record)
        result = outcome.result() if outcome else '*'
        progress.append({'moves': moves, 'result': result, 'discarded': is_discarded,
                         'records': game_records, 'rng_state': rng.getstate()})
        records.extend(game_records)
        print(f'Self-play game {game_index + 1}/{games}: {result}, {len(moves)} plies', flush=True)
    return records, {'requested_games': games, 'unfinished_games_discarded': unfinished,
                     'completed_games': games - unfinished, 'records': len(records)}


def tactical_report(model, tactics, simulations):
    evaluate = NeuralEvaluator(model)
    results = []
    for item in tactics:
        board = chess.Board(item['fen'])
        visits = search(board, evaluate, simulations)
        move = choose_move(visits)
        solved = move in item['best_moves']
        if not solved:
            # Accept any alternative move that also delivers immediate checkmate when the
            # expected solution is a 1-ply mate.
            ref = board.copy(stack=False)
            ref.push_uci(item['best_moves'][0])
            if ref.is_checkmate():
                cand = board.copy(stack=False)
                cand.push_uci(move)
                solved = cand.is_checkmate()
        results.append({'name': item['name'], 'move': move, 'solved': solved})
    return results


def _validate_pair_settings(alpha, expected_pairs):
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError('paired gate alpha must be finite and between 0 and 0.5')
    if expected_pairs is not None and (isinstance(expected_pairs, bool)
            or not isinstance(expected_pairs, int) or expected_pairs < 1):
        raise ValueError('expected gate pairs must be a positive integer')


def _validate_minimum_score(minimum_score):
    if (isinstance(minimum_score, bool) or not math.isfinite(minimum_score)
            or not 0.5 < minimum_score <= 1.0):
        raise ValueError('minimum promotion score must be greater than 0.5 and at most 1')


def paired_score_test(games, *, expected_pairs=None, alpha=0.05):
    """Exact, fixed-sample, one-sided randomization test at the opening-pair level.

    Under the null, swapping the engine identities within each independently
    sampled opening pair is exchangeable. We condition on the absolute pair
    score difference and enumerate its signs exactly by dynamic programming.
    Both color-reversed games stay together; draws and tied pairs are retained
    in the reported pentanomial counts. This is not an Elo confidence interval,
    an arbitrary-mean test, or a sequential test. The opening sample and sample
    size must be fixed before play; repeated candidates need an alpha budget.
    """
    _validate_pair_settings(alpha, expected_pairs)
    result = {'method': 'exact-paired-sign-flip-v1', 'alpha': alpha,
              'p_value': None, 'completed_pairs': 0, 'pentanomial': [0] * 5,
              'valid': False, 'rejection_reasons': []}
    reasons = result['rejection_reasons']
    if not games:
        reasons.append('missing_paired_results')
        return result
    groups = {}
    for game in games:
        pair = game.get('pair')
        white = game.get('candidate_white')
        score = game.get('candidate_score')
        if (isinstance(pair, bool) or not isinstance(pair, int) or pair < 0
                or not isinstance(white, bool)):
            reasons.append('invalid_pair_structure')
            return result
        if score is not None and (isinstance(score, bool) or score not in (0.0, 0.5, 1.0)):
            raise ValueError('candidate game scores must be 0, 0.5, 1, or None')
        if game.get('fault') or game.get('faults'):
            reasons.append('match_game_faults')
        if score is None:
            reasons.append('unfinished_match_games')
        else:
            expected_result = ('1/2-1/2' if score == 0.5 else
                               '1-0' if (score == 1.0) == white else '0-1')
            if game.get('result') != expected_result:
                reasons.append('inconsistent_game_result')
        groups.setdefault(pair, []).append(game)
    if expected_pairs is not None and set(groups) != set(range(expected_pairs)):
        reasons.append('incomplete_opening_pairs')
    positions, differences = set(), []
    for legs in groups.values():
        if len(legs) != 2 or {game['candidate_white'] for game in legs} != {False, True}:
            reasons.append('incomplete_opening_pairs')
            continue
        keys = {game.get('opening_position') for game in legs}
        if len(keys) != 1 or not all(isinstance(key, str) and key for key in keys):
            reasons.append('missing_or_inconsistent_opening_position')
            continue
        key = next(iter(keys))
        if key in positions:
            reasons.append('duplicate_opening_positions')
        positions.add(key)
        if any(game['candidate_score'] is None for game in legs):
            continue
        pair_score = sum(game['candidate_score'] for game in legs)
        result['pentanomial'][int(pair_score * 2)] += 1
        differences.append(int(pair_score * 2) - 2)
    result['completed_pairs'] = len(differences)
    result['rejection_reasons'] = list(dict.fromkeys(reasons))
    if result['rejection_reasons']:
        return result
    distribution = {0: 1}
    nonzero = [abs(difference) for difference in differences if difference]
    for magnitude in nonzero:
        next_distribution = {}
        for statistic, count in distribution.items():
            for signed in (-magnitude, magnitude):
                target = statistic + signed
                next_distribution[target] = next_distribution.get(target, 0) + count
        distribution = next_distribution
    observed = sum(differences)
    extreme = sum(count for statistic, count in distribution.items() if statistic >= observed)
    result.update(valid=True, p_value=extreme / (1 << len(nonzero)),
                  nonzero_pairs=len(nonzero), observed_statistic=observed)
    return result


def promotion_decision(points, games, candidate_tactics, previous_tactics, *,
                       minimum_score=0.55, match_games=None, alpha=0.05,
                       expected_pairs=None):
    """Require complete paired evidence, a practical score gain, and no tactical loss.

    Aggregate-only reports from older runs remain readable by status tools, but
    cannot authorize a new promotion because their independent pairs are unknown.
    """
    if (isinstance(games, bool) or not isinstance(games, int) or games < 0
            or isinstance(points, bool) or not math.isfinite(points)
            or not 0 <= points <= games):
        raise ValueError('invalid completed-game count or match points')
    _validate_minimum_score(minimum_score)
    names = [item['name'] for item in previous_tactics]
    if (names != [item['name'] for item in candidate_tactics]
            or len(set(names)) != len(names)):
        raise ValueError('tactical gate suites differ or contain duplicate names')
    if any(not isinstance(item['solved'], bool) for item in previous_tactics + candidate_tactics):
        raise ValueError('tactical outcomes must be booleans')
    regressions = [old['name'] for old, new in zip(previous_tactics, candidate_tactics)
                   if old['solved'] and not new['solved']]
    paired = paired_score_test(match_games, expected_pairs=expected_pairs, alpha=alpha)
    reasons = list(paired['rejection_reasons'])
    score = points / games if games else None
    if match_games is not None:
        completed = [game['candidate_score'] for game in match_games
                     if game.get('candidate_score') is not None]
        if len(completed) != games or not math.isclose(sum(completed), points, abs_tol=1e-12):
            raise ValueError('aggregate match score disagrees with paired game records')
    if games == 0:
        reasons.append('no_completed_games')
    if score is not None and score < minimum_score:
        reasons.append('score_below_minimum')
    if paired['valid'] and paired['p_value'] > alpha:
        reasons.append('insufficient_paired_evidence')
    if not names:
        reasons.append('missing_tactical_suite')
    if regressions:
        reasons.append('tactical_regression')
    return {'passed': not reasons, 'score': score, 'score_lower_bound': None,
            'tactical_regressions': regressions, 'paired_test': paired,
            'rejection_reasons': list(dict.fromkeys(reasons))}


def evaluate_candidate(candidate, incumbent, *, tactics, openings, pairs, simulations,
                       max_plies, minimum_score=0.55, alpha=0.05, stop=None,
                       progress_path=None, progress_identity=None):
    from .progress import GameProgress

    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (pairs, simulations, max_plies)) or not openings
            or pairs > len(openings)):
        raise ValueError('invalid gate limits or insufficient distinct openings')
    # Select every opening before play. Transposed move sequences are the same
    # experiment, so compare positions rather than only the spelling of moves.
    selected, positions = [], set()
    for opening in openings[:pairs]:
        board = chess.Board()
        for move in opening:
            board.push_uci(move)
        key = position_key(board)
        if key in positions or board.outcome(claim_draw=False) is not None:
            raise ValueError('gate requires distinct nonterminal opening positions')
        positions.add(key)
        selected.append((tuple(opening), key))
    _validate_pair_settings(alpha, pairs)
    _validate_minimum_score(minimum_score)
    progress = GameProgress(progress_path, {
        'kind': 'promotion-v1', 'identity': progress_identity, 'tactics': tactics,
        'openings': openings, 'pairs': pairs, 'simulations': simulations,
        'max_plies': max_plies, 'minimum_score': minimum_score, 'alpha': alpha})
    if len(progress.chunks) > pairs * 2:
        raise ValueError('too many saved promotion games')
    games, points = [], 0.0
    for index, game in enumerate(progress.chunks):
        opening, opening_position = selected[index // 2]
        candidate_white = index % 2 == 0
        if (not isinstance(game, dict) or set(game) != {'pair', 'candidate_white',
                'opening_position', 'opening', 'result', 'candidate_score', 'moves'}
                or game['pair'] != index // 2 or game['candidate_white'] is not candidate_white
                or game['opening_position'] != opening_position or game['opening'] != list(opening)
                or not isinstance(game['moves'], list)
                or game['moves'][:len(opening)] != list(opening)
                or not len(opening) <= len(game['moves']) <= len(opening) + max_plies):
            raise ValueError('invalid saved promotion game metadata')
        board = chess.Board()
        try:
            for move in game['moves']:
                if board.outcome(claim_draw=False) is not None:
                    raise ValueError('saved game continues after its outcome')
                board.push_uci(move)
        except (TypeError, ValueError) as error:
            raise ValueError('invalid saved promotion moves') from error
        outcome = board.outcome(claim_draw=False)
        score = None if outcome is None else (0.5 if outcome.winner is None else float(outcome.winner == candidate_white))
        if (game['result'] != (outcome.result() if outcome else '*')
                or isinstance(game['candidate_score'], bool) or game['candidate_score'] != score
                or (outcome is None and len(game['moves']) != len(opening) + max_plies)):
            raise ValueError('saved promotion outcome is inconsistent')
        games.append(game)
        if score is not None:
            points += score
    candidate_eval, incumbent_eval = NeuralEvaluator(candidate), NeuralEvaluator(incumbent)
    for index in range(len(games), pairs * 2):
        if stop is not None and stop.requested:
            raise InterruptedError('candidate gate stopped; completed games are saved')
        pair, candidate_white = index // 2, index % 2 == 0
        opening, opening_position = selected[pair]
        board = chess.Board()
        for move in opening:
            board.push_uci(move)
        plies = 0
        while board.outcome(claim_draw=False) is None and plies < max_plies:
            if stop is not None and stop.requested:
                raise InterruptedError('candidate gate stopped; completed games are saved')
            evaluate = candidate_eval if board.turn == candidate_white else incumbent_eval
            board.push_uci(choose_move(search(board, evaluate, simulations)))
            plies += 1
        outcome = board.outcome(claim_draw=False)
        # Move caps are unfinished games, never fabricated draws.
        score = None if outcome is None else (0.5 if outcome.winner is None else float(outcome.winner == candidate_white))
        game = {'pair': pair, 'candidate_white': candidate_white,
                'opening_position': opening_position, 'opening': list(opening),
                'result': outcome.result() if outcome else '*', 'candidate_score': score,
                'moves': [move.uci() for move in board.move_stack]}
        progress.append(game)
        games.append(game)
        if score is not None:
            points += score
        print(f'Promotion game {index + 1}/{pairs * 2}: {game["result"]}, {plies} plies', flush=True)
    if stop is not None and stop.requested:
        raise InterruptedError('candidate gate stopped; completed games are saved')
    finished = sum(game['candidate_score'] is not None for game in games)
    candidate_tactics = tactical_report(candidate, tactics, simulations)
    incumbent_tactics = tactical_report(incumbent, tactics, simulations)
    decision = promotion_decision(points, finished, candidate_tactics, incumbent_tactics,
                                  minimum_score=minimum_score, match_games=games,
                                  alpha=alpha, expected_pairs=pairs)
    return {**decision, 'games': games, 'completed_games': finished,
            'candidate_tactics': candidate_tactics, 'incumbent_tactics': incumbent_tactics,
            'simulations_per_move': simulations, 'pairs': pairs}
