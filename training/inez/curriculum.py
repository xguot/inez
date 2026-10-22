# SPDX-License-Identifier: GPL-3.0-only
"""Game-safe human initialization, Stockfish relabeling and independent gates."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import random
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
                      validate_record, write_dataset)


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


def import_games(path, output, *, uri, license_name, tactics, max_games=1000,
                 samples_per_game=16, min_ply=8, min_rating=2300, seed=20261019,
                 openings_path=None, blunder_samples_per_game=0):
    """Import a bounded corpus; remove held-out tactical boards and split overlap."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if max_games < 1 or samples_per_game < 1 or min_ply < 0 or blunder_samples_per_game < 0:
        raise ValueError('invalid importer bounds')
    paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
    uris = list(uri) if isinstance(uri, (list, tuple)) else [uri] * len(paths)
    if not paths or len(paths) != len(uris):
        raise ValueError('archive paths and URIs must be nonempty and aligned')

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
                if (game.errors or initial.chess960 or not initial.is_valid()
                        or game.headers.get('Variant', 'Standard') != 'Standard'
                        or min(ratings) < min_rating):
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
        manifest = write_dataset([raw], output, sources, seed=str(seed), require_disjoint=True)
    manifest['preparation'] = {**stats, 'min_rating': min_rating, 'min_ply': min_ply,
                               'samples_per_game': samples_per_game, 'max_games': max_games,
                               'blunder_samples_per_game': blunder_samples_per_game,
                               'openings_sha256': sha256(openings_path) if openings_path and Path(openings_path).is_file() else None,
                               'tactics_sha256': sha256(tactics), 'labels': 'human_moves_and_game_result'}
    atomic_json(output / 'manifest.json', manifest)
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
    infos = engine.analyse(board, chess.engine.Limit(nodes=nodes),
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


def relabel_dataset(source, output, stockfish, *, nodes=10000, multipv=4, temperature=0.15, threads=8):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((source / 'manifest.json').read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    num_workers = max(1, min(int(threads), 16))
    hash_per_worker = max(32, 512 // num_workers)
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
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            staging = Path(temporary) / 'teacher'
            staging.mkdir()
            manifest = copy.deepcopy(source_manifest)
            for split in ('train', 'validation', 'evaluation'):
                records = load_records([source / (split + '.jsonl')])
                destination = staging / (split + '.jsonl')
                chunk_size = max(1, min(32, (len(records) + num_workers - 1) // num_workers))
                chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

                def process_batch(chunk):
                    worker_engine = engine_queue.get()
                    try:
                        out = []
                        for rec in chunk:
                            board = validate_record(rec)
                            cache_key = (position_key(board), str(rec.get('source_id', '')).startswith('mco-'))
                            with cache_lock:
                                cached = analysis_cache.get(cache_key)
                            if cached is not None:
                                labeled = copy.deepcopy(rec)
                                labeled.update(cached)
                                validate_record(labeled)
                                out.append(labeled)
                            else:
                                labeled = stockfish_labels(
                                    rec, worker_engine, nodes=nodes, multipv=multipv, temperature=temperature
                                )
                                snapshot = {
                                    'policy': labeled['policy'],
                                    'wdl': labeled['wdl'],
                                    'label_kind': labeled['label_kind'],
                                    'teacher': labeled['teacher'],
                                }
                                with cache_lock:
                                    if len(analysis_cache) < 200000:
                                        analysis_cache[cache_key] = snapshot
                                out.append(labeled)
                        return out
                    finally:
                        engine_queue.put(worker_engine)

                completed = 0
                with destination.open('w') as stream, ThreadPoolExecutor(max_workers=num_workers) as pool:
                    for labeled_chunk in pool.map(process_batch, chunks):
                        for labeled in labeled_chunk:
                            stream.write(json.dumps(labeled) + '\n')
                            completed += 1
                            if completed % 200 == 0 or completed == len(records):
                                print(f'Stockfish {split}: {completed}/{len(records)}', flush=True)
                manifest['splits'][split] = {'records': len(records), 'sha256': sha256(destination)}
            manifest['teacher'] = {'name': engine_name, 'binary_sha256': sha256(stockfish),
                'nodes': nodes, 'multipv': multipv, 'temperature': temperature, 'threads': threads,
                'hash_mb': hash_per_worker * num_workers, 'uci_show_wdl': True,
                'policy_target': 'softmax((win-loss)/temperature) over MultiPV first moves',
                'value_target': 'native UCI WDL of best returned line',
                'human_manifest_sha256': sha256(source / 'manifest.json')}
            atomic_json(staging / 'manifest.json', manifest)
            os.rename(staging, output)
    return manifest


def _soft_wdl_from_scalar(value: float) -> tuple[float, float, float]:
    """Convert scalar search value in [-1, 1] to soft WDL (Braun 2026 Appendix D.3)."""
    v = max(-1.0, min(1.0, float(value)))
    rem = 1.0 - abs(v)
    return (max(v, 0.0) + rem / 3.0, rem / 3.0, max(-v, 0.0) + rem / 3.0)


def selfplay_records(model, *, games, simulations, max_plies, seed, excluded,
                     initial_fen=chess.STARTING_FEN, stop=None):
    evaluator = NeuralEvaluator(model)
    rng = random.Random(seed)
    records, unfinished = [], 0
    for game_index in range(games):
        board = chess.Board(initial_fen)
        targets, moves = [], []
        while board.outcome(claim_draw=False) is None and len(moves) < max_plies:
            if stop is not None and stop.requested:
                raise InterruptedError('self-play generation stopped; resume from the saved curriculum')
            infos = []
            visits = search(board, evaluator, simulations, noise=True, rng=rng, on_update=infos.append)
            if position_key(board) not in excluded:
                total = sum(visits.values())
                root_val = infos[-1].value if infos else None
                targets.append((len(moves), {move: count / total for move, count in visits.items() if count},
                                root_val))
            move = choose_move(visits, temperature=1.0 if len(moves) < 20 else 0.0, rng=rng)
            board.push_uci(move)
            moves.append(move)
        outcome = board.outcome(claim_draw=False)
        if outcome is None:
            unfinished += 1
            continue  # A move cap is not evidence that a position is drawn.
        if outcome.winner is None:
            white_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in PIECE_VALUES.items() if pt != chess.KING)
            black_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in PIECE_VALUES.items() if pt != chess.KING)
            if abs(white_mat - black_mat) >= 5.0:
                unfinished += 1
                continue  # Never label a botched winning material endgame as a draw target.
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
                blended = [0.90 * d + 0.10 * s for d, s in zip(disc_wdl, search_wdl)]
                norm = sum(blended)
                record['wdl'] = [b / norm for b in blended]
            validate_record(record)
            records.append(record)
        print(f'Self-play game {game_index + 1}/{games}: {outcome.result()}, {len(moves)} plies', flush=True)
    return records, {'requested_games': games, 'unfinished_games_discarded': unfinished,
                     'completed_games': games - unfinished, 'records': len(records)}


def tactical_report(model, tactics, simulations):
    evaluate = NeuralEvaluator(model)
    results = []
    for item in tactics:
        visits = search(chess.Board(item['fen']), evaluate, simulations)
        move = choose_move(visits)
        results.append({'name': item['name'], 'move': move, 'solved': move in item['best_moves']})
    return results


def wilson_lower(points, games, z=1.96):
    if games <= 0:
        return 0.0
    p = points / games
    return (p + z*z/(2*games) - z*math.sqrt(p*(1-p)/games + z*z/(4*games*games))) / (1+z*z/games)


def promotion_decision(points, games, candidate_tactics, previous_tactics, *, minimum_score=0.55):
    regressions = [old['name'] for old, new in zip(previous_tactics, candidate_tactics)
                   if old['name'] != new['name'] or old['solved'] and not new['solved']]
    if len(candidate_tactics) != len(previous_tactics):
        raise ValueError('tactical gate suites differ')
    lower = wilson_lower(points, games)
    # Fractional draw scores use a conservative bounded-score Wilson-style gate;
    # this is promotion evidence, not a claim of an exact Elo confidence interval.
    passed = games > 0 and points/games >= minimum_score and lower > 0.5 and not regressions
    return {'passed': passed, 'score': points/games if games else None,
            'score_lower_bound': lower, 'tactical_regressions': regressions}


def evaluate_candidate(candidate, incumbent, *, tactics, openings, pairs, simulations, max_plies, minimum_score=0.55, stop=None):
    if pairs < 1 or simulations < 1 or max_plies < 1:
        raise ValueError('invalid gate limits')
    candidate_eval, incumbent_eval = NeuralEvaluator(candidate), NeuralEvaluator(incumbent)
    games, points = [], 0.0
    for pair in range(pairs):
        opening = openings[pair % len(openings)]
        for candidate_white in (True, False):
            board = chess.Board()
            for move in opening:
                board.push_uci(move)
            plies = 0
            while board.outcome(claim_draw=False) is None and plies < max_plies:
                if stop is not None and stop.requested:
                    raise InterruptedError('candidate gate stopped; training checkpoint is saved')
                evaluate = candidate_eval if board.turn == candidate_white else incumbent_eval
                board.push_uci(choose_move(search(board, evaluate, simulations)))
                plies += 1
            outcome = board.outcome(claim_draw=False)
            # Truncated games carry no evidence for promotion. Record them but
            # never fabricate a decisive result or a training draw label.
            score = None if outcome is None else (0.5 if outcome.winner is None else float(outcome.winner == candidate_white))
            games.append({'pair': pair, 'candidate_white': candidate_white,
                          'result': outcome.result() if outcome else '*', 'candidate_score': score,
                          'moves': [move.uci() for move in board.move_stack]})
            if score is not None:
                points += score
    finished = sum(game['candidate_score'] is not None for game in games)
    candidate_tactics = tactical_report(candidate, tactics, simulations)
    incumbent_tactics = tactical_report(incumbent, tactics, simulations)
    decision = promotion_decision(points, finished, candidate_tactics, incumbent_tactics,
                                  minimum_score=minimum_score)
    if finished != len(games):
        decision['passed'] = False
    return {**decision, 'games': games, 'completed_games': finished,
            'candidate_tactics': candidate_tactics, 'incumbent_tactics': incumbent_tactics,
            'simulations_per_move': simulations, 'pairs': pairs}
