# SPDX-License-Identifier: GPL-3.0-only
"""Game-safe human initialization, Stockfish relabeling and independent gates."""

from contextlib import contextmanager
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import zipfile

import chess
import chess.engine
import chess.pgn

from .encoding import wdl_from_result
from .puct import NeuralEvaluator, choose_move, search
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


def import_games(path, output, *, uri, license_name, tactics, max_games=1000,
                 samples_per_game=16, min_ply=8, min_rating=2300, seed=20261019):
    """Import a bounded corpus; remove held-out tactical boards and split overlap."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if max_games < 1 or samples_per_game < 1 or min_ply < 0:
        raise ValueError('invalid importer bounds')
    excluded = {position_key(chess.Board(item['fen'])) for item in read_tactics(tactics)}
    digest = sha256(path)
    source = 'elite-' + digest[:16]
    rng = random.Random(seed)
    records, seen = [], set()
    stats = {'read_games': 0, 'accepted_games': 0, 'invalid_or_filtered_games': 0}
    with pgn_stream(path) as stream:
        while stats['accepted_games'] < max_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
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
                if len(moves) <= min_ply:
                    raise ValueError('short game')
                identity = hashlib.sha256((initial.fen() + ' '.join(moves)).encode()).hexdigest()
                if identity in seen:
                    continue
                seen.add(identity)
                candidates = rng.sample(range(min_ply, len(moves)), min(samples_per_game, len(moves) - min_ply))
                accepted = []
                for ply in sorted(candidates):
                    record = make_record(initial, moves, ply, result, source, identity, {moves[ply]: 1.0})
                    board = validate_record(record)
                    if position_key(board) not in excluded:
                        accepted.append(record)
                if not accepted:
                    continue
                records.extend(accepted)
                stats['accepted_games'] += 1
            except (ValueError, TypeError, OverflowError):
                stats['invalid_or_filtered_games'] += 1
    if not records:
        raise ValueError('no games survived import filters')
    splits = assign_splits(records, seed=str(seed))
    # Reserve held-out board identities before training, even for transpositions
    # from otherwise distinct games. Evaluation has priority over validation.
    reserved, filtered = set(excluded), {}
    for split in ('evaluation', 'validation', 'train'):
        items = [r for r in splits[split] if position_key(validate_record(r)) not in reserved]
        filtered[split] = items
        reserved.update(position_key(validate_record(r)) for r in items)
    records = [r for split in ('train', 'validation', 'evaluation') for r in filtered[split]]
    if any(not filtered[split] for split in filtered):
        raise ValueError('need more imported games to populate all three independent splits')
    sources = {source: {'uri': uri, 'license': license_name, 'sha256': digest}}
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        raw = Path(temporary) / 'records.jsonl'
        raw.write_text(''.join(json.dumps(r) + '\n' for r in records))
        manifest = write_dataset([raw], output, sources, seed=str(seed), require_disjoint=True)
    manifest['preparation'] = {**stats, 'min_rating': min_rating, 'min_ply': min_ply,
                               'samples_per_game': samples_per_game, 'max_games': max_games,
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
    infos = engine.analyse(board, chess.engine.Limit(nodes=nodes),
                           multipv=min(multipv, board.legal_moves.count()))
    if not isinstance(infos, list):
        infos = [infos]
    if not infos or any('wdl' not in info or not info.get('pv') for info in infos):
        raise ValueError('Stockfish must provide UCI_ShowWDL and principal variations')
    outcomes = [info['wdl'].pov(board.turn) for info in infos]
    values = [(wdl.wins - wdl.losses) / 1000 for wdl in outcomes]
    weights = [math.exp((value - max(values)) / temperature) for value in values]
    policy = {info['pv'][0].uci(): weight / sum(weights) for info, weight in zip(infos, weights)}
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
    with chess.engine.SimpleEngine.popen_uci(str(stockfish), timeout=120) as engine:
        if 'stockfish' not in engine.id.get('name', '').lower() or 'UCI_ShowWDL' not in engine.options:
            raise ValueError('teacher must be Stockfish with UCI_ShowWDL support')
        engine.configure({'Threads': threads, 'Hash': 256, 'UCI_ShowWDL': True})
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            staging = Path(temporary) / 'teacher'
            staging.mkdir()
            manifest = copy.deepcopy(source_manifest)
            for split in ('train', 'validation', 'evaluation'):
                records = load_records([source / (split + '.jsonl')])
                destination = staging / (split + '.jsonl')
                with destination.open('w') as stream:
                    for number, record in enumerate(records, 1):
                        labeled = stockfish_labels(record, engine, nodes=nodes, multipv=multipv, temperature=temperature)
                        stream.write(json.dumps(labeled) + '\n')
                        if number % 100 == 0:
                            print(f'Stockfish {split}: {number}/{len(records)}', flush=True)
                manifest['splits'][split] = {'records': len(records), 'sha256': sha256(destination)}
            manifest['teacher'] = {'name': engine.id['name'], 'binary_sha256': sha256(stockfish),
                'nodes': nodes, 'multipv': multipv, 'temperature': temperature, 'threads': threads,
                'hash_mb': 256, 'uci_show_wdl': True,
                'policy_target': 'softmax((win-loss)/temperature) over MultiPV first moves',
                'value_target': 'native UCI WDL of best returned line',
                'human_manifest_sha256': sha256(source / 'manifest.json')}
            atomic_json(staging / 'manifest.json', manifest)
            os.rename(staging, output)
    return manifest


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
            visits = search(board, evaluator, simulations, noise=True, rng=rng)
            if position_key(board) not in excluded:
                total = sum(visits.values())
                targets.append((len(moves), {move: count / total for move, count in visits.items() if count}))
            move = choose_move(visits, temperature=1.0 if len(moves) < 20 else 0.0, rng=rng)
            board.push_uci(move)
            moves.append(move)
        outcome = board.outcome(claim_draw=False)
        if outcome is None:
            unfinished += 1
            continue  # A move cap is not evidence that a position is drawn.
        identity = hashlib.sha256(' '.join(moves).encode()).hexdigest()
        for ply, policy in targets:
            record = make_record(chess.Board(initial_fen), moves, ply, outcome.result(), 'inez-selfplay',
                                 identity, policy, kind='search',
                                 teacher={'name': 'Inez PUCT', 'version': 'v1', 'limits': {'nodes': simulations}})
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
