#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Preparation integrity, teacher history isolation, and durable chunk recovery."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import chess
from inez.curriculum import (_teacher_cache_key, atomic_json, relabel_dataset,
                             sha256, verify_dataset)
from inez.records import validate_record

spec = importlib.util.spec_from_file_location('preparation_cli', ROOT / 'tools/train_inezita.py')
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def record(index):
    return {'record_version': 1, 'encoding_version': 1, 'source_id': 'fixture',
            'game_id': str(index), 'initial_fen': chess.STARTING_FEN.rsplit(' ', 1)[0] + f' {index + 1}',
            'moves': ['e2e4', 'e7e5'], 'ply': 0, 'result': '1-0', 'policy': {'e2e4': 1.0},
            'wdl': [1.0, 0.0, 0.0], 'label_kind': 'game_result', 'teacher': None}


def labeled(record, engine, **kwargs):
    result = copy.deepcopy(record)
    result.update(label_kind='teacher_wdl', wdl=[0.8, 0.1, 0.1],
                  teacher={'name': 'Stockfish', 'version': 'Stockfish fixture',
                           'limits': {'nodes': kwargs['nodes']}})
    validate_record(result)
    return result


class FakeEngine:
    id = {'name': 'Stockfish fixture'}
    options = {'UCI_ShowWDL': object()}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def configure(self, options):
        pass


class PreparationTests(unittest.TestCase):
    def fixture(self, root, counts=(64, 1, 1)):
        source = root / 'human'
        source.mkdir()
        manifest = {'manifest_version': 1, 'record_version': 1, 'encoding_version': 1,
                    'audit': {'cross_split_positions': 0, 'duplicate_samples': 0}, 'splits': {}}
        index = 0
        for split, count in zip(('train', 'validation', 'evaluation'), counts):
            path = source / (split + '.jsonl')
            path.write_text(''.join(json.dumps(record(i)) + '\n' for i in range(index, index + count)))
            index += count
            manifest['splits'][split] = {'records': count, 'sha256': sha256(path)}
        atomic_json(source / 'manifest.json', manifest)
        binary = root / 'stockfish'
        binary.write_bytes(b'fixture binary')
        return source, binary

    def test_cached_archives_require_intact_zip_and_matching_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'games.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('games.pgn', '[Event "Fixture"]\n*\n')
            with patch.object(cli.urllib.request, 'urlopen') as network:
                cli.download_archive(archive, 'fixture://archive')
                cli.download_archive(archive, 'fixture://archive')
                network.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'provenance'):
                cli.download_archive(archive, 'fixture://different-archive')
            archive.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                cli.download_archive(archive, 'fixture://archive')

    def test_config_rejects_misaligned_archives_missing_openings_and_invalid_bounds(self):
        base = json.loads((ROOT / 'configs/train/inezita-0.1.3.json').read_text())
        changes = [('archives', ['one']), ('max_games', 0), ('teacher_nodes', True),
                   ('teacher_temperature', float('nan')), ('openings', '/missing/openings.tsv'),
                   ('min_time_control_seconds', -1), ('min_time_control_seconds', True),
                   ('require_normal_termination', 'yes')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for key, value in changes:
                config = copy.deepcopy(base)
                config['preparation'][key] = value
                path.write_text(json.dumps(config))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    cli.read_config(path)

    def test_dataset_rejects_incomplete_corrupt_and_miscounted_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.fixture(root)
            verify_dataset(source, full=True)
            manifest_path = source / 'manifest.json'
            original = manifest_path.read_text()
            manifest = json.loads(original)
            manifest['splits']['train']['records'] += 1
            atomic_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'record count'):
                verify_dataset(source)
            manifest_path.write_text(original)
            (source / 'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_dataset(source)
            manifest_path.unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                verify_dataset(source)

    def test_teacher_cache_preserves_counters_repetition_and_mco_policy(self):
        first = record(0)
        second = record(1)
        self.assertNotEqual(_teacher_cache_key(first, validate_record(first)),
                            _teacher_cache_key(second, validate_record(second)))
        repeated = record(0)
        repeated['moves'] = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
        repeated['ply'] = 4
        fresh = copy.deepcopy(repeated)
        fresh['initial_fen'] = validate_record(repeated).fen(en_passant='fen')
        fresh['moves'], fresh['ply'] = [], 0
        self.assertEqual(validate_record(repeated).fen(), validate_record(fresh).fen())
        self.assertNotEqual(_teacher_cache_key(repeated, validate_record(repeated)),
                            _teacher_cache_key(fresh, validate_record(fresh)))
        first['source_id'] = second['source_id'] = 'mco-fixture'
        second['initial_fen'] = first['initial_fen']
        second['policy'] = {'d2d4': 1.0}
        self.assertNotEqual(_teacher_cache_key(first, validate_record(first)),
                            _teacher_cache_key(second, validate_record(second)))

    def test_interrupted_teacher_reuses_completed_chunks_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary = self.fixture(root)
            output = root / 'teacher'
            calls = []
            def interrupt(record, engine, **kwargs):
                calls.append(record['game_id'])
                if len(calls) == 33:
                    raise RuntimeError('interrupted')
                return labeled(record, engine, **kwargs)
            with patch('inez.curriculum.chess.engine.SimpleEngine.popen_uci', return_value=FakeEngine()), \
                 patch('inez.curriculum.stockfish_labels', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    relabel_dataset(source, output, binary, threads=1)
            self.assertFalse(output.exists())
            self.assertTrue((root / 'teacher.work/train/0000000.complete.json').is_file())
            resumed = []
            def resume(record, engine, **kwargs):
                resumed.append(record['game_id'])
                return labeled(record, engine, **kwargs)
            with patch('inez.curriculum.chess.engine.SimpleEngine.popen_uci', return_value=FakeEngine()), \
                 patch('inez.curriculum.stockfish_labels', side_effect=resume):
                relabel_dataset(source, output, binary, threads=1)
            self.assertEqual(len(resumed), 34)
            self.assertTrue(set(resumed).isdisjoint({str(i) for i in range(32)}))
            verify_dataset(output, full=True)

    def test_corrupt_chunk_is_recomputed_and_changed_request_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary = self.fixture(root)
            output = root / 'teacher'
            calls = []
            def interrupt(record, engine, **kwargs):
                calls.append(record['game_id'])
                if len(calls) == 33:
                    raise RuntimeError('interrupted')
                return labeled(record, engine, **kwargs)
            with patch('inez.curriculum.chess.engine.SimpleEngine.popen_uci', return_value=FakeEngine()), \
                 patch('inez.curriculum.stockfish_labels', side_effect=interrupt):
                with self.assertRaises(RuntimeError):
                    relabel_dataset(source, output, binary, threads=1)
            with self.assertRaisesRegex(ValueError, 'request changed'):
                relabel_dataset(source, output, binary, threads=1, nodes=50000)
            (root / 'teacher.work/train/0000000.jsonl').write_text('corrupt\n')
            with patch('inez.curriculum.chess.engine.SimpleEngine.popen_uci', return_value=FakeEngine()), \
                 patch('inez.curriculum.stockfish_labels', side_effect=labeled) as labels:
                relabel_dataset(source, output, binary, threads=1)
                self.assertEqual(labels.call_count, 66)
            verify_dataset(output, full=True)

    def test_prepared_data_matches_budgets_hashes_but_allows_new_training_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary = self.fixture(root, counts=(1, 1, 1))
            config = json.loads((ROOT / 'configs/train/inezita-0.1.3.json').read_text())
            config['preparation']['teacher_nodes'] = 10000
            human = json.loads((source / 'manifest.json').read_text())
            settings = config['preparation']
            human['preparation'] = {key: settings[key] for key in
                ('min_rating', 'min_ply', 'samples_per_game', 'max_games', 'blunder_samples_per_game',
                 'min_time_control_seconds', 'require_normal_termination')}
            human['preparation'].update(tactics_sha256=sha256(cli.rooted(config['tactics'])),
                openings_sha256=sha256(cli.rooted(settings['openings'])))
            atomic_json(source / 'manifest.json', human)
            request = {'preparation': settings, 'seed': config['seed'] - 1,
                       'teacher_binary_sha256': sha256(binary),
                       'tactics_sha256': human['preparation']['tactics_sha256'],
                       'openings_sha256': human['preparation']['openings_sha256']}
            atomic_json(root / 'request.json', request)
            with patch('inez.curriculum.chess.engine.SimpleEngine.popen_uci', return_value=FakeEngine()), \
                 patch('inez.curriculum.stockfish_labels', side_effect=labeled):
                relabel_dataset(source, root / 'teacher', binary, threads=1)
            identity = cli.verify_prepared_data(root, config, full=True)
            self.assertEqual(identity['human_manifest_sha256'], sha256(source / 'manifest.json'))
            config['seed'] += 100
            cli.verify_prepared_data(root, config)
            config['preparation']['teacher_nodes'] += 1
            with self.assertRaisesRegex(ValueError, 'label budget'):
                cli.verify_prepared_data(root, config)

    def test_importer_filters_time_forfeits_short_clocks_and_low_ratings(self):
        from inez.curriculum import import_games
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pgn = root / 'elite.pgn'
            games = []
            # 1: Time forfeit (should be filtered when require_normal_termination=True)
            games.append(
                '[Event "Rated Blitz game"]\n[WhiteElo "2550"]\n[BlackElo "2500"]\n'
                '[TimeControl "180+2"]\n[Termination "Time forfeit"]\n[Result "1-0"]\n\n'
                '1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 1-0\n'
            )
            # 2: Zero-increment fast blitz 180+0 (should be filtered when min_time_control_seconds=260)
            games.append(
                '[Event "Rated Blitz game"]\n[WhiteElo "2550"]\n[BlackElo "2500"]\n'
                '[TimeControl "180+0"]\n[Termination "Normal"]\n[Result "1-0"]\n\n'
                '1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 Bb4 5. Bg2 O-O 1-0\n'
            )
            # 3: Rating below 2400 (should be filtered when min_rating=2400)
            games.append(
                '[Event "Rated Blitz game"]\n[WhiteElo "2390"]\n[BlackElo "2500"]\n'
                '[TimeControl "300+0"]\n[Termination "Normal"]\n[Result "1-0"]\n\n'
                '1. Nf3 d5 2. g3 Nf6 3. Bg2 c6 4. O-O Bf5 5. d3 e6 1-0\n'
            )
            # Add valid 2400+ Normal games (>=14 plies) with 180+2 / 600+0 to populate all 3 splits
            import random
            rng = random.Random(11)
            for idx in range(40):
                board = chess.Board()
                game = chess.pgn.Game()
                game.headers.update(
                    Event='Rated Rapid game', WhiteElo='2450', BlackElo='2420',
                    TimeControl='180+2' if idx % 2 == 0 else '600+0',
                    Termination='Normal', Result='1/2-1/2',
                )
                node = game
                for _ in range(18):
                    if board.outcome(claim_draw=False) is not None:
                        break
                    move = rng.choice(list(board.legal_moves))
                    node = node.add_variation(move)
                    board.push(move)
                outcome = board.outcome(claim_draw=False)
                if outcome:
                    game.headers['Result'] = outcome.result()
                games.append(str(game))
            pgn.write_text('\n\n'.join(games) + '\n')
            manifest = import_games(
                pgn, root / 'human', uri='fixture://elite', license_name='CC0-1.0',
                tactics=ROOT / 'tests/fixtures/independent_tactics.json',
                max_games=50, samples_per_game=1, min_ply=6, min_rating=2400,
                min_time_control_seconds=260, require_normal_termination=True, seed=20261024,
            )
            prep = manifest['preparation']
            self.assertEqual(prep['read_games'], 43)
            self.assertEqual(prep['accepted_games'], 40)
            self.assertEqual(prep['invalid_or_filtered_games'], 3)
            self.assertEqual(prep['min_rating'], 2400)
            self.assertEqual(prep['min_time_control_seconds'], 260)
            self.assertTrue(prep['require_normal_termination'])


if __name__ == '__main__':
    unittest.main()
