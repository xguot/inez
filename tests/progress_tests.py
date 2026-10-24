#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Interrupted game generation resumes without replaying completed evidence."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from inez.curriculum import evaluate_candidate, selfplay_records
from inez.progress import GameProgress, content_digest


IDENTITY = {'weights_sha256': 'fixture-weights', 'source_sha256': 'fixture-code'}
TACTICS = [{'name': 'fixture', 'solved': True}]
FOOLS = ['f2f3', 'e7e5', 'g2g4', 'd8h4']


def selfplay_search(board, evaluator, simulations, *, rng, on_update, **kwargs):
    on_update(SimpleNamespace(value=rng.uniform(-1, 1)))
    return {FOOLS[len(board.move_stack)]: 1}


def resign_chunk(path):
    saved = json.loads(path.read_text())
    signed = {key: value for key, value in saved.items() if key != 'sha256'}
    saved['sha256'] = content_digest(signed)
    path.write_text(json.dumps(saved))


class ProgressTests(unittest.TestCase):
    def selfplay(self, path=None, **kwargs):
        settings = dict(games=3, simulations=1, max_plies=4, seed=5, excluded=set(),
                        progress_path=path, progress_identity=IDENTITY)
        settings.update(kwargs)
        return selfplay_records(None, **settings)

    def test_selfplay_resume_skips_completed_games_and_restores_rng_exactly(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('inez.curriculum.NeuralEvaluator'), \
                patch('inez.curriculum.search', side_effect=selfplay_search) as search:
            expected = self.selfplay()
            stop = SimpleNamespace(requested=False)
            original = GameProgress.append
            def stop_after_first(progress, payload):
                original(progress, payload)
                stop.requested = True
            search.reset_mock()
            with patch.object(GameProgress, 'append', stop_after_first), \
                    self.assertRaises(InterruptedError):
                self.selfplay(tmp, stop=stop)
            self.assertEqual(search.call_count, 4)
            first = (Path(tmp) / 'game-0000000.json').read_bytes()
            search.reset_mock()
            actual = self.selfplay(tmp)
            self.assertEqual(actual, expected)
            self.assertEqual(search.call_count, 8)
            self.assertEqual((Path(tmp) / 'game-0000000.json').read_bytes(), first)
            search.reset_mock()
            self.assertEqual(self.selfplay(tmp), expected)
            search.assert_not_called()

    def test_discarded_attempts_are_committed_without_draw_targets(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('inez.curriculum.NeuralEvaluator'), \
                patch('inez.curriculum.search', side_effect=selfplay_search) as search:
            records, report = self.selfplay(tmp, max_plies=1)
            self.assertEqual(records, [])
            self.assertEqual(report['unfinished_games_discarded'], 3)
            self.assertEqual(len(list(Path(tmp).glob('game-*.json'))), 3)
            search.reset_mock()
            self.assertEqual(self.selfplay(tmp, max_plies=1), (records, report))
            search.assert_not_called()
            chunk = json.loads((Path(tmp) / 'game-0000000.json').read_text())['payload']
            self.assertEqual(chunk['result'], '*')
            self.assertTrue(chunk['discarded'])
            self.assertEqual(chunk['records'], [])

    def test_changed_requests_and_identities_fail_before_any_new_search(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('inez.curriculum.NeuralEvaluator'), \
                patch('inez.curriculum.search', side_effect=selfplay_search) as search:
            self.selfplay(tmp)
            for change in ({'seed': 6}, {'simulations': 2}, {'max_plies': 5},
                    {'games': 4}, {'excluded': {'reserved-position'}},
                    {'progress_identity': {'weights_sha256': 'other'}}):
                search.reset_mock()
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'request'):
                    self.selfplay(tmp, **change)
                search.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'progress_identity'):
                self.selfplay(Path(tmp) / 'missing-identity', progress_identity=None)

    def test_chunk_corruption_holes_and_semantically_invalid_records_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('inez.curriculum.NeuralEvaluator'), \
                patch('inez.curriculum.search', side_effect=selfplay_search):
            root = Path(tmp)
            self.selfplay(root)
            path = root / 'game-0000000.json'
            original = path.read_bytes()
            saved = json.loads(original)
            saved['payload']['result'] = '1-0'
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.selfplay(root)
            resign_chunk(path)
            with self.assertRaisesRegex(ValueError, 'outcome'):
                self.selfplay(root)
            saved = json.loads(original)
            saved['payload']['rng_state'] = [1]
            path.write_text(json.dumps(saved))
            resign_chunk(path)
            with self.assertRaisesRegex(ValueError, 'random state'):
                self.selfplay(root)
            path.write_bytes(original)
            path.rename(root / 'game-0000003.json')
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                self.selfplay(root)

    def gate(self, path=None, **kwargs):
        settings = dict(tactics=TACTICS, openings=[[], ['e2e4']], pairs=2,
                        simulations=1, max_plies=1, progress_path=path,
                        progress_identity=IDENTITY)
        settings.update(kwargs)
        return evaluate_candidate(None, None, **settings)

    def test_promotion_resume_skips_completed_legs_and_keeps_unfinished_results(self):
        def search_one(board, evaluator, simulations):
            return {next(iter(board.legal_moves)).uci(): 1}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('inez.curriculum.NeuralEvaluator'), \
                patch('inez.curriculum.tactical_report', return_value=TACTICS), \
                patch('inez.curriculum.search', side_effect=search_one) as search:
            expected = self.gate()
            stop = SimpleNamespace(requested=False)
            original = GameProgress.append
            def stop_after_first(progress, payload):
                original(progress, payload)
                stop.requested = True
            with patch.object(GameProgress, 'append', stop_after_first), \
                    self.assertRaises(InterruptedError):
                self.gate(tmp, stop=stop)
            search.reset_mock()
            actual = self.gate(tmp)
            self.assertEqual(actual, expected)
            self.assertEqual(search.call_count, 3)
            self.assertFalse(actual['passed'])
            self.assertEqual([game['result'] for game in actual['games']], ['*'] * 4)
            self.assertIn('unfinished_match_games', actual['rejection_reasons'])
            with self.assertRaisesRegex(ValueError, 'request'):
                self.gate(tmp, alpha=0.01)
            path = Path(tmp) / 'game-0000000.json'
            saved = json.loads(path.read_text())
            saved['payload']['candidate_white'] = False
            path.write_text(json.dumps(saved))
            resign_chunk(path)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                self.gate(tmp)

    def test_completed_promotion_outcomes_are_reconstructed_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('inez.curriculum.NeuralEvaluator'), \
                patch('inez.curriculum.tactical_report', return_value=TACTICS), \
                patch('inez.curriculum.search', return_value={'d8h4': 1}) as search:
            settings = dict(openings=[FOOLS[:3]], pairs=1)
            expected = self.gate(tmp, **settings)
            self.assertEqual(expected['completed_games'], 2)
            self.assertEqual([game['candidate_score'] for game in expected['games']], [0, 1])
            search.reset_mock()
            self.assertEqual(self.gate(tmp, **settings), expected)
            search.assert_not_called()
            path = Path(tmp) / 'game-0000000.json'
            saved = json.loads(path.read_text())
            saved['payload']['candidate_score'] = 1.0
            path.write_text(json.dumps(saved))
            resign_chunk(path)
            with self.assertRaisesRegex(ValueError, 'outcome'):
                self.gate(tmp, **settings)


if __name__ == '__main__':
    unittest.main()
