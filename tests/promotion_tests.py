#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fixed-sample paired promotion evidence and failure-safe input handling."""

import copy
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))

from inez.curriculum import evaluate_candidate, paired_score_test, promotion_decision


TACTICS = [{'name': 'held-out-mate', 'solved': True}]


def match_records(pairs):
    games = []
    for pair, scores in enumerate(pairs):
        for white, score in zip((True, False), scores):
            result = ('*' if score is None else '1/2-1/2' if score == 0.5 else
                      '1-0' if (score == 1.0) == white else '0-1')
            games.append({'pair': pair, 'candidate_white': white,
                          'candidate_score': score, 'result': result,
                          'opening_position': f'independent-opening-{pair}'})
    return games


def decision(games, **options):
    scores = [game['candidate_score'] for game in games if game['candidate_score'] is not None]
    return promotion_decision(sum(scores), len(scores), TACTICS, TACTICS,
                              match_games=games, **options)


class PromotionTests(unittest.TestCase):
    def test_exact_distribution_matches_independent_sign_enumeration(self):
        # Pair differences include all five possible pentanomial outcomes.
        games = match_records([(1, 1), (1, 0.5), (1, 0), (0, 0.5), (0, 0), (1, 1)])
        result = paired_score_test(games, expected_pairs=6)
        differences = [2, 1, 0, -1, -2, 2]
        magnitudes = [abs(value) for value in differences if value]
        possible = [sum(sign * magnitude for sign, magnitude in zip(signs, magnitudes))
                    for signs in itertools.product((-1, 1), repeat=len(magnitudes))]
        expected = sum(value >= sum(differences) for value in possible) / len(possible)
        self.assertEqual(result['p_value'], expected)
        self.assertEqual(result['pentanomial'], [1, 1, 1, 1, 2])
        self.assertEqual(result['completed_pairs'], 6)
        self.assertTrue(result['valid'])

    def test_equal_aggregate_results_can_have_different_pair_evidence(self):
        # Both matches are 18 wins / 6 losses (75%). Their within-opening
        # correlation differs, so the evidence cannot be reduced to 24 IID games.
        concentrated = decision(match_records([(1, 1)] * 9 + [(0, 0)] * 3))
        spread = decision(match_records([(1, 1)] * 6 + [(1, 0)] * 6))
        self.assertEqual(concentrated['score'], spread['score'])
        self.assertGreater(concentrated['paired_test']['p_value'], 0.05)
        self.assertEqual(spread['paired_test']['p_value'], 1 / 64)
        self.assertFalse(concentrated['passed'])
        self.assertTrue(spread['passed'])

    def test_clear_wins_pass_but_ties_and_small_samples_do_not(self):
        clear = decision(match_records([(1, 1)] * 8), expected_pairs=8)
        self.assertTrue(clear['passed'])
        self.assertEqual(clear['paired_test']['p_value'], 1 / 256)
        self.assertIsNone(clear['score_lower_bound'])
        for pairs in ([(1, 1)], [(1, 0)] * 8, [(0.5, 0.5)] * 8, [(0, 0)] * 8):
            with self.subTest(pairs=pairs):
                self.assertFalse(decision(match_records(pairs))['passed'])
        tied = paired_score_test(match_records([(0.5, 0.5)] * 8))
        self.assertEqual(tied['p_value'], 1.0)
        self.assertEqual(tied['pentanomial'], [0, 0, 8, 0, 0])

    def test_alpha_and_practical_score_threshold_are_both_required(self):
        games = match_records([(1, 1)] * 8)
        self.assertFalse(decision(games, alpha=0.001)['passed'])
        games = match_records([(1, 1)] * 6 + [(1, 0)] * 6)
        result = decision(games, minimum_score=0.8)
        self.assertLess(result['paired_test']['p_value'], 0.05)
        self.assertFalse(result['passed'])
        self.assertIn('score_below_minimum', result['rejection_reasons'])

    def test_aggregate_legacy_evidence_cannot_authorize_promotion(self):
        result = promotion_decision(100, 100, TACTICS, TACTICS)
        self.assertFalse(result['passed'])
        self.assertIn('missing_paired_results', result['rejection_reasons'])
        self.assertIsNone(result['paired_test']['p_value'])

    def test_missing_legs_pairs_and_color_reversal_are_rejected(self):
        games = match_records([(1, 1)] * 8)
        result = decision(games[:-1], expected_pairs=8)
        self.assertFalse(result['passed'])
        self.assertIn('incomplete_opening_pairs', result['rejection_reasons'])
        result = decision(games[:-2], expected_pairs=8)
        self.assertIn('incomplete_opening_pairs', result['rejection_reasons'])
        bad = copy.deepcopy(games)
        bad[1]['candidate_white'] = True
        result = decision(bad)
        self.assertFalse(result['passed'])
        self.assertIn('incomplete_opening_pairs', result['rejection_reasons'])
        self.assertIsNone(result['paired_test']['p_value'])

    def test_unfinished_games_and_engine_faults_reject_even_with_many_wins(self):
        games = match_records([(1, 1)] * 8 + [(1, None)])
        result = decision(games, expected_pairs=9)
        self.assertFalse(result['passed'])
        self.assertIn('unfinished_match_games', result['rejection_reasons'])
        games = match_records([(1, 1)] * 8)
        games[0]['fault'] = 'engine_crash'
        result = decision(games)
        self.assertFalse(result['passed'])
        self.assertIn('match_game_faults', result['rejection_reasons'])

    def test_missing_reused_or_mismatched_opening_positions_reject(self):
        for mode in ('missing', 'reused', 'mismatched'):
            games = match_records([(1, 1)] * 8)
            if mode == 'missing':
                del games[0]['opening_position']
            elif mode == 'reused':
                for game in games[2:4]:
                    game['opening_position'] = games[0]['opening_position']
            else:
                games[0]['opening_position'] = 'different-position'
            with self.subTest(mode=mode):
                result = decision(games)
                self.assertFalse(result['passed'])
                self.assertIsNone(result['paired_test']['p_value'])

    def test_inconsistent_score_result_and_aggregate_reject(self):
        games = match_records([(1, 1)] * 8)
        games[0]['result'] = '0-1'
        self.assertFalse(decision(games)['passed'])
        games = match_records([(1, 1)] * 8)
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            promotion_decision(15, 16, TACTICS, TACTICS, match_games=games)

    def test_invalid_scores_limits_and_tactical_suites_are_rejected(self):
        for alpha in (True, 0, 0.5, float('nan'), float('inf')):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                paired_score_test([], alpha=alpha)
        for expected in (True, 0, -1, 1.5):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                paired_score_test([], expected_pairs=expected)
        for score in (True, -1, 0.2, float('nan'), float('inf')):
            games = match_records([(1, 1)] * 8)
            games[0]['candidate_score'] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                paired_score_test(games)
        for minimum in (0.5, 1.1, float('nan')):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                decision(match_records([(1, 1)] * 8), minimum_score=minimum)
        for points, games in ((-1, 2), (3, 2), (float('nan'), 2), (1, 1.5)):
            with self.subTest(points=points, games=games), self.assertRaises(ValueError):
                promotion_decision(points, games, TACTICS, TACTICS)
        with self.assertRaisesRegex(ValueError, 'suites'):
            promotion_decision(16, 16, [{'name': 'different', 'solved': True}], TACTICS)
        with self.assertRaisesRegex(ValueError, 'booleans'):
            promotion_decision(16, 16, [{'name': 'held-out-mate', 'solved': 1}], TACTICS)

    def test_tactical_regression_blocks_statistically_clear_match_win(self):
        games = match_records([(1, 1)] * 8)
        result = promotion_decision(16, 16, [{'name': 'held-out-mate', 'solved': False}],
                                    TACTICS, match_games=games)
        self.assertFalse(result['passed'])
        self.assertEqual(result['tactical_regressions'], ['held-out-mate'])
        self.assertIn('tactical_regression', result['rejection_reasons'])
        result = promotion_decision(16, 16, [], [], match_games=games)
        self.assertFalse(result['passed'])
        self.assertIn('missing_tactical_suite', result['rejection_reasons'])

    def test_evaluation_rejects_repeated_and_transposed_openings_before_model_load(self):
        for openings in ([[], []],
                [['e2e4', 'e7e5', 'g1f3', 'b8c6'], ['g1f3', 'b8c6', 'e2e4', 'e7e5']],
                [['f2f3', 'e7e5', 'g2g4', 'd8h4']]):
            with self.subTest(openings=openings), self.assertRaisesRegex(ValueError, 'opening'):
                evaluate_candidate(None, None, tactics=TACTICS, openings=openings,
                                   pairs=len(openings), simulations=1, max_plies=1)
        with self.assertRaisesRegex(ValueError, 'limits'):
            evaluate_candidate(None, None, tactics=TACTICS, openings=[[]],
                               pairs=2, simulations=1, max_plies=1)

    def test_evaluation_records_complete_color_pairs_and_rejects_move_caps(self):
        def move(board, evaluator, simulations):
            return {next(iter(board.legal_moves)).uci(): 1}
        with patch('inez.curriculum.NeuralEvaluator'), \
                patch('inez.curriculum.search', side_effect=move), \
                patch('inez.curriculum.tactical_report', return_value=TACTICS):
            report = evaluate_candidate(None, None, tactics=TACTICS,
                openings=[[], ['e2e4']], pairs=2, simulations=1, max_plies=1)
        self.assertFalse(report['passed'])
        self.assertEqual(report['completed_games'], 0)
        self.assertEqual(len(report['games']), 4)
        self.assertEqual([game['candidate_white'] for game in report['games']], [True, False] * 2)
        self.assertEqual(report['games'][0]['opening_position'], report['games'][1]['opening_position'])
        self.assertNotEqual(report['games'][0]['opening_position'], report['games'][2]['opening_position'])
        self.assertEqual([game['result'] for game in report['games']], ['*'] * 4)
        self.assertIn('unfinished_match_games', report['rejection_reasons'])


if __name__ == '__main__':
    unittest.main()
