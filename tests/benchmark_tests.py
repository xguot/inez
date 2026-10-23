#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Benchmark integrity checks: frozen weights, opening diversity and paired results."""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import chess
import chess.pgn

from benchmark_inezita import (artifact_record, audit_games, previous_artifact,
    position_key, select_openings, sha256)


def opening(moves):
    board = chess.Board()
    for move in moves.split():
        board.push_uci(move)
    return chess.pgn.Game.from_board(board)


class BenchmarkTests(unittest.TestCase):
    def test_opening_slice_is_unique_disjoint_and_preserves_history(self):
        first = opening('e2e4 e7e5 g1f3 b8c6')
        second = opening('d2d4 d7d5 c2c4 e7e6')
        third = opening('c2c4 e7e5 b1c3 g8f6')
        text = '\n\n'.join(str(g) for g in (first, first, second, third))
        pilot = select_openings(text, 0, 1, set())
        main = select_openings(text, 1, 2, set())
        self.assertEqual(list(pilot[0].mainline_moves()), list(first.mainline_moves()))
        self.assertEqual(list(main[0].mainline_moves()), list(second.mainline_moves()))
        self.assertEqual(list(main[1].mainline_moves()), list(third.mainline_moves()))
        excluded = {position_key(first.end().board())}
        self.assertEqual(list(select_openings(text, 0, 1, excluded)[0].mainline_moves()),
                         list(second.mainline_moves()))
        with self.assertRaisesRegex(ValueError, 'lacks'):
            select_openings(text, 0, 4, set())

    def test_copied_cluster_history_selects_previous_accepted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'inezita'
            for name in ('older', 'current'):
                (run / name).mkdir(parents=True)
                (run / name / 'manifest.json').write_text('{}')
            history = [f'/cluster/results/inezita/{name}' for name in ('older', 'current')]
            (run / 'accepted.json').write_text(json.dumps({'artifact': history[-1], 'history': history}))
            self.assertEqual(previous_artifact(run), (run / 'older').resolve())

    def test_modified_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'weights.pt').write_bytes(b'frozen test weights')
            (root / 'manifest.json').write_text(json.dumps({'weights': 'weights.pt',
                'weights_sha256': sha256(root / 'weights.pt'), 'config': {}}))
            artifact_record(root)
            (root / 'weights.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                artifact_record(root)

    def test_pair_audit_checks_colors_opening_history_and_faults(self):
        book = opening('e2e4 e7e5')
        games = []
        for white in ('inezita-candidate', 'inezita-incumbent'):
            game = opening('e2e4 e7e5 g1f3')
            game.headers.update(Round='1', White=white,
                Black='inezita-incumbent' if white == 'inezita-candidate' else 'inezita-candidate',
                Result='1-0', Termination='time forfeit')
            games.append(game)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'games.pgn'
            path.write_text('\n\n'.join(str(g) for g in games))
            summary = audit_games(path, [book])
            self.assertEqual((summary['games'], summary['pairs']), (2, 1))
            self.assertEqual((summary['wins'], summary['losses']), (1, 1))
            self.assertEqual(summary['pentanomial'], [0, 0, 1, 0, 0])
            self.assertEqual(summary['faults'], {'time forfeit': 2})
            games[1].headers.update(games[0].headers)
            path.write_text('\n\n'.join(str(g) for g in games))
            with self.assertRaisesRegex(ValueError, 'color reversal'):
                audit_games(path, [book])
            games[1] = opening('d2d4 d7d5')
            games[1].headers.update(games[0].headers)
            path.write_text('\n\n'.join(str(g) for g in games))
            with self.assertRaisesRegex(ValueError, 'opening/history'):
                audit_games(path, [book])

    def test_unfinished_game_supplies_no_draw_or_completed_pair(self):
        book = opening('e2e4 e7e5')
        game = chess.pgn.read_game(io.StringIO(str(book)))
        game.headers.update(Round='1', White='inezita-candidate', Black='inezita-incumbent',
                            Result='*', Termination='unterminated')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'games.pgn'
            path.write_text(str(game))
            summary = audit_games(path, [book])
            self.assertEqual((summary['unfinished'], summary['draws'], summary['pairs']), (1, 0, 0))


if __name__ == '__main__':
    unittest.main()
