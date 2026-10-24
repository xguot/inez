#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Native evaluator integration smoke test; no strength estimates or adjudication."""

from contextlib import ExitStack
import os
from pathlib import Path
import sys
import unittest

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]
BINARY = sys.argv.pop(1) if len(sys.argv) > 1 else str(ROOT / 'build/inezita')


class BaselineUciTests(unittest.TestCase):
    def test_completed_search_finds_mate_through_real_uci_client(self):
        with chess.engine.SimpleEngine.popen_uci(
                [BINARY, '--baseline'], timeout=15) as engine:
            engine.configure({'Evaluator': 'material', 'BatchSize': 4, 'Cpuct': '2.2'})
            board = chess.Board('6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1')
            result = engine.play(board, chess.engine.Limit(nodes=60))
            self.assertEqual(result.move, chess.Move.from_uci('e1e8'))
            board.push(result.move)
            self.assertTrue(board.is_checkmate())

    def test_both_evaluators_play_legal_moves_with_colors_reversed(self):
        with ExitStack() as stack:
            engines = []
            for evaluator, batch_size in (('material', 4), ('uniform', 1)):
                engine = stack.enter_context(chess.engine.SimpleEngine.popen_uci(
                    [BINARY, '--baseline'], timeout=15, env={**os.environ, 'INEZ_BASELINE': '1'}))
                self.assertEqual(engine.id['name'], 'Inezita 0.1.3')
                engine.configure({'Evaluator': evaluator, 'BatchSize': batch_size})
                engines.append(engine)
            openings = ('e2e4 e7e5 g1f3 b8c6 f1c4', 'e2e4 e7e6 d2d4 d7d5 e4e5')
            for opening in openings:
                for material_white in (True, False):
                    board, game_id, played = chess.Board(), object(), 0
                    for notation in opening.split():
                        board.push_uci(notation)
                    for _ in range(100):
                        if board.outcome(claim_draw=True) is not None:
                            break
                        engine = engines[0 if board.turn == material_white else 1]
                        result = engine.play(board, chess.engine.Limit(nodes=48), game=game_id)
                        self.assertIn(result.move, board.legal_moves)
                        board.push(result.move)
                        played += 1
                    self.assertGreater(played, 0)
                    # A smoke-test ply cap supplies no result and no Elo evidence.


if __name__ == '__main__':
    unittest.main()
