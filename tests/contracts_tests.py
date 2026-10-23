#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Compare native inference features with independently reconstructed Python features."""

from pathlib import Path
import random
import sys
import unittest

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from inez.encoding import (CONTRACT_VERSION, FEATURE_COUNT, POLICY_SIZE, encode_position,
                           expected_value, legal_masked_softmax, move_to_policy_index,
                           policy_index_to_move, wdl_from_result)
from differential import RulesProcess

BINARY = None


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.native = RulesProcess(BINARY)
        self.addCleanup(self.native.close)

    def compare(self, response, board):
        self.assertEqual(response.get("contract_version"), CONTRACT_VERSION)
        self.assertEqual(response.get("fen"), board.fen(en_passant="fen"))
        expected = encode_position(board)
        actual = response.get("features", [])
        self.assertEqual(len(actual), 64 * FEATURE_COUNT)
        for index, (left, right) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(left, right, places=7, msg="feature index {}".format(index))
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        self.assertEqual(response.get("legal_moves"), [move.uci() for move in moves])
        indices = [move_to_policy_index(board, move) for move in moves]
        self.assertEqual(response.get("legal_indices"), indices)
        self.assertEqual(len(indices), len(set(indices)))
        for index, move in zip(indices, moves):
            self.assertEqual(policy_index_to_move(board, index), move)

    def load(self, fen=chess.STARTING_FEN):
        board = chess.Board(fen)
        self.compare(self.native.request("fen " + fen), board)
        return board

    def test_special_positions(self):
        for fen in (
            chess.STARTING_FEN,
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 41 12",
            "4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1",
            "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",
            "4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1",
            "7k/6Q1/5K2/8/8/8/8/8 b - - 150 1",
            "7k/8/8/8/8/8/8/K7 w - - 150 1",
        ):
            with self.subTest(fen=fen):
                board = self.load(fen)
                for move in list(board.legal_moves):
                    board.push(move)
                    self.compare(self.native.request("push " + move.uci()), board)
                    board.pop()
                    self.compare(self.native.request("pop"), board)

    def test_repetition_and_history_survive_undo(self):
        board = self.load()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 3:
            board.push_uci(move)
            self.compare(self.native.request("push " + move), board)
        while board.move_stack:
            board.pop()
            self.compare(self.native.request("pop"), board)
        # Importing the same FEN discards all earlier history on both sides.
        fen = board.fen(en_passant="fen")
        self.compare(self.native.request("fen " + fen), chess.Board(fen))

    def test_seeded_games(self):
        rng = random.Random(20261019)
        for _ in range(3):
            board = self.load()
            for _ in range(50):
                if board.is_game_over():
                    break
                move = rng.choice(sorted(board.legal_moves, key=lambda item: item.uci()))
                board.push(move)
                self.compare(self.native.request("push " + move.uci()), board)
            for _ in range(min(10, len(board.move_stack))):
                board.pop()
                self.compare(self.native.request("pop"), board)

    def test_rejected_input_preserves_features(self):
        board = self.load()
        for command in ("fen broken", "push e2e5", "push a7a8n", "pop"):
            self.assertIn("error", self.native.request(command))
            self.compare(self.native.request("encode"), board)

    def test_values_and_masking(self):
        self.assertEqual(wdl_from_result("0-1", chess.BLACK), (1, 0, 0))
        self.assertEqual(wdl_from_result("0-1", chess.WHITE), (0, 0, 1))
        self.assertAlmostEqual(expected_value((0.6, 0.3, 0.1)), 0.5)
        logits = [0.0] * POLICY_SIZE
        logits[12] = 1000.0
        probabilities = legal_masked_softmax(logits, [12, 34])
        self.assertEqual(probabilities[12], 1)
        self.assertEqual(probabilities[34], 0)
        self.assertEqual(legal_masked_softmax(logits, []), [0.0] * POLICY_SIZE)
        with self.assertRaises(ValueError):
            legal_masked_softmax(logits, [12, 12])
        with self.assertRaises(ValueError):
            expected_value((0.8, 0.4, -0.2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: contracts_tests.py /path/to/inezita-encode")
    BINARY = sys.argv[1]
    unittest.main(argv=[sys.argv[0]])
