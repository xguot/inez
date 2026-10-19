#!/usr/bin/env python3
"""Meaningful encoding, neural-interface, and portable-artifact regressions."""

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import chess
import torch
from torch.nn import functional as F

from inez.encoding import (
    FEATURE_COUNT, HISTORY_FRAMES, POLICY_SIZE, convert_wdl_perspective,
    encode_position, expected_value, legal_masked_softmax, legal_policy_indices,
    move_to_policy_index, policy_index_to_move, wdl_from_result,
)
from inez.model import (
    ChessTransformer, ModelConfig, legal_masked_policy, load_checkpoint,
    save_checkpoint, wdl_probabilities_and_value,
)


def small_config():
    return ModelConfig(width=32, layers=2, heads=4, dynamic_rank=4, policy_width=8)


def example_batch():
    first = chess.Board()
    second = first.copy()
    second.push_uci("e2e4")
    return torch.tensor([encode_position(first), encode_position(second)], dtype=torch.float32).reshape(
        2, 64, FEATURE_COUNT)


class EncodingTests(unittest.TestCase):
    def test_import_has_no_optional_runtime_dependency(self):
        training = str(Path(__file__).resolve().parents[1] / "training")
        script = ("import sys; sys.path.insert(0, {!r}); import inez.encoding; "
                  "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules").format(training)
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_initial_features_and_missing_history(self):
        board = chess.Board()
        original = board.fen()
        features = encode_position(board)
        self.assertEqual(len(features), 64 * 119)
        self.assertEqual(features[chess.E2 * 119], 1.0)
        self.assertEqual(features[chess.E8 * 119 + 11], 1.0)
        for square in chess.SQUARES:
            offset = square * 119
            self.assertEqual(features[offset + 13], 1.0)
            self.assertEqual(features[offset + 14:offset + 112], [0.0] * 98)
            self.assertEqual(features[offset + 112:offset + 116], [1.0] * 4)
            self.assertEqual(features[offset + 116:offset + 119], [0.0] * 3)
        self.assertEqual(board.fen(), original)
        self.assertEqual(len(board.move_stack), 0)

    def test_black_orients_every_history_frame_to_current_player(self):
        board = chess.Board()
        board.push_uci("e2e4")
        original = board.copy(stack=True)
        features = encode_position(board)
        # Current black e7 pawn becomes our e2 pawn. Past white e2 pawn is
        # still the opponent, and lies on oriented e7 in the previous frame.
        self.assertEqual(features[chess.E2 * 119], 1.0)
        self.assertEqual(features[chess.E7 * 119 + 14 + 6], 1.0)
        self.assertEqual(features[chess.E1 * 119 + 5], 1.0)
        self.assertTrue(all(features[square * 119 + 118] == 1.0 for square in chess.SQUARES))
        self.assertEqual(board.fen(), original.fen())
        self.assertEqual(board.move_stack, original.move_stack)

    def test_repetition_flags_exclude_future_history(self):
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(move)
        features = encode_position(board)
        for square in chess.SQUARES:
            self.assertEqual(features[square * 119 + 12], 1.0)
            for frame in range(1, 5):
                self.assertEqual(features[square * 119 + frame * 14 + 12], 0.0)
        imported = encode_position(chess.Board(board.fen()))
        self.assertEqual(imported[12], 0.0)
        self.assertEqual(imported[14:112], [0.0] * 98)

    def test_canonical_ep_and_saturated_clock(self):
        pinned = chess.Board("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1")
        self.assertFalse(any(encode_position(pinned)[square * 119 + 116] for square in chess.SQUARES))
        legal = chess.Board("4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1")
        features = encode_position(legal)
        self.assertEqual(sum(features[square * 119 + 116] for square in chess.SQUARES), 1.0)
        self.assertEqual(features[chess.F6 * 119 + 116], 1.0)
        for clock in (1, 149, 150, 500):
            board = chess.Board("7k/8/8/8/8/8/8/KR6 w - - {} 1".format(clock))
            expected = struct.unpack("f", struct.pack("f", min(clock, 150) / 150.0))[0]
            self.assertEqual(encode_position(board)[117], expected)

    def test_move_vocabulary_roundtrips_including_both_colors_promotions(self):
        for fen in (chess.STARTING_FEN,
                    "4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1",
                    "4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1"):
            board = chess.Board(fen)
            indices = legal_policy_indices(board)
            self.assertEqual(len(indices), len(set(indices)))
            for move in board.legal_moves:
                index = move_to_policy_index(board, move)
                self.assertEqual(policy_index_to_move(board, index), move)
        board = chess.Board("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1")
        self.assertEqual(move_to_policy_index(board, chess.Move.from_uci("g7g8q")), 54 * 64 + 62)
        for piece_index, suffix in enumerate("nbr"):
            self.assertEqual(move_to_policy_index(board, chess.Move.from_uci("g7h8" + suffix)),
                             4096 + piece_index * 64 + 6 * 8 + 7)
        with self.assertRaises(ValueError):
            policy_index_to_move(chess.Board(), 0)
        with self.assertRaises(ValueError):
            policy_index_to_move(chess.Board(), POLICY_SIZE)
        with self.assertRaises(ValueError):
            move_to_policy_index(chess.Board(), chess.Move.from_uci("e2e5"))

    def test_pure_policy_and_wdl_helpers(self):
        logits = [10000.0] * POLICY_SIZE
        probabilities = legal_masked_softmax(logits, [1, 20])
        self.assertEqual(probabilities[1], 0.5)
        self.assertEqual(probabilities[20], 0.5)
        self.assertEqual(sum(probabilities), 1.0)
        self.assertEqual(legal_masked_softmax(logits, []), [0.0] * POLICY_SIZE)
        logits[2] = float("nan")
        self.assertEqual(legal_masked_softmax(logits, [1, 20])[1], 0.5)
        self.assertEqual(wdl_from_result("1-0", chess.BLACK), (0.0, 0.0, 1.0))
        self.assertEqual(convert_wdl_perspective((0.6, 0.3, 0.1), chess.BLACK, chess.WHITE),
                         (0.1, 0.3, 0.6))
        self.assertAlmostEqual(expected_value((0.6, 0.3, 0.1)), 0.5)
        with self.assertRaises(ValueError):
            wdl_from_result("*", chess.WHITE)
        with self.assertRaises(ValueError):
            expected_value((float("nan"), 0.0, 1.0))
        with self.assertRaises(ValueError):
            legal_masked_softmax(logits, [1, 1])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny CPU regressions should not spawn a large BLAS thread pool.
        torch.set_num_threads(1)

    def test_default_is_a_small_chess_model(self):
        model = ChessTransformer()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreaterEqual(parameters, 3_000_000)
        self.assertLessEqual(parameters, 5_000_000)

    def test_forward_and_both_heads_receive_finite_gradients(self):
        torch.manual_seed(19)
        model = ChessTransformer(small_config())
        policy, wdl = model(example_batch())
        self.assertEqual(tuple(policy.shape), (2, 4288))
        self.assertEqual(tuple(wdl.shape), (2, 3))
        self.assertTrue(bool(torch.isfinite(policy).all()))
        loss = F.cross_entropy(policy, torch.tensor([12, 4200])) + F.cross_entropy(wdl, torch.tensor([0, 2]))
        loss.backward()
        for name in ("input_projection.weight", "geometry.relative", "geometry.dynamic.0.weight",
                     "policy_from.weight", "underpromotions.weight", "value_head.2.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(bool(torch.isfinite(gradient).all()), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_masks_empty_positions_and_extreme_logits(self):
        logits = torch.full((2, POLICY_SIZE), 10000.0, requires_grad=True)
        mask = torch.zeros((2, POLICY_SIZE), dtype=torch.bool)
        mask[1, [5, 17]] = True
        probabilities = legal_masked_policy(logits, mask)
        self.assertTrue(bool(torch.isfinite(probabilities).all()))
        self.assertEqual(float(probabilities[0].sum().detach()), 0.0)
        self.assertEqual(float(probabilities[1].sum().detach()), 1.0)
        self.assertEqual(float(probabilities[1, 5].detach()), 0.5)
        self.assertTrue(bool((probabilities[~mask] == 0).all()))
        probabilities[1, 5].backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool((logits.grad[~mask] == 0).all()))
        nonfinite = torch.full((1, POLICY_SIZE), float("nan"))
        self.assertTrue(bool((legal_masked_policy(nonfinite, torch.zeros_like(nonfinite, dtype=torch.bool)) == 0).all()))
        board = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")
        self.assertTrue(board.is_game_over())
        self.assertGreater(len(legal_policy_indices(board)), 0)

    def test_wdl_expectation_is_current_player_win_minus_loss(self):
        probabilities, value = wdl_probabilities_and_value(torch.tensor([[math.log(2), 0.0, math.log(3)]]))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))
        torch.testing.assert_close(value, torch.tensor([-1.0 / 6.0]))

    def test_invalid_inputs_and_configurations_rejected(self):
        for values in ({"width": 31}, {"heads": 0}, {"layers": True}, {"dropout": float("nan")},
                       {"ffn_ratio": 0}, {"dynamic_rank": -1}):
            with self.assertRaises(ValueError):
                ModelConfig(**values)
        model = ChessTransformer(small_config())
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 64, 118))
        features = example_batch()
        features[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(features)
        with self.assertRaises(ValueError):
            legal_masked_policy(torch.zeros(1, POLICY_SIZE), torch.zeros(1, POLICY_SIZE))

    def test_export_load_preserves_exact_cpu_predictions(self):
        torch.manual_seed(18)
        model = ChessTransformer(small_config()).eval()
        features = example_batch()
        with torch.no_grad():
            expected = model(features)
        with tempfile.TemporaryDirectory() as temporary:
            directory = save_checkpoint(model, Path(temporary) / "model")
            restored = load_checkpoint(directory, expected_config=small_config())
            self.assertFalse(restored.training)
            self.assertEqual(restored.metadata(), model.metadata())
            with torch.no_grad():
                actual = restored(features)
            for before, after in zip(expected, actual):
                torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, directory)

    def test_incompatible_manifest_and_corrupt_weights_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = save_checkpoint(ChessTransformer(small_config()), Path(temporary) / "model")
            manifest_path = directory / "manifest.json"
            original = json.loads(manifest_path.read_text())
            modifications = (
                lambda manifest: manifest.update(artifact_version=2),
                lambda manifest: manifest.update(architecture="a-language-model"),
                lambda manifest: manifest["encoding"].update(contract_version=2),
                lambda manifest: manifest["encoding"].update(contract_version=True),
                lambda manifest: manifest["encoding"].update(feature_count=118),
                lambda manifest: manifest["config"].update(unknown_parameter=1),
                lambda manifest: manifest.update(weights="../other.pt"),
            )
            for modify in modifications:
                manifest = json.loads(json.dumps(original))
                modify(manifest)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_checkpoint(directory)
            manifest_path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                load_checkpoint(directory, expected_config=ModelConfig(width=64, layers=2, heads=4))
            weights = directory / "weights.pt"
            weights.write_bytes(weights.read_bytes() + b"corruption")
            with self.assertRaises(ValueError):
                load_checkpoint(directory)

    def test_state_dictionary_structure_is_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = save_checkpoint(ChessTransformer(small_config()), Path(temporary) / "model")
            weights = directory / "weights.pt"
            state = torch.load(weights, weights_only=True)
            del state["input_projection.weight"]
            torch.save(state, weights)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                load_checkpoint(directory)

    def test_artifacts_are_explicitly_float32_and_reject_wrong_tensor_dtype(self):
        model = ChessTransformer(small_config()).double()
        with tempfile.TemporaryDirectory() as temporary:
            directory = save_checkpoint(model, Path(temporary) / "model")
            restored = load_checkpoint(directory)
            self.assertEqual(restored.input_projection.weight.dtype, torch.float32)
            torch.testing.assert_close(restored.input_projection.weight,
                                       model.input_projection.weight.float(), rtol=0.0, atol=0.0)
            weights = directory / "weights.pt"
            state = torch.load(weights, weights_only=True)
            state["input_projection.weight"] = state["input_projection.weight"].double()
            torch.save(state, weights)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_checkpoint(directory)


if __name__ == "__main__":
    unittest.main()
