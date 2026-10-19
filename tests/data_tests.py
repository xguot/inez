#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Record validation, future-information boundaries, and dataset provenance."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from inez.encoding import encode_position
from inez.records import (SPLITS, assign_splits, audit_splits, game_fingerprint,
                          load_records, read_json, validate_record, write_dataset)


def record(moves=None, ply=0, result="0-1", game_id="game-1", source_id="source-1"):
    return {"record_version": 1, "encoding_version": 1, "source_id": source_id,
            "game_id": game_id, "initial_fen": chess.STARTING_FEN,
            "moves": ["f2f3", "e7e5", "g2g4", "d8h4"] if moves is None else moves, "ply": ply,
            "result": result, "policy": {},
            "wdl": [1, 0, 0] if ply % 2 else [0, 0, 1],
            "label_kind": "game_result", "teacher": None}


class DataTests(unittest.TestCase):
    def test_game_result_perspective_and_terminal_labels(self):
        for ply in range(5):
            self.assertEqual(len(validate_record(record(ply=ply)).move_stack), ply)
        terminal = record(ply=4)
        terminal["policy"] = {"a2a3": 1}
        with self.assertRaisesRegex(ValueError, "terminal"):
            validate_record(terminal)
        wrong = record(ply=1)
        wrong["wdl"] = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, "perspective"):
            validate_record(wrong)

    def test_features_do_not_see_future_game_moves(self):
        left = record(["e2e4", "e7e5", "g1f3"], ply=1)
        right = record(["e2e4", "c7c5", "g1f3"], ply=1)
        self.assertEqual(encode_position(validate_record(left)), encode_position(validate_record(right)))
        self.assertNotEqual(game_fingerprint(left), game_fingerprint(right))

    def test_rejects_malformed_records(self):
        invalid = []
        for field, value in (("record_version", 2), ("encoding_version", True),
                             ("ply", -1), ("ply", True), ("ply", 9),
                             ("moves", ["e2e5"]), ("initial_fen", "bad"),
                             ("result", "1-0"), ("wdl", [float("nan"), 0, 1]),
                             ("wdl", [True, 0, 0]), ("policy", {"e2e5": 1}),
                             ("policy", {"e2e4": 0.2}), ("label_kind", "unknown")):
            example = record()
            example[field] = value
            invalid.append(example)
        extra = record()
        extra["unexpected"] = 42
        invalid.append(extra)
        after_mate = record()
        after_mate["moves"].append("a2a3")
        invalid.append(after_mate)
        for example in invalid:
            with self.subTest(example=example):
                with self.assertRaises(ValueError):
                    validate_record(example)
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                read_json(text)

    def test_teacher_labels_require_provenance(self):
        example = record()
        example.update(label_kind="teacher_wdl", wdl=[0.2, 0.7, 0.1])
        with self.assertRaises(ValueError):
            validate_record(example)
        example["teacher"] = {"name": "fixture", "version": "test-1", "limits": {"nodes": 100}}
        validate_record(example)
        example["teacher"]["limits"]["nodes"] = 0
        with self.assertRaises(ValueError):
            validate_record(example)

    def test_non_native_fen_forms_are_rejected(self):
        for fen in (
            chess.STARTING_FEN.replace("KQkq", "HAha"),
            chess.STARTING_FEN.replace("0 1", "0 0"),
            chess.STARTING_FEN.replace("0 1", "+0 +1"),
            chess.STARTING_FEN.replace("0 1", "0 18446744073709551616"),
            chess.STARTING_FEN.replace("RNBQ", "R~NBQ"),
            "4k3/8/8/5pP1/8/8/8/4K3 w - f6 1 1",
        ):
            example = record([], result="1-0")
            example.update(initial_fen=fen, wdl=[1, 0, 0])
            with self.subTest(fen=fen):
                with self.assertRaises(ValueError):
                    validate_record(example)

    def test_games_stay_together_despite_different_sample_or_source(self):
        examples = [record(ply=0), record(ply=1), record(ply=2, source_id="copy", game_id="duplicate")]
        for group in ("game", "source"):
            splits = assign_splits(examples, group_by=group)
            self.assertEqual(sorted(len(items) for items in splits.values()), [0, 0, 3])
            self.assertEqual(splits, assign_splits(list(reversed(examples)), group_by=group))
        inconsistent = record(["d2d4"], game_id="game-1")
        with self.assertRaisesRegex(ValueError, "different full games"):
            assign_splits([record(), inconsistent])
        white_win = record(["e2e4"], result="1-0")
        black_win = record(["e2e4"], result="0-1")
        with self.assertRaisesRegex(ValueError, "conflicting game results"):
            assign_splits([white_win, black_win])

    def test_transposition_audit_reports_opening_overlap(self):
        first = record(["g1f3", "g8f6", "g2g3", "g7g6"], ply=4)
        second = record(["g2g3", "g7g6", "g1f3", "g8f6"], ply=4, game_id="other")
        splits = {"train": [first], "validation": [], "evaluation": [second]}
        audit = audit_splits(splits)
        self.assertEqual(audit["cross_split_positions"], 1)
        self.assertEqual(audit["duplicate_samples"], 0)

    def test_publish_manifest_checksums_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "records.jsonl"
            input_path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            self.assertEqual(len(load_records([input_path])), 1)
            sources = {"source-1": {"uri": "fixture://test-game", "license": "CC0-1.0", "sha256": "a" * 64}}
            output = root / "dataset"
            manifest = write_dataset([input_path], output, sources)
            self.assertEqual(manifest["inputs"][0]["sha256"], hashlib.sha256(input_path.read_bytes()).hexdigest())
            for split in SPLITS:
                digest = hashlib.sha256((output / (split + ".jsonl")).read_bytes()).hexdigest()
                self.assertEqual(manifest["splits"][split]["sha256"], digest)
            self.assertEqual(read_json((output / "manifest.json").read_text()), manifest)
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_dataset([input_path], output, sources)
            self.assertEqual(read_json((output / "manifest.json").read_text()), manifest)


if __name__ == "__main__":
    unittest.main()
