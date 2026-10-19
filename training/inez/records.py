# SPDX-License-Identifier: GPL-3.0-only
"""Versioned training records, game-safe splits, and dataset provenance."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import chess

from .encoding import CONTRACT_VERSION, expected_value, wdl_from_result

RECORD_VERSION = 1
MANIFEST_VERSION = 1
SPLITS = ("train", "validation", "evaluation")
MAX_COUNTER = 2**64 - 1


def _object(value, fields, context):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("{} must contain exactly {}".format(context, sorted(fields)))


def _text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(context + " must be a nonempty string")


def _integer(value, context, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(context + " must be an integer >= " + str(minimum))


def _probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probabilities must be finite numbers in [0, 1]")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(text):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def _teacher(teacher):
    _object(teacher, ("name", "version", "limits"), "teacher")
    _text(teacher["name"], "teacher name")
    _text(teacher["version"], "teacher version")
    limits = teacher["limits"]
    if not isinstance(limits, dict) or not limits or set(limits) - {"nodes", "depth", "movetime_ms"}:
        raise ValueError("teacher limits require nodes, depth, or movetime_ms")
    for key, value in limits.items():
        _integer(value, "teacher " + key, 1)


def _initial_board(fen):
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("initial_fen must have six fields")
    if not re.fullmatch(r"[pnbrqkPNBRQK1-8]+(?:/[pnbrqkPNBRQK1-8]+){7}", fields[0]):
        raise ValueError("initial_fen must use standard piece placement")
    rights = fields[2]
    if rights != "-" and (not rights or rights != "".join(c for c in "KQkq" if c in rights)):
        raise ValueError("castling rights must be unique and ordered KQkq")
    for index in (4, 5):
        if not re.fullmatch(r"[0-9]+", fields[index]) or int(fields[index]) > MAX_COUNTER:
            raise ValueError("FEN counters must be unsigned 64-bit integers")
    if int(fields[5]) == 0:
        raise ValueError("fullmove number must be positive")
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("invalid initial chess position")
    if board.ep_square is not None and board.halfmove_clock != 0:
        raise ValueError("en passant requires a zero halfmove clock")
    return board


def validate_record(record):
    """Validate a full game and its sampled prefix; return that prefix board.

    Policy is a sparse distribution over legal UCI moves. Empty distributions
    explicitly represent value-only samples. WDL always uses the sampled side
    to move's perspective. Future moves are never added to the returned board.
    """
    _object(record, ("record_version", "encoding_version", "source_id", "game_id",
                     "initial_fen", "moves", "ply", "result", "policy", "wdl",
                     "label_kind", "teacher"), "record")
    if type(record["record_version"]) is not int or record["record_version"] != RECORD_VERSION:
        raise ValueError("unsupported record version")
    if type(record["encoding_version"]) is not int or record["encoding_version"] != CONTRACT_VERSION:
        raise ValueError("unsupported encoding version")
    for field in ("source_id", "game_id", "initial_fen"):
        _text(record[field], field)
    board = _initial_board(record["initial_fen"])
    moves = record["moves"]
    if not isinstance(moves, list) or any(not isinstance(move, str) for move in moves):
        raise ValueError("moves must be a list of UCI strings")
    _integer(record["ply"], "ply")
    if record["ply"] > len(moves):
        raise ValueError("sample ply exceeds the recorded game")
    sample = board.copy(stack=True) if record["ply"] == 0 else None
    for ply, notation in enumerate(moves, 1):
        if board.is_game_over(claim_draw=False):
            raise ValueError("game contains moves after a forced outcome")
        move = chess.Move.from_uci(notation)
        if move not in board.legal_moves:
            raise ValueError("illegal game move at ply {}: {}".format(ply, notation))
        board.push(move)
        if board.halfmove_clock > MAX_COUNTER or board.fullmove_number > MAX_COUNTER:
            raise ValueError("game move exceeds native move-counter range")
        if ply == record["ply"]:
            sample = board.copy(stack=True)
    result = record["result"]
    if result not in ("1-0", "0-1", "1/2-1/2", "*"):
        raise ValueError("invalid game result")
    outcome = board.outcome(claim_draw=False)
    if outcome is not None and result != outcome.result():
        raise ValueError("game result disagrees with forced final outcome")

    policy = record["policy"]
    if not isinstance(policy, dict):
        raise ValueError("policy must map UCI moves to probabilities")
    if sample.is_game_over(claim_draw=False) and policy:
        raise ValueError("terminal records must have an empty policy")
    for notation, probability in policy.items():
        if not isinstance(notation, str) or chess.Move.from_uci(notation) not in sample.legal_moves:
            raise ValueError("policy contains an illegal move")
        _probability(probability)
    if policy and not math.isclose(sum(policy.values()), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("nonempty policy probabilities must sum to one")
    if not isinstance(record["wdl"], list) or len(record["wdl"]) != 3:
        raise ValueError("wdl must contain [win, draw, loss]")
    for probability in record["wdl"]:
        _probability(probability)
    expected_value(record["wdl"])
    if not math.isclose(sum(record["wdl"]), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("WDL probabilities must sum to one")

    kind = record["label_kind"]
    if kind not in ("game_result", "teacher_wdl", "search"):
        raise ValueError("unsupported label kind")
    if kind == "game_result":
        if record["teacher"] is not None:
            raise ValueError("game-result labels must not claim a teacher")
        if result == "*":
            raise ValueError("unfinished games have no game-result label")
        expected = wdl_from_result(result, sample.turn)
        if any(abs(a - b) > 1e-6 for a, b in zip(record["wdl"], expected)):
            raise ValueError("game-result WDL has the wrong result or perspective")
    else:
        _teacher(record["teacher"])
    sample_outcome = sample.outcome(claim_draw=False)
    if sample_outcome is not None:
        expected = wdl_from_result(sample_outcome.result(), sample.turn)
        if any(abs(a - b) > 1e-6 for a, b in zip(record["wdl"], expected)):
            raise ValueError("terminal WDL contradicts the rules")
    return sample


def game_fingerprint(record):
    """Full trajectory identity, independent of sample ply, source, and labels."""
    initial = chess.Board(record["initial_fen"]).fen(en_passant="fen")
    return _digest({"initial_fen": initial, "moves": record["moves"]})


def _load_records(paths):
    records = []
    inputs = []
    for path in paths:
        path = Path(path)
        data = path.read_bytes()
        inputs.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest()})
        for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = read_json(line)
                validate_record(record)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError("{}:{}: {}".format(path, number, error)) from error
            records.append(record)
    if not records:
        raise ValueError("dataset has no records")
    return records, inputs


def load_records(paths):
    return _load_records(paths)[0]


def assign_splits(records, seed="inez-v1", group_by="game", fractions=(0.8, 0.1, 0.1)):
    """Stable splits; identical trajectories never cross split boundaries."""
    _text(seed, "split seed")
    if group_by not in ("game", "source"):
        raise ValueError("group_by must be game or source")
    if len(fractions) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in fractions):
        raise ValueError("split fractions must be three positive finite numbers")
    if not math.isclose(sum(fractions), 1, abs_tol=1e-9, rel_tol=0):
        raise ValueError("split fractions must sum to one")
    identities = {}
    parents = {record["source_id"]: record["source_id"] for record in records}

    def find(source):
        while parents[source] != source:
            parents[source] = parents[parents[source]]
            source = parents[source]
        return source

    owners = {}
    for record in records:
        fingerprint = game_fingerprint(record)
        identity = (record["source_id"], record["game_id"])
        if identity in identities and identities[identity] != fingerprint:
            raise ValueError("one source/game_id identifies different full games")
        identities[identity] = fingerprint
        source = record["source_id"]
        if fingerprint in owners:
            roots = sorted((find(source), find(owners[fingerprint])))
            parents[roots[1]] = roots[0]
        owners[fingerprint] = source

    result = {split: [] for split in SPLITS}
    results = {}
    for record in records:
        identity = (record["source_id"], record["game_id"])
        if identity in results and results[identity] != record["result"]:
            raise ValueError("one source/game_id has conflicting game results")
        results[identity] = record["result"]
        group = game_fingerprint(record) if group_by == "game" else find(record["source_id"])
        digest = hashlib.sha256((seed + "\0" + group).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        split = SPLITS[0] if bucket < fractions[0] else (
            SPLITS[1] if bucket < fractions[0] + fractions[1] else SPLITS[2])
        result[split].append(record)
    for split in SPLITS:
        result[split].sort(key=_canonical)
    return result


def audit_splits(splits):
    """Report geometric position overlap, including common opening positions."""
    positions = {}
    samples = set()
    duplicates = 0
    for split in SPLITS:
        for record in splits[split]:
            board = validate_record(record)
            identity = (game_fingerprint(record), record["ply"])
            if identity in samples:
                duplicates += 1
            samples.add(identity)
            key = " ".join(board.fen(en_passant="legal").split()[:4])
            positions.setdefault(key, set()).add(split)
    overlapping = sorted(key for key, names in positions.items() if len(names) > 1)
    return {"duplicate_samples": duplicates,
            "cross_split_positions": len(overlapping),
            "overlap_examples": overlapping[:20]}


def _sources(value, required):
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("source manifest must cover exactly the dataset source IDs")
    for name, source in value.items():
        _object(source, ("uri", "license", "sha256"), "source " + name)
        _text(source["uri"], "source URI")
        _text(source["license"], "source license")
        checksum = source["sha256"]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("source SHA256 must be 64 lowercase hexadecimal digits")


def write_dataset(paths, output, sources, seed="inez-v1", group_by="game", require_disjoint=False):
    """Publish validated JSONL splits and a checksummed manifest as one directory."""
    paths = [Path(path) for path in paths]
    records, inputs = _load_records(paths)
    _sources(sources, {record["source_id"] for record in records})
    splits = assign_splits(records, seed=seed, group_by=group_by)
    audit = audit_splits(splits)
    if require_disjoint and audit["cross_split_positions"]:
        raise ValueError("cross-split position overlap; review or filter the dataset")
    output = Path(output)
    if output.exists():
        raise ValueError("output directory already exists")
    if not output.parent.is_dir():
        raise ValueError("output parent directory does not exist")
    manifest = {"manifest_version": MANIFEST_VERSION, "record_version": RECORD_VERSION,
                "encoding_version": CONTRACT_VERSION, "seed": seed, "group_by": group_by,
                "fractions": dict(zip(SPLITS, (0.8, 0.1, 0.1))),
                "sources": sources, "audit": audit, "inputs": inputs, "splits": {}}
    with tempfile.TemporaryDirectory(prefix=".inez-dataset-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for split in SPLITS:
            content = "".join(_canonical(record) + "\n" for record in splits[split])
            (staging / (split + ".jsonl")).write_text(content, encoding="utf-8")
            manifest["splits"][split] = {"records": len(splits[split]),
                                         "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            raise ValueError("output directory already exists")
        os.rename(staging, output)
    return manifest
