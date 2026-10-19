#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Validate or split version-1 Inez JSONL training records."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from inez.records import audit_splits, assign_splits, load_records, read_json, write_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate records and audit prospective splits")
    validate.add_argument("inputs", nargs="+", type=Path)
    split = subparsers.add_parser("split", help="write splits and a provenance manifest")
    split.add_argument("inputs", nargs="+", type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--sources", required=True, type=Path)
    split.add_argument("--require-disjoint", action="store_true")
    for command in (validate, split):
        command.add_argument("--seed", default="inez-v1")
        command.add_argument("--group-by", choices=("game", "source"), default="game")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            records = load_records(args.inputs)
            splits = assign_splits(records, seed=args.seed, group_by=args.group_by)
            result = {"records": len(records),
                      "splits": {name: len(items) for name, items in splits.items()},
                      "audit": audit_splits(splits)}
        else:
            sources = read_json(args.sources.read_text(encoding="utf-8"))
            result = write_dataset(args.inputs, args.output, sources, seed=args.seed,
                                   group_by=args.group_by, require_disjoint=args.require_disjoint)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(2, "error: {}\n".format(error))


if __name__ == "__main__":
    main()
