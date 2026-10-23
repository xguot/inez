#!/usr/bin/env python3
"""Compare the streaming rules CLI with an independent python-chess oracle.

Usage: python3 tests/differential.py /path/to/inezita-rules
The pinned oracle is a test dependency, not an engine dependency.
"""

import argparse
import json
import random
import selectors
import subprocess
import sys
import tempfile

try:
    import chess
except ImportError:
    sys.exit("Install the test oracle with: python3 -m pip install -r tests/requirements.txt")


class RulesProcess:
    def __init__(self, executable):
        self.diagnostics = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.diagnostics, text=True, bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def request(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        if not self.selector.select(timeout=30):
            raise AssertionError("CLI response timed out for {!r}".format(command))
        line = self.process.stdout.readline()
        if not line:
            self.diagnostics.seek(0)
            raise AssertionError("CLI exited for {!r}: {}".format(command, self.diagnostics.read()))
        try:
            result = json.loads(line)
        except ValueError as error:
            raise AssertionError("Non-JSON response to {!r}: {!r}".format(command, line)) from error
        if not isinstance(result, dict):
            raise AssertionError("Expected a JSON object, received {!r}".format(result))
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.diagnostics.close()


def expected_snapshot(board):
    count = 1
    while board.is_repetition(count + 1):
        count += 1
    outcome = board.outcome(claim_draw=False)
    return {
        "fen": board.fen(en_passant="fen"),
        "moves": sorted(move.uci() for move in board.legal_moves),
        "check": board.is_check(),
        "insufficient_material": board.is_insufficient_material(),
        "can_claim_fifty": board.can_claim_fifty_moves(),
        "can_claim_threefold": board.can_claim_threefold_repetition(),
        "repetition_count": count,
        "outcome": "ongoing" if outcome is None else outcome.termination.name.lower(),
        "winner": None if outcome is None or outcome.winner is None else (
            "white" if outcome.winner else "black"
        ),
    }


def compare(response, board, context):
    expected = expected_snapshot(board)
    for key, value in expected.items():
        if response.get(key) != value:
            raise AssertionError(
                "{}: {} mismatch\nFEN: {}\nexpected: {!r}\nactual: {!r}".format(
                    context, key, board.fen(en_passant="fen"), value, response.get(key)
                )
            )


class DifferentialTests:
    def __init__(self, cli):
        self.cli = cli
        self.board = chess.Board()
        self.snapshots = 0

    def check(self, response, context):
        compare(response, self.board, context)
        self.snapshots += 1

    def load(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen)
        if not self.board.is_valid():
            raise AssertionError("Invalid test fixture: {}".format(fen))
        self.check(self.cli.request("fen " + fen), "load")

    def push(self, move):
        parsed = chess.Move.from_uci(move)
        if parsed not in self.board.legal_moves:
            raise AssertionError("Illegal fixture move {} in {}".format(move, self.board.fen()))
        self.board.push(parsed)
        self.check(self.cli.request("push " + move), "push " + move)

    def pop(self):
        self.board.pop()
        self.check(self.cli.request("pop"), "pop")

    def undo_all(self):
        while self.board.move_stack:
            self.pop()

    def fixtures(self):
        # Pins, double check, castling attacks, legal and discovered-check EP,
        # promotion choices, rule counters, and conservative material detection.
        fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
            "4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1",
            "4kr2/8/8/8/8/8/8/4K2R w K - 0 1",
            "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",
            "4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1",
            "4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1",
            "7k/8/8/8/8/8/8/K7 w - - 0 1",
            "7k/8/8/8/8/8/8/KB6 w - - 0 1",
            "7k/8/8/8/8/8/8/KN6 w - - 0 1",
            "7k/8/8/8/8/8/8/KNN5 w - - 0 1",
            "7k/8/8/8/8/8/8/KBN5 w - - 0 1",
            "7k/8/8/8/8/8/8/KR6 w - - 99 1",
            "7k/8/8/8/8/8/8/KR6 w - - 100 1",
            "7k/8/8/8/8/8/8/KR6 w - - 149 1",
            "7k/8/8/8/8/8/8/KR6 w - - 150 1",
            "7k/6Q1/5K2/8/8/8/8/8 b - - 150 1",
            "7k/5K2/6Q1/8/8/8/8/8 b - - 150 1",
        ]
        for fen in fens:
            self.load(fen)
            # Exercise every legal special move and restore the exact state.
            for move in list(self.board.legal_moves):
                self.push(move.uci())
                self.pop()

        self.load()
        self.check(self.cli.request("start"), "start")
        for _ in range(4):
            for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
                self.push(move)
        self.undo_all()

        # A pawn double step keeps the raw EP square even when no capture exists.
        self.push("e2e4")
        self.undo_all()

        # Quiet move at 99 plies allows a claim; a pawn move resets the clock.
        self.load("7k/8/8/8/8/8/P7/KR6 w - - 99 1")
        self.push("a2a3")
        self.undo_all()

        # Capturing the original rook removes its castling right.
        self.load("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.push("a1a8")
        self.undo_all()

    def perft(self):
        fixtures = [
            (chess.STARTING_FEN, 3, 8902),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812),
        ]
        for fen, depth, nodes in fixtures:
            self.load(fen)
            response = self.cli.request("perft {}".format(depth))
            if response != {"nodes": nodes}:
                raise AssertionError("perft {}: expected {}, got {}".format(fen, nodes, response))
            self.check(self.cli.request("state"), "state after perft")

    def errors(self):
        self.load()
        for command in ("push e2e5", "push z9z0", "pop", "perft -1", "fen invalid"):
            response = self.cli.request(command)
            if not isinstance(response.get("error"), str) or not response["error"]:
                raise AssertionError("Expected error for {!r}, got {!r}".format(command, response))
            self.check(self.cli.request("state"), "state after rejected " + command)

    def random_games(self, seed, games, plies):
        rng = random.Random(seed)
        for game in range(games):
            self.load()
            for ply in range(plies):
                if self.board.is_game_over(claim_draw=False):
                    break
                moves = sorted(self.board.legal_moves, key=lambda move: move.uci())
                self.push(rng.choice(moves).uci())
            self.undo_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", help="path to inezita-rules")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--plies", type=int, default=80)
    args = parser.parse_args()
    if chess.__version__ != "1.11.2":
        parser.error("Expected chess==1.11.2, found {}".format(chess.__version__))
    if args.games < 0 or args.plies < 0:
        parser.error("--games and --plies must be nonnegative")
    cli = RulesProcess(args.executable)
    try:
        tests = DifferentialTests(cli)
        tests.fixtures()
        tests.perft()
        tests.errors()
        tests.random_games(args.seed, args.games, args.plies)
        print("Differential oracle passed: {} snapshots, seed={}, games={}".format(
            tests.snapshots, args.seed, args.games))
    finally:
        cli.close()


if __name__ == "__main__":
    main()
