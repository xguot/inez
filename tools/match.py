#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Automated UCI match runner, tactical/endgame suite, and PUCT benchmark tool.

Runs paired-opening matches with reversed colors, tracks crashes/illegal moves/
timeouts, reports 95% Elo confidence intervals and SPRT log-likelihood ratios,
evaluates a tactical/endgame test suite, and measures nodes/sec, batch
utilization, and leaf-evaluation latency across fixed-node and timed runs.
"""

import argparse
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import chess
except ImportError:
    sys.exit("Install python-chess with: python3 -m pip install -r tests/requirements.txt")


PAIRED_OPENINGS: List[Tuple[str, str]] = [
    ("startpos", chess.STARTING_FEN),
    ("italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("sicilian_open", "r1bqkbnr/pp2pppp/2np4/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5"),
    ("french_advance", "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"),
    ("caro_kann", "rn1qkbnr/pp2pppp/2p5/5b2/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 1 5"),
    ("queens_gambit", "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3"),
    ("slav_defense", "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3"),
    ("english_symmetrical", "r1bqkbnr/pppp1ppp/2n5/4p3/2P5/2N3P1/PP1PPP1P/R1BQKBNR b KQkq - 0 3"),
]

TACTICAL_ENDGAME_SUITE: List[Tuple[str, str, List[str]]] = [
    ("back_rank_mate_1", "6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1", ["d1d8"]),
    ("queen_mate_1", "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1", ["g5g7"]),
    ("smothered_mate_1", "6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1", ["h6f7"]),
    ("free_queen_capture", "4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1", ["c3d5"]),
    ("promote_with_mate", "7k/4P1pp/8/8/8/8/8/6K1 w - - 0 1", ["e7e8q", "e7e8r"]),
    ("avoid_stalemate_mate_1", "7k/5K2/8/8/6Q1/8/8/8 w - - 0 1", ["g4g7"]),
]

DIAG_REGEX = re.compile(
    r"diag nodes=(\d+) depth=(\d+) nps=([0-9.]+) batches=(\d+) "
    r"leaves=(\d+) avg_batch=([0-9.]+) batch_util=([0-9.]+) eval_us=([0-9.]+)"
)


@dataclass
class DiagSample:
    nodes: int = 0
    depth: int = 0
    nps: float = 0.0
    batches: int = 0
    leaves: int = 0
    avg_batch: float = 0.0
    batch_util: float = 0.0
    eval_us: float = 0.0


@dataclass
class EngineSpec:
    name: str
    executable: str
    options: Dict[str, str] = field(default_factory=dict)


class UciProcess:
    def __init__(self, spec: EngineSpec):
        self.spec = spec
        env = os.environ.copy()
        if "Evaluator" in spec.options:
            env["INEZ_BASELINE"] = "1"
        self.process = subprocess.Popen(
            [spec.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.out_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.diag_queue: "queue.Queue[DiagSample]" = queue.Queue()
        self._out_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._err_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._out_thread.start()
        self._err_thread.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            self.out_queue.put(line.rstrip("\r\n"))
        self.out_queue.put(None)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        while True:
            line = self.process.stderr.readline()
            if not line:
                break
            match = DIAG_REGEX.search(line)
            if match:
                sample = DiagSample(
                    nodes=int(match.group(1)),
                    depth=int(match.group(2)),
                    nps=float(match.group(3)),
                    batches=int(match.group(4)),
                    leaves=int(match.group(5)),
                    avg_batch=float(match.group(6)),
                    batch_util=float(match.group(7)),
                    eval_us=float(match.group(8)),
                )
                self.diag_queue.put(sample)

    def clear_diags(self) -> None:
        while True:
            try:
                self.diag_queue.get_nowait()
            except queue.Empty:
                break

    def wait_for_diag(self, timeout: float = 5.0) -> Optional[DiagSample]:
        try:
            return self.diag_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"engine {self.spec.name} crashed before {command!r}")
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, prefix: str, timeout: float) -> Tuple[Optional[str], List[str], str]:
        """Returns (matching_line, prior_lines, status) where status is ok|timeout|crash."""
        deadline = time.monotonic() + timeout
        lines: List[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, lines, "timeout"
            try:
                item = self.out_queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                if self.process.poll() is not None and self.out_queue.empty():
                    return None, lines, "crash"
                continue
            if item is None:
                return None, lines, "crash"
            stripped = item.strip()
            if stripped.startswith(prefix):
                return stripped, lines, "ok"
            lines.append(stripped)

    def initialize(self, timeout: float = 5.0) -> None:
        self.send("uci")
        line, _, status = self.read_until("uciok", timeout)
        if status != "ok" or line is None:
            raise RuntimeError(f"engine {self.spec.name} failed uci handshake ({status})")
        for key, value in self.spec.options.items():
            self.send(f"setoption name {key} value {value}")
        self.send("isready")
        line, _, status = self.read_until("readyok", timeout)
        if status != "ok" or line is None:
            raise RuntimeError(f"engine {self.spec.name} failed isready ({status})")

    def new_game(self, timeout: float = 5.0) -> None:
        self.send("ucinewgame")
        self.send("isready")
        line, _, status = self.read_until("readyok", timeout)
        if status != "ok" or line is None:
            raise RuntimeError(f"engine {self.spec.name} failed ucinewgame ({status})")

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=3.0)
            except Exception:
                self.process.kill()
                self.process.wait()
        self._out_thread.join(timeout=1.0)
        self._err_thread.join(timeout=1.0)
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))


def score_to_elo(score: float) -> float:
    clamped = min(max(score, 1e-4), 1.0 - 1e-4)
    return -400.0 * math.log10(1.0 / clamped - 1.0)


def compute_elo_and_ci(wins: int, draws: int, losses: int) -> Dict[str, Optional[float]]:
    total = wins + draws + losses
    if total == 0:
        return {"score": None, "elo_diff": None, "elo_ci95": None}
    score = (wins + 0.5 * draws) / total
    if total < 2:
        return {"score": round(score, 4), "elo_diff": round(score_to_elo(score), 2), "elo_ci95": None}
    w_p = wins / total
    d_p = draws / total
    l_p = losses / total
    variance = (
        w_p * (1.0 - score) ** 2
        + d_p * (0.5 - score) ** 2
        + l_p * (0.0 - score) ** 2
    ) / max(1, total - 1)
    std_err = math.sqrt(max(variance, 1e-9))
    low = max(1e-4, min(1.0 - 1e-4, score - 1.96 * std_err))
    high = max(1e-4, min(1.0 - 1e-4, score + 1.96 * std_err))
    elo = score_to_elo(score)
    ci95 = 0.5 * abs(score_to_elo(high) - score_to_elo(low))
    return {
        "score": round(score, 4),
        "elo_diff": round(elo, 2),
        "elo_ci95": round(ci95, 2),
    }


def compute_sprt(
    wins: int,
    draws: int,
    losses: int,
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
) -> Dict[str, object]:
    total = wins + draws + losses
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    if total == 0:
        return {
            "elo0": elo0,
            "elo1": elo1,
            "llr": 0.0,
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "status": "continue",
        }
    s = (wins + 0.5 * draws) / total
    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)
    w_p = wins / total
    d_p = draws / total
    l_p = losses / total
    var = (
        w_p * (1.0 - s) ** 2
        + d_p * (0.5 - s) ** 2
        + l_p * (0.0 - s) ** 2
    )
    var = max(var, 0.05)
    llr = total * (s1 - s0) * (2.0 * s - s0 - s1) / (2.0 * var)
    status = "continue"
    if llr >= upper:
        status = "H1_accepted"
    elif llr <= lower:
        status = "H0_accepted"
    return {
        "elo0": elo0,
        "elo1": elo1,
        "llr": round(llr, 4),
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "status": status,
    }


def play_single_game(
    white: UciProcess,
    black: UciProcess,
    start_fen: str,
    nodes: Optional[int],
    movetime_ms: Optional[int],
    max_plies: int,
    move_timeout_s: float,
) -> Tuple[float, str, Dict[str, Dict[str, int]]]:
    """Plays one game. Returns (white_score, reason, fault_counts)."""
    faults = {
        "white": {"crashes": 0, "illegal_moves": 0, "timeouts": 0},
        "black": {"crashes": 0, "illegal_moves": 0, "timeouts": 0},
    }
    white.new_game()
    black.new_game()

    board = chess.Board(start_fen)
    moves_played: List[str] = []

    for _ in range(max_plies):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0.5, outcome.termination.name.lower(), faults
            return (1.0 if outcome.winner == chess.WHITE else 0.0), outcome.termination.name.lower(), faults

        side_key = "white" if board.turn == chess.WHITE else "black"
        engine = white if board.turn == chess.WHITE else black

        pos_cmd = (
            "position startpos"
            if start_fen == chess.STARTING_FEN
            else f"position fen {start_fen}"
        )
        if moves_played:
            pos_cmd += " moves " + " ".join(moves_played)
        engine.send(pos_cmd)

        if nodes is not None:
            engine.send(f"go nodes {nodes}")
        elif movetime_ms is not None:
            engine.send(f"go movetime {movetime_ms}")
        else:
            engine.send("go nodes 64")

        best_line, _, status = engine.read_until("bestmove ", move_timeout_s)
        if status == "timeout":
            faults[side_key]["timeouts"] += 1
            return (0.0 if side_key == "white" else 1.0), f"{side_key}_timeout", faults
        if status == "crash" or best_line is None:
            faults[side_key]["crashes"] += 1
            return (0.0 if side_key == "white" else 1.0), f"{side_key}_crash", faults

        parts = best_line.split()
        uci_move = parts[1] if len(parts) >= 2 else "0000"
        try:
            move = chess.Move.from_uci(uci_move)
        except ValueError:
            faults[side_key]["illegal_moves"] += 1
            return (0.0 if side_key == "white" else 1.0), f"{side_key}_illegal_move", faults

        if move not in board.legal_moves:
            faults[side_key]["illegal_moves"] += 1
            return (0.0 if side_key == "white" else 1.0), f"{side_key}_illegal_move", faults

        board.push(move)
        moves_played.append(uci_move)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None and outcome.winner is not None:
        return (1.0 if outcome.winner == chess.WHITE else 0.0), outcome.termination.name.lower(), faults
    return 0.5, "max_plies_adjudicated_draw", faults


def run_tactical_suite(executable: str, nodes: int, batch_size: int) -> Dict[str, object]:
    spec = EngineSpec(
        name="candidate",
        executable=executable,
        options={"Evaluator": "material", "BatchSize": str(batch_size)},
    )
    proc = UciProcess(spec)
    try:
        proc.initialize()
        results = []
        solved = 0
        for name, fen, expected in TACTICAL_ENDGAME_SUITE:
            proc.new_game()
            proc.send(f"position fen {fen}")
            proc.send(f"go nodes {nodes}")
            line, _, status = proc.read_until("bestmove ", 10.0)
            if status != "ok" or line is None:
                raise RuntimeError(f"tactical test {name} failed with {status}")
            parts = line.split()
            move = parts[1] if len(parts) >= 2 else "0000"
            ok = move in expected
            if ok:
                solved += 1
            results.append({"name": name, "move": move, "expected": expected, "solved": ok})
        return {
            "solved": solved,
            "total": len(TACTICAL_ENDGAME_SUITE),
            "solve_rate": round(solved / len(TACTICAL_ENDGAME_SUITE), 4),
            "positions": results,
        }
    finally:
        proc.close()


def run_diagnostics_benchmark(executable: str, nodes: int, movetime_ms: int, batch_size: int) -> Dict[str, object]:
    results = {}
    for mode, go_cmd, cfg_batch in [
        ("fixed_nodes_serial", f"go nodes {nodes}", 1),
        ("fixed_nodes_batched", f"go nodes {nodes}", batch_size),
        ("timed_movetime_batched", f"go movetime {movetime_ms}", batch_size),
    ]:
        spec = EngineSpec(
            name=mode,
            executable=executable,
            options={"Evaluator": "material", "BatchSize": str(cfg_batch)},
        )
        proc = UciProcess(spec)
        try:
            proc.initialize()
            proc.new_game()
            proc.clear_diags()
            proc.send("position startpos")
            proc.send(go_cmd)
            line, _, status = proc.read_until("bestmove ", 15.0)
            sample = proc.wait_for_diag(timeout=5.0)
            if status != "ok" or line is None or sample is None:
                raise RuntimeError(f"benchmark {mode} failed ({status})")
            results[mode] = {
                "bestmove": line.split()[1],
                "nodes": sample.nodes,
                "max_depth": sample.depth,
                "nodes_per_sec": round(sample.nps, 2),
                "batches": sample.batches,
                "batch_leaves": sample.leaves,
                "avg_batch_size": round(sample.avg_batch, 2),
                "batch_utilization": round(sample.batch_util, 4),
                "avg_eval_latency_us": round(sample.eval_us, 2),
            }
        finally:
            proc.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", help="path to inezita UCI executable")
    parser.add_argument("--pairs", type=int, default=4, help="number of paired openings (2 games per pair)")
    parser.add_argument("--nodes", type=int, default=64, help="fixed nodes per move in match games")
    parser.add_argument("--movetime-ms", type=int, default=None, help="optional fixed movetime (ms) instead of nodes")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for candidate and batched benchmarks")
    parser.add_argument("--max-plies", type=int, default=140, help="maximum plies per game before draw adjudication")
    parser.add_argument("--sprt-elo0", type=float, default=0.0, help="SPRT H0 Elo bound")
    parser.add_argument("--sprt-elo1", type=float, default=50.0, help="SPRT H1 Elo bound")
    parser.add_argument("--sprt-alpha", type=float, default=0.05, help="SPRT type-I error probability")
    parser.add_argument("--sprt-beta", type=float, default=0.05, help="SPRT type-II error probability")
    parser.add_argument("--sprt-stop", action="store_true", help="allow early stopping when SPRT crosses bound")
    parser.add_argument("--output-json", default=None, help="optional path to write JSON report")
    args = parser.parse_args()

    if args.pairs < 1:
        parser.error("--pairs must be at least 1")

    candidate_spec = EngineSpec(
        name="inezita-material",
        executable=args.executable,
        options={"Evaluator": "material", "BatchSize": str(args.batch_size)},
    )
    baseline_spec = EngineSpec(
        name="inezita-uniform",
        executable=args.executable,
        options={"Evaluator": "uniform", "BatchSize": "1"},
    )

    candidate = UciProcess(candidate_spec)
    baseline = UciProcess(baseline_spec)

    wins = 0
    draws = 0
    losses = 0
    fault_totals = {
        "candidate": {"crashes": 0, "illegal_moves": 0, "timeouts": 0},
        "baseline": {"crashes": 0, "illegal_moves": 0, "timeouts": 0},
    }
    game_records = []

    try:
        candidate.initialize()
        baseline.initialize()

        for pair_idx in range(args.pairs):
            opening_name, fen = PAIRED_OPENINGS[pair_idx % len(PAIRED_OPENINGS)]
            for candidate_is_white in (True, False):
                white_proc = candidate if candidate_is_white else baseline
                black_proc = baseline if candidate_is_white else candidate
                w_score, reason, faults = play_single_game(
                    white=white_proc,
                    black=black_proc,
                    start_fen=fen,
                    nodes=None if args.movetime_ms is not None else args.nodes,
                    movetime_ms=args.movetime_ms,
                    max_plies=args.max_plies,
                    move_timeout_s=15.0,
                )

                cand_faults = faults["white"] if candidate_is_white else faults["black"]
                base_faults = faults["black"] if candidate_is_white else faults["white"]
                for k in ("crashes", "illegal_moves", "timeouts"):
                    fault_totals["candidate"][k] += cand_faults[k]
                    fault_totals["baseline"][k] += base_faults[k]

                cand_score = w_score if candidate_is_white else (1.0 - w_score)
                if cand_score == 1.0:
                    wins += 1
                elif cand_score == 0.0:
                    losses += 1
                else:
                    draws += 1

                game_records.append({
                    "pair": pair_idx + 1,
                    "game": len(game_records) + 1,
                    "opening": opening_name,
                    "candidate_color": "white" if candidate_is_white else "black",
                    "candidate_score": cand_score,
                    "reason": reason,
                })

            sprt_now = compute_sprt(
                wins, draws, losses,
                args.sprt_elo0, args.sprt_elo1,
                args.sprt_alpha, args.sprt_beta,
            )
            if args.sprt_stop and sprt_now["status"] != "continue":
                break
    finally:
        candidate.close()
        baseline.close()

    tactics = run_tactical_suite(args.executable, max(80, args.nodes), args.batch_size)
    benchmarks = run_diagnostics_benchmark(
        args.executable,
        nodes=max(128, args.nodes),
        movetime_ms=40,
        batch_size=max(4, args.batch_size),
    )

    stats = compute_elo_and_ci(wins, draws, losses)
    sprt = compute_sprt(
        wins, draws, losses,
        args.sprt_elo0, args.sprt_elo1,
        args.sprt_alpha, args.sprt_beta,
    )

    report = {
        "match": {
            "candidate": candidate_spec.name,
            "baseline": baseline_spec.name,
            "predeclared_pairs": args.pairs,
            "games_played": wins + draws + losses,
            "nodes_per_move": None if args.movetime_ms is not None else args.nodes,
            "movetime_ms": args.movetime_ms,
            "batch_size": args.batch_size,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": stats["score"],
            "elo_diff": stats["elo_diff"],
            "elo_ci95": stats["elo_ci95"],
            "sprt": sprt,
            "faults": fault_totals,
            "games": game_records,
        },
        "tactical_suite": tactics,
        "diagnostics": benchmarks,
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))

    for side in ("candidate", "baseline"):
        for fault_key, count in fault_totals[side].items():
            if count != 0:
                raise SystemExit(f"Fault detected for {side}: {fault_key}={count}")
    if tactics["solved"] != tactics["total"]:
        raise SystemExit(
            f"Tactical suite failed: solved {tactics['solved']}/{tactics['total']}"
        )
    if (wins + 0.5 * draws) <= 0.5 * (wins + draws + losses):
        raise SystemExit(
            f"Candidate did not outscore uniform baseline: +{wins} ={draws} -{losses}"
        )


if __name__ == "__main__":
    main()
