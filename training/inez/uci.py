# SPDX-License-Identifier: GPL-3.0-only
"""UCI adapter for the trained neural model and reference PUCT search."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import chess
import torch

from .model import load_checkpoint
from .puct import NeuralEvaluator, choose_move, search

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / 'results/training/inezita'
DEFAULT_THREADS = min(4, os.cpu_count() or 1)
GO_NUMBERS = {'nodes', 'movetime', 'wtime', 'btime', 'winc', 'binc', 'movestogo', 'depth'}
GO_FLAGS = {'infinite', 'ponder', 'mate', 'searchmoves'}


def resolve_artifact(value):
    """Read accepted.json without rewriting paths in copied HPC results."""
    path = Path(value).expanduser().resolve()
    if path.is_dir() and (path / 'manifest.json').is_file():
        return path
    pointer = path / 'accepted.json' if path.is_dir() else path
    if pointer.name != 'accepted.json' or not pointer.is_file():
        raise ValueError(f'choose a model artifact directory or run with accepted.json: {path}')
    record = json.loads(pointer.read_text())
    artifact = record['artifact']
    if artifact not in record['history']:
        raise ValueError('accepted artifact is absent from the run history')
    saved, run = Path(artifact), pointer.parent
    candidates = []
    if not saved.is_absolute():
        candidates.append(run / saved)
    # Prefer the local copy of a run over its original cluster location.
    for index in reversed(range(len(saved.parts))):
        if saved.parts[index] == run.name:
            local = run.joinpath(*saved.parts[index + 1:]).resolve()
            if local.is_relative_to(run):
                candidates.append(local)
            break
    if saved.is_absolute():
        candidates.append(saved)
    for candidate in candidates:
        if (candidate / 'manifest.json').is_file():
            return candidate.resolve()
    raise ValueError(f'accepted artifact is missing locally: {artifact}; '
                     'copy the complete artifact or set ModelPath to its local directory')


def parse_position(tokens):
    """Build a new board transactionally, preserving the complete move history."""
    if not tokens:
        raise ValueError('position requires startpos or fen')
    if tokens[0] == 'startpos':
        board, index = chess.Board(), 1
    elif tokens[0] == 'fen' and len(tokens) >= 7:
        board, index = chess.Board(' '.join(tokens[1:7])), 7
        if not board.is_valid():
            raise ValueError('invalid chess position')
    else:
        raise ValueError('position requires startpos or a six-field FEN')
    if index < len(tokens):
        if tokens[index] != 'moves':
            raise ValueError('unexpected position arguments')
        for notation in tokens[index + 1:]:
            move = chess.Move.from_uci(notation)
            if move not in board.legal_moves:
                raise ValueError(f'illegal position move: {notation}')
            board.push(move)
    return board


@dataclass(frozen=True)
class Limits:
    nodes: int | None
    seconds: float | None
    depth: int | None
    infinite: bool
    searchmoves: tuple | None


def parse_go(tokens, board, *, simulations, overhead):
    numbers, moves, infinite = {}, None, False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token in GO_NUMBERS:
            if index == len(tokens):
                raise ValueError(f'missing go {token} value')
            value = int(tokens[index])
            if not 0 <= value <= 2**63 - 1 or token in ('movestogo', 'depth') and value == 0:
                raise ValueError(f'invalid go {token} value')
            numbers[token] = value
            index += 1
        elif token == 'infinite':
            infinite = True
        elif token in ('ponder', 'mate'):
            raise ValueError(f'go {token} is unsupported; use time, nodes, depth or infinite')
        elif token == 'searchmoves':
            moves = []
            while index < len(tokens) and tokens[index] not in GO_NUMBERS | GO_FLAGS:
                move = chess.Move.from_uci(tokens[index])
                if move not in board.legal_moves:
                    raise ValueError(f'illegal searchmove: {tokens[index]}')
                moves.append(move.uci())
                index += 1
            if not moves:
                raise ValueError('searchmoves must not be empty')
    seconds = None
    side = 'w' if board.turn == chess.WHITE else 'b'
    if 'movetime' in numbers:
        seconds = max(0, numbers['movetime'] - overhead) / 1000
    elif side + 'time' in numbers:
        remaining = numbers[side + 'time']
        available = max(0, remaining - overhead)
        allocation = remaining / numbers.get('movestogo', 30) + 0.8 * numbers.get(side + 'inc', 0)
        seconds = min(available, allocation) / 1000
    nodes, depth = numbers.get('nodes'), numbers.get('depth')
    if infinite:
        nodes, seconds, depth = None, None, None
    elif nodes is None and seconds is None and depth is None:
        nodes = simulations
    return Limits(nodes, seconds, depth, infinite, tuple(dict.fromkeys(moves)) if moves is not None else None)


class NeuralUci:
    def __init__(self, *, model_path=DEFAULT_MODEL, device='cpu', threads=DEFAULT_THREADS,
                 simulations=128, cpuct=2.5, overhead=50, output=None, log=None):
        self.model_path, self.device = str(model_path), device
        self.threads, self.simulations, self.cpuct, self.overhead = threads, simulations, cpuct, overhead
        self.output, self.log = output or sys.stdout, log or sys.stderr
        self.board = chess.Board()
        self.evaluator = None
        self.worker = None
        self.cancel = threading.Event()
        self.output_lock = threading.Lock()
        self.failure = None

    def write(self, line):
        with self.output_lock:
            print(line, file=self.output, flush=True)

    def diagnostic(self, message):
        print(f'Inezita: {message}', file=self.log, flush=True)

    def ensure_model(self):
        if self.evaluator is not None:
            return
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable')
        torch.set_num_threads(self.threads)
        artifact = resolve_artifact(self.model_path)
        model = load_checkpoint(artifact, device=self.device)
        self.evaluator = NeuralEvaluator(model)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        self.diagnostic(f'loaded {artifact}; parameters={parameters}; device={self.evaluator.device}')

    def stop(self):
        self.cancel.set()
        if self.worker is not None:
            self.worker.join()
            self.worker = None

    def start_search(self, limits):
        self.stop()
        self.ensure_model()
        board = self.board.copy(stack=True)
        self.cancel.clear()
        self.worker = threading.Thread(target=self.think, args=(board, limits), name='inezita-search')
        self.worker.start()

    def think(self, board, limits):
        started = time.monotonic()
        deadline = started + limits.seconds if limits.seconds is not None else None
        def stopped():
            return self.cancel.is_set() or deadline is not None and time.monotonic() >= deadline
        def progress(info):
            elapsed = time.monotonic() - started
            line = (f'info depth {info.depth} nodes {info.nodes} time {int(elapsed * 1000)} '
                    f'nps {int(info.nodes / max(elapsed, 1e-9))}')
            if info.value is not None:
                # Same display mapping as the native engine; this is not calibrated Elo.
                cp = round(600 * math.atanh(max(-0.999, min(0.999, info.value))))
                line += f' score cp {cp}'
            if info.pv:
                line += ' pv ' + ' '.join(info.pv)
            self.write(line)
        try:
            visits = search(board, self.evaluator, simulations=limits.nodes or None,
                cpuct=self.cpuct, stop=lambda: stopped() or limits.nodes == 0,
                on_update=progress, searchmoves=limits.searchmoves, depth_limit=limits.depth)
            if limits.infinite:
                # A terminal root has no tree to search, but infinite still waits for stop.
                self.cancel.wait()
            if board.is_game_over(claim_draw=False):
                move = '0000'
            elif sum(visits.values()):
                move = choose_move(visits)
            else:
                # Immediate stop / exhausted clock: return a legal move without pretending
                # an unevaluated position has a neural score.
                move = next(iter(visits))
            self.write('bestmove ' + move)
        except Exception as error:
            self.failure = error
            self.diagnostic(f'neural search failed: {error}')
            self.write('info string ERROR neural search failed; see engine stderr')
            self.write('bestmove 0000')

    def set_option(self, tokens):
        if not tokens or tokens[0].lower() != 'name':
            raise ValueError('setoption requires name')
        split = next((i for i, token in enumerate(tokens) if token.lower() == 'value'), len(tokens))
        name = ' '.join(tokens[1:split]).lower()
        value = ' '.join(tokens[split + 1:])
        integer_options = {'threads': ('threads', 1, 64),
                           'simulations': ('simulations', 1, 10_000_000),
                           'moveoverhead': ('overhead', 0, 5000)}
        if name in integer_options:
            attribute, minimum, maximum = integer_options[name]
            number = int(value)
            if not minimum <= number <= maximum:
                raise ValueError(f'{name} must be in {minimum}..{maximum}')
            self.stop()
            setattr(self, attribute, number)
            if name == 'threads':
                torch.set_num_threads(number)
        elif name == 'cpuct':
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError('Cpuct must be finite and positive')
            self.stop()
            self.cpuct = number
        elif name in ('modelpath', 'device'):
            if not value or name == 'device' and value.lower() not in ('cpu', 'mps', 'cuda'):
                raise ValueError(f'invalid {name}')
            self.stop()
            setattr(self, 'model_path' if name == 'modelpath' else 'device',
                    value if name == 'modelpath' else value.lower())
            self.evaluator = None
        else:
            self.diagnostic(f'ignored unsupported option: {name}')

    def handle(self, line):
        if self.failure is not None:
            raise RuntimeError(f'previous neural search failed: {self.failure}')
        tokens = line.split()
        if not tokens:
            return True
        command, args = tokens[0], tokens[1:]
        if command == 'uci':
            self.write('id name Inezita 0.1.1')
            self.write('id author Inezita Developers')
            self.write(f'option name ModelPath type string default {self.model_path}')
            self.write(f'option name Device type combo default {self.device} var cpu var mps var cuda')
            self.write(f'option name Threads type spin default {self.threads} min 1 max 64')
            self.write(f'option name Simulations type spin default {self.simulations} min 1 max 10000000')
            self.write(f'option name Cpuct type string default {self.cpuct}')
            self.write(f'option name MoveOverhead type spin default {self.overhead} min 0 max 5000')
            self.write('uciok')
        elif command == 'isready':
            self.ensure_model()
            self.write('readyok')
        elif command == 'setoption':
            self.set_option(args)
        elif command == 'ucinewgame':
            self.stop()
            self.board = chess.Board()
        elif command == 'position':
            board = parse_position(args)
            self.stop()
            self.board = board
        elif command == 'go':
            self.start_search(parse_go(args, self.board, simulations=self.simulations, overhead=self.overhead))
        elif command == 'stop':
            self.stop()
        elif command == 'quit':
            self.stop()
            return False
        return True

    def run(self, stream):
        try:
            for line in stream:
                try:
                    if not self.handle(line):
                        break
                except ValueError as error:
                    if line.split()[:1] in (['isready'], ['go']):
                        raise RuntimeError(str(error)) from error
                    self.diagnostic(str(error))
        finally:
            self.stop()
        if self.failure is not None:
            raise RuntimeError(f'neural search failed: {self.failure}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=os.environ.get('INEZ_MODEL', str(DEFAULT_MODEL)),
                        help='artifact directory, training run directory or accepted.json')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    parser.add_argument('--simulations', type=int, default=128)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 64 or not 1 <= args.simulations <= 10_000_000:
        parser.error('threads must be 1..64 and simulations must be 1..10000000')
    engine = NeuralUci(model_path=args.model, device=args.device,
                       threads=args.threads, simulations=args.simulations)
    try:
        engine.run(sys.stdin)
    except (Exception, KeyboardInterrupt) as error:
        engine.diagnostic(str(error))
        engine.write('info string ERROR initialization or search failed; see engine stderr')
        return 1
    return 0
