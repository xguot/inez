#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Neural UCI protocol, cancellation, copied artifacts and real client integration."""

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'tools'))

import chess
import chess.engine
import torch

from inez.model import ChessTransformer, ModelConfig, save_checkpoint
from inez.puct import search
from inez.uci import NeuralUci, parse_go, parse_position, resolve_artifact
from match import EngineSpec, UciProcess


def uniform(board):
    moves = list(board.legal_moves)
    return {move.uci(): 1 / len(moves) for move in moves}, 0.0


class ParserTests(unittest.TestCase):
    def test_positions_preserve_history_and_reject_null_or_illegal_moves(self):
        board = parse_position('startpos moves g1f3 g8f6 f3g1 f6g8'.split())
        self.assertEqual(len(board.move_stack), 4)
        self.assertTrue(board.is_repetition(2))
        fen = '4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1'
        board = parse_position(('fen ' + fen + ' moves g7h8n').split())
        self.assertEqual(board.piece_at(chess.H8), chess.Piece(chess.KNIGHT, chess.WHITE))
        for text in ('startpos moves e2e5', 'startpos moves 0000', 'startpos junk',
                     'fen 8/8/8/8/8/8/8/8 w - - 0 1'):
            with self.assertRaises(ValueError, msg=text):
                parse_position(text.split())

    def test_clock_allocation_uses_moving_side_and_reserves_overhead(self):
        board = chess.Board()
        limits = parse_go('wtime 30000 btime 200 winc 1000 binc 0 movestogo 20'.split(),
                          board, simulations=128, overhead=50)
        self.assertAlmostEqual(limits.seconds, 2.3)
        self.assertIsNone(limits.nodes)
        board.push_uci('e2e4')
        limits = parse_go('wtime 30000 btime 200 winc 1000 binc 0 movestogo 20'.split(),
                          board, simulations=128, overhead=50)
        self.assertAlmostEqual(limits.seconds, 0.01)
        limits = parse_go('btime 20 binc 1000'.split(), board, simulations=128, overhead=50)
        self.assertEqual(limits.seconds, 0)

    def test_go_limits_and_restricted_moves(self):
        board = chess.Board()
        kwargs = dict(simulations=128, overhead=50)
        self.assertEqual(parse_go([], board, **kwargs).nodes, 128)
        limits = parse_go('nodes 9 movetime 250 searchmoves e2e4 d2d4 e2e4'.split(), board, **kwargs)
        self.assertEqual(limits.nodes, 9)
        self.assertAlmostEqual(limits.seconds, 0.2)
        self.assertEqual(limits.searchmoves, ('e2e4', 'd2d4'))
        self.assertEqual(parse_go('depth 3'.split(), board, **kwargs).depth, 3)
        limits = parse_go('infinite nodes 1 movetime 10 depth 1'.split(), board, **kwargs)
        self.assertTrue(limits.infinite)
        self.assertEqual((limits.nodes, limits.seconds, limits.depth), (None, None, None))
        for text in ('nodes -1', 'nodes', 'depth 0', 'movestogo 0', 'searchmoves e2e5',
                     'searchmoves', 'ponder', 'mate 1'):
            with self.assertRaises(ValueError, msg=text):
                parse_go(text.split(), board, **kwargs)

    def test_bad_position_and_options_leave_existing_state_unchanged(self):
        engine = NeuralUci(output=io.StringIO(), log=io.StringIO())
        engine.handle('position startpos moves e2e4')
        before = engine.board.fen()
        with self.assertRaises(ValueError):
            engine.handle('position startpos moves e2e5')
        self.assertEqual(engine.board.fen(), before)
        for value in ('nan', 'inf', '0', '-1'):
            with self.assertRaises(ValueError):
                engine.handle('setoption name Cpuct value ' + value)
        self.assertEqual(engine.cpuct, 2.5)


class SearchControlTests(unittest.TestCase):
    def test_interrupt_before_evaluation_does_not_invent_visits_or_value(self):
        updates = []
        visits = search(chess.Board(), lambda board: self.fail('unexpected evaluation'),
                        simulations=None, stop=lambda: True, on_update=updates.append)
        self.assertEqual(sum(visits.values()), 0)
        self.assertEqual(updates[-1].nodes, 0)
        self.assertIsNone(updates[-1].value)

    def test_depth_restrictions_and_progress_are_real_tree_statistics(self):
        board, updates = chess.Board(), []
        visits = search(board, uniform, simulations=None, depth_limit=2,
                        searchmoves=('e2e4',), on_update=updates.append, update_interval=1)
        self.assertEqual(set(visits), {'e2e4'})
        self.assertEqual(sum(visits.values()), updates[-1].nodes)
        self.assertEqual(updates[-1].depth, 2)
        self.assertEqual(updates[-1].pv[0], 'e2e4')
        cursor = board.copy(stack=True)
        for move in updates[-1].pv:
            self.assertIn(chess.Move.from_uci(move), cursor.legal_moves)
            cursor.push_uci(move)
        self.assertEqual(board.fen(), chess.STARTING_FEN)

    def test_stop_and_readiness_remain_responsive_during_a_forward_pass(self):
        output, log = io.StringIO(), io.StringIO()
        entered, release = threading.Event(), threading.Event()
        def slow(board):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('test forward pass not released')
            return uniform(board)
        engine = NeuralUci(output=output, log=log)
        engine.evaluator = slow
        try:
            engine.handle('go infinite')
            self.assertTrue(entered.wait(2))
            engine.handle('isready')
            self.assertIn('readyok', output.getvalue())
            self.assertNotIn('bestmove', output.getvalue())
            engine.cancel.set()
            release.set()
            engine.handle('stop')
            self.assertEqual(output.getvalue().count('bestmove '), 1)
        finally:
            release.set()
            engine.stop()

    def test_neural_backend_failure_is_explicit_and_blocks_further_search(self):
        output, log = io.StringIO(), io.StringIO()
        engine = NeuralUci(output=output, log=log)
        def broken(board):
            raise RuntimeError('backend failed')
        engine.evaluator = broken
        engine.handle('go nodes 8')
        engine.stop()
        self.assertIn('ERROR neural search failed', output.getvalue())
        self.assertEqual(output.getvalue().count('bestmove 0000'), 1)
        self.assertIn('backend failed', log.getvalue())
        with self.assertRaisesRegex(RuntimeError, 'previous neural search failed'):
            engine.handle('go nodes 8')


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.artifact = cls.root / 'model with spaces'
        torch.manual_seed(23)
        model = ChessTransformer(ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4))
        save_checkpoint(model, cls.artifact)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def raw_client(self):
        spec = EngineSpec('neural', str(ROOT / 'inez-neural'),
                          {'ModelPath': str(self.artifact), 'Threads': '1'})
        with patch.dict(os.environ, {'INEZ_PYTHON': sys.executable}):
            client = UciProcess(spec)
        self.addCleanup(client.close)
        client.initialize(timeout=15)
        return client

    def test_copied_cluster_pointer_resolves_locally_without_edits(self):
        run = self.root / 'copied-pilot'
        run.mkdir(exist_ok=True)
        local = run / 'teacher/candidate-0001000'
        if not local.exists():
            local.parent.mkdir()
            shutil.copytree(self.artifact, local)
        remote = '/sfs/weka/scratch/user/inez/results/training/copied-pilot/teacher/candidate-0001000'
        pointer = run / 'accepted.json'
        content = json.dumps({'artifact': remote, 'history': [remote]})
        pointer.write_text(content)
        self.assertEqual(resolve_artifact(run), local.resolve())
        self.assertEqual(resolve_artifact(pointer), local.resolve())
        self.assertEqual(pointer.read_text(), content)
        with patch.dict(os.environ, {'INEZ_PYTHON': sys.executable}):
            with chess.engine.SimpleEngine.popen_uci(
                    [str(ROOT / 'inez-neural'), '--model', str(run), '--threads', '1'], timeout=15) as engine:
                result = engine.play(chess.Board(), chess.engine.Limit(nodes=4))
                self.assertIn(result.move, chess.Board().legal_moves)

    def test_python_chess_client_can_play_both_colors_and_restrict_root_moves(self):
        with chess.engine.SimpleEngine.popen_uci(
                [sys.executable, str(ROOT / 'tools/neural_uci.py'), '--model', str(self.artifact),
                 '--threads', '1'], timeout=15) as engine:
            self.assertIn('Inez Neural', engine.id['name'])
            self.assertNotIn('Evaluator', engine.options)
            board = chess.Board()
            for move in ('e2e4', 'c7c5'):
                result = engine.play(board, chess.engine.Limit(nodes=8),
                                     root_moves=[chess.Move.from_uci(move)], info=chess.engine.INFO_ALL)
                self.assertEqual(result.move.uci(), move)
                self.assertEqual(result.info['nodes'], 8)
                self.assertTrue(result.info['pv'])
                board.push(result.move)
            result = engine.play(board, chess.engine.Limit(depth=2), info=chess.engine.INFO_ALL)
            self.assertIn(result.move, board.legal_moves)
            self.assertEqual(result.info['depth'], 2)

    def test_infinite_ready_stop_and_repeated_search_have_one_bestmove_each(self):
        client = self.raw_client()
        for moves in ('', ' moves e2e4'):
            board = chess.Board()
            if moves:
                board.push_uci('e2e4')
            client.send('position startpos' + moves)
            client.send('go infinite')
            _, prior, status = client.read_until('info ', 5)
            self.assertEqual(status, 'ok')
            self.assertFalse(any(line.startswith('bestmove') for line in prior))
            client.send('isready')
            _, prior, status = client.read_until('readyok', 5)
            self.assertEqual(status, 'ok')
            self.assertFalse(any(line.startswith('bestmove') for line in prior))
            client.send('stop')
            bestmove, _, status = client.read_until('bestmove ', 5)
            self.assertEqual(status, 'ok')
            self.assertIn(chess.Move.from_uci(bestmove.split()[1]), board.legal_moves)
            client.send('isready')
            _, prior, status = client.read_until('readyok', 5)
            self.assertEqual(status, 'ok')
            self.assertFalse(any(line.startswith('bestmove') for line in prior))

    def test_clock_search_zero_budget_terminal_and_quit(self):
        client = self.raw_client()
        for command in ('go movetime 200', 'go nodes 0', 'go wtime 0 btime 10000'):
            started = time.monotonic()
            client.send(command)
            bestmove, _, status = client.read_until('bestmove ', 5)
            self.assertEqual(status, 'ok')
            self.assertIn(chess.Move.from_uci(bestmove.split()[1]), chess.Board().legal_moves)
            self.assertLess(time.monotonic() - started, 2)
        client.send('position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
        client.send('go nodes 4')
        bestmove, _, status = client.read_until('bestmove ', 5)
        self.assertEqual((bestmove, status), ('bestmove 0000', 'ok'))
        client.send('go infinite')
        client.send('quit')
        client.process.wait(timeout=5)
        self.assertEqual(client.process.returncode, 0)

    def test_lazy_handshake_and_missing_or_corrupt_weights_fail_closed(self):
        for corrupt in (False, True):
            path = self.root / ('corrupt' if corrupt else 'missing')
            if corrupt:
                shutil.copytree(self.artifact, path)
                with (path / 'weights.pt').open('ab') as stream:
                    stream.write(b'corrupted')
            command = [sys.executable, str(ROOT / 'tools/neural_uci.py'), '--model', str(path)]
            proc = subprocess.run(command, input='uci\nquit\n', capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('uciok', proc.stdout)
            proc = subprocess.run(command, input='uci\nisready\n', capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 1)
            self.assertNotIn('readyok', proc.stdout)
            self.assertIn('ERROR', proc.stdout)
            self.assertNotIn('bestmove', proc.stdout)

    def test_launcher_handles_other_working_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = subprocess.run([str(ROOT / 'inez-neural'), '--model', str(self.artifact), '--threads', '1'],
                input='uci\nisready\nquit\n', cwd=directory,
                env={**os.environ, 'INEZ_PYTHON': sys.executable},
                capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('readyok', proc.stdout)
            self.assertIn('parameters=', proc.stderr)
            self.assertTrue(all(line.startswith(('id ', 'option ', 'uciok', 'readyok'))
                                for line in proc.stdout.splitlines()))


if __name__ == '__main__':
    unittest.main()
