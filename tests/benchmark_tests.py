#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Benchmark integrity checks: frozen weights, opening diversity and paired results."""

from argparse import Namespace
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import chess
import chess.pgn

from benchmark_inezita import (artifact_record, audit_games, external_opponent_record,
    fastchess_command, parse_uci_options, previous_artifact, position_key,
    select_openings, sha256, verify_external_opponent_unchanged, wait_for_tournament)
from inez.device import validate_cuda_runtime


def opening(moves):
    board = chess.Board()
    for move in moves.split():
        board.push_uci(move)
    return chess.pgn.Game.from_board(board)


class BenchmarkTests(unittest.TestCase):
    def test_native_signal_handlers_are_restored_as_default(self):
        child = MagicMock()
        child.wait.return_value = 0
        child.poll.return_value = 0
        with patch('benchmark_inezita.signal.getsignal', return_value=None), \
             patch('benchmark_inezita.signal.signal') as install:
            def validate_handler(signum, handler):
                if handler is None:
                    raise TypeError('handler must be callable, SIG_DFL or SIG_IGN')
            install.side_effect = validate_handler
            self.assertEqual(wait_for_tournament(child), 0)
            self.assertEqual(install.call_args_list[-2].args, (signal.SIGTERM, signal.SIG_DFL))
            self.assertEqual(install.call_args_list[-1].args, (signal.SIGINT, signal.SIG_DFL))
        child.send_signal.assert_not_called()

    def test_interrupted_child_is_reaped_and_original_error_preserved(self):
        child = MagicMock()
        child.poll.return_value = None
        child.wait.side_effect = [InterruptedError('scheduler stop'), 0]
        with patch('benchmark_inezita.signal.getsignal', return_value=signal.SIG_IGN), \
             patch('benchmark_inezita.signal.signal') as install:
            with self.assertRaisesRegex(InterruptedError, 'scheduler stop'):
                wait_for_tournament(child)
            self.assertEqual(install.call_args_list[-1].args, (signal.SIGINT, signal.SIG_IGN))
        child.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertEqual(child.wait.call_args.kwargs, {'timeout': 20})

    def test_unresponsive_child_is_killed_during_cleanup(self):
        child = MagicMock()
        child.poll.return_value = None
        child.wait.side_effect = [InterruptedError('scheduler stop'),
                                 subprocess.TimeoutExpired('fastchess', 20), -9]
        with patch('benchmark_inezita.signal.getsignal', return_value=None), \
             patch('benchmark_inezita.signal.signal'):
            with self.assertRaisesRegex(InterruptedError, 'scheduler stop'):
                wait_for_tournament(child)
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 3)

    def test_partial_signal_install_failure_still_reaps_child(self):
        child = MagicMock()
        child.poll.return_value = None
        def install(signum, handler):
            if signum == signal.SIGINT and callable(handler):
                raise OSError('cannot install')
        with patch('benchmark_inezita.signal.getsignal', return_value=None), \
             patch('benchmark_inezita.signal.signal', side_effect=install):
            with self.assertRaisesRegex(OSError, 'cannot install'):
                wait_for_tournament(child)
        child.send_signal.assert_called_once_with(signal.SIGINT)
        child.wait.assert_called_once_with(timeout=20)

    def test_cpu_preflight_does_not_touch_cuda(self):
        with patch('inez.device.torch.cuda.is_available') as available, \
             patch('inez.device.torch.ones') as allocate:
            self.assertEqual(validate_cuda_runtime('cpu')['device'], 'cpu')
        available.assert_not_called()
        allocate.assert_not_called()

    def test_cuda_preflight_detects_v100_kernel_mismatch_before_model_load(self):
        with patch('inez.device.torch.cuda.is_available', return_value=True), \
             patch('inez.device.torch.cuda.get_device_name', return_value='Tesla V100-SXM2-32GB'), \
             patch('inez.device.torch.cuda.get_device_capability', return_value=(7, 0)), \
             patch('inez.device.torch.cuda.get_arch_list', return_value=['sm_75', 'sm_80']), \
             patch('inez.device.torch.version.cuda', '13.0'), \
             patch('inez.device.torch.ones', side_effect=RuntimeError('no kernel image is available')) as allocate:
            with self.assertRaisesRegex(RuntimeError, 'V100.*sm_70.*a100_40gb.*cu126.*no kernel image'):
                validate_cuda_runtime('cuda')
        self.assertEqual(str(allocate.call_args.kwargs['device']), 'cuda')

    def test_cuda_preflight_executes_math_and_synchronizes_supported_device(self):
        tensor = MagicMock()
        tensor.__matmul__.return_value.sum.return_value.item.return_value = 4096
        with patch('inez.device.torch.cuda.is_available', return_value=True), \
             patch('inez.device.torch.cuda.get_device_name', return_value='NVIDIA A100-SXM4-40GB'), \
             patch('inez.device.torch.cuda.get_device_capability', return_value=(8, 0)), \
             patch('inez.device.torch.cuda.get_arch_list', return_value=['sm_80']), \
             patch('inez.device.torch.ones', return_value=tensor), \
             patch('inez.device.torch.cuda.synchronize') as synchronize:
            info = validate_cuda_runtime('cuda')
        self.assertEqual(info['kernel_probe'], 'passed')
        self.assertEqual(info['compute_capability'], [8, 0])
        synchronize.assert_called_once()

    def test_opening_slice_is_unique_disjoint_and_preserves_history(self):
        first = opening('e2e4 e7e5 g1f3 b8c6')
        second = opening('d2d4 d7d5 c2c4 e7e6')
        third = opening('c2c4 e7e5 b1c3 g8f6')
        text = '\n\n'.join(str(g) for g in (first, first, second, third))
        pilot = select_openings(text, 0, 1, set())
        main = select_openings(text, 1, 2, set())
        self.assertEqual(list(pilot[0].mainline_moves()), list(first.mainline_moves()))
        self.assertEqual(list(main[0].mainline_moves()), list(second.mainline_moves()))
        self.assertEqual(list(main[1].mainline_moves()), list(third.mainline_moves()))
        excluded = {position_key(first.end().board())}
        self.assertEqual(list(select_openings(text, 0, 1, excluded)[0].mainline_moves()),
                         list(second.mainline_moves()))
        with self.assertRaisesRegex(ValueError, 'lacks'):
            select_openings(text, 0, 4, set())

    def test_copied_cluster_history_selects_previous_accepted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'inezita'
            for name in ('older', 'current'):
                (run / name).mkdir(parents=True)
                (run / name / 'manifest.json').write_text('{}')
            history = [f'/cluster/results/inezita/{name}' for name in ('older', 'current')]
            (run / 'accepted.json').write_text(json.dumps({'artifact': history[-1], 'history': history}))
            self.assertEqual(previous_artifact(run), (run / 'older').resolve())

    def test_modified_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'weights.pt').write_bytes(b'frozen test weights')
            (root / 'manifest.json').write_text(json.dumps({'weights': 'weights.pt',
                'weights_sha256': sha256(root / 'weights.pt'), 'config': {}}))
            artifact_record(root)
            (root / 'weights.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                artifact_record(root)

    def test_pair_audit_checks_colors_opening_history_and_faults(self):
        book = opening('e2e4 e7e5')
        games = []
        for white in ('inezita-candidate', 'inezita-incumbent'):
            game = opening('e2e4 e7e5 g1f3')
            game.headers.update(Round='1', White=white,
                Black='inezita-incumbent' if white == 'inezita-candidate' else 'inezita-candidate',
                Result='1-0', Termination='time forfeit')
            games.append(game)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'games.pgn'
            path.write_text('\n\n'.join(str(g) for g in games))
            summary = audit_games(path, [book])
            self.assertEqual((summary['games'], summary['pairs']), (2, 1))
            self.assertEqual((summary['wins'], summary['losses']), (1, 1))
            self.assertEqual(summary['pentanomial'], [0, 0, 1, 0, 0])
            self.assertEqual(summary['faults'], {'time forfeit': 2})
            games[1].headers.update(games[0].headers)
            path.write_text('\n\n'.join(str(g) for g in games))
            with self.assertRaisesRegex(ValueError, 'color reversal'):
                audit_games(path, [book])
            games[1] = opening('d2d4 d7d5')
            games[1].headers.update(games[0].headers)
            path.write_text('\n\n'.join(str(g) for g in games))
            with self.assertRaisesRegex(ValueError, 'opening/history'):
                audit_games(path, [book])

    def test_unfinished_game_supplies_no_draw_or_completed_pair(self):
        book = opening('e2e4 e7e5')
        game = chess.pgn.read_game(io.StringIO(str(book)))
        game.headers.update(Round='1', White='inezita-candidate', Black='inezita-incumbent',
                            Result='*', Termination='unterminated')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'games.pgn'
            path.write_text(str(game))
            summary = audit_games(path, [book])
            self.assertEqual((summary['unfinished'], summary['draws'], summary['pairs']), (1, 0, 0))

    def test_external_opponent_record_and_fastchess_command_isolate_engine_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sf_bin = root / 'stockfish'
            sf_bin.write_bytes(b'stockfish-binary')
            lc0_bin = root / 'lc0'
            lc0_bin.write_bytes(b'lc0-binary')
            lc0_weights = root / 'net.pb.gz'
            lc0_weights.write_bytes(b'lc0-weights')

            with patch('benchmark_inezita.resolve_stockfish', return_value=str(sf_bin)):
                sf_args = Namespace(
                    opponent=None, reference='stockfish-elo-1320', opponent_engine=None,
                    opponent_name=None, opponent_arg=[], opponent_option=['UCI_Elo=1400'],
                    opponent_nodes=None, lc0_weights=None, binary=root / 'inezita',
                    fastchess=root / 'fastchess', device='cuda', threads=4,
                    nodes=None, time_control='60+0.6', pairs=10,
                )
                sf_rec = external_opponent_record(sf_args)
            self.assertEqual(sf_rec['kind'], 'external_uci')
            self.assertEqual(sf_rec['name'], 'stockfish-17.1-elo1320')
            self.assertEqual(sf_rec['options']['UCI_Elo'], '1400')
            self.assertEqual(sf_rec['options']['UCI_LimitStrength'], 'true')

            cmd = fastchess_command(sf_args, root, {'path': '/model'}, sf_rec)
            self.assertIn('name=stockfish-17.1-elo1320', cmd)
            self.assertIn('option.UCI_LimitStrength=true', cmd)
            self.assertIn('option.UCI_Elo=1400', cmd)

            lc0_args = Namespace(
                opponent=None, reference='lc0', opponent_engine=lc0_bin,
                opponent_name=None, opponent_arg=[], opponent_option=[],
                opponent_nodes=128, lc0_weights=lc0_weights, binary=root / 'inezita',
                fastchess=root / 'fastchess', device='cuda', threads=4,
                nodes=128, time_control='60+0.6', pairs=10,
            )
            lc0_rec = external_opponent_record(lc0_args)
            self.assertEqual(lc0_rec['weights_sha256'], sha256(lc0_weights))
            verify_external_opponent_unchanged(lc0_rec)
            lc0_weights.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'weights changed'):
                verify_external_opponent_unchanged(lc0_rec)

            book = opening('e2e4 e7e5')
            games = []
            for white in ('inezita-candidate', 'stockfish-17.1-elo1320'):
                game = opening('e2e4 e7e5 g1f3')
                game.headers.update(
                    Round='1', White=white,
                    Black='stockfish-17.1-elo1320' if white == 'inezita-candidate' else 'inezita-candidate',
                    Result='1-0' if white == 'inezita-candidate' else '0-1',
                    Termination='normal',
                )
                games.append(game)
            pgn_path = root / 'games.pgn'
            pgn_path.write_text('\n\n'.join(str(g) for g in games))
            summary = audit_games(pgn_path, [book], opponent_name='stockfish-17.1-elo1320')
            self.assertEqual((summary['wins'], summary['losses'], summary['pairs']), (2, 0, 1))
            self.assertEqual(summary['pentanomial'], [0, 0, 0, 0, 1])

    def test_parse_uci_options_rejects_malformed_entries(self):
        self.assertEqual(parse_uci_options(['Hash=64', 'UCI_LimitStrength=true']),
                         {'Hash': '64', 'UCI_LimitStrength': 'true'})
        with self.assertRaisesRegex(ValueError, 'NAME=VALUE'):
            parse_uci_options(['invalid_option'])


if __name__ == '__main__':
    unittest.main()
