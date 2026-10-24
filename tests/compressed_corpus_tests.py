#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Compressed corpus integrity, dependency failure, and bounded PGN reads."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'training'))
import inez.curriculum as curriculum
from train_inezita import _validate_archive

PGN = '[Event "Fixture"]\n\n1. e4 e5 1/2-1/2\n'
# One complete frame, independent of optional Python/command-line compressors.
FRAME = bytes.fromhex('28b52ffd20242101005b4576656e74202246697874757265225d0a0a312e20653420653520312f322d312f320a')


class CompressedPreparationTests(unittest.TestCase):
    def decoder(self, folder, source):
        path = folder / 'decoder'
        path.write_text('#!' + sys.executable + '\n' + source)
        path.chmod(0o755)
        return str(path)

    def test_missing_decoder_fails_with_install_guidance(self):
        with patch.object(curriculum, '_stdlib_zstd', return_value=None), \
                patch.object(curriculum.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'install/load zstd'):
                curriculum.verify_zstd_runtime()

    def test_external_decoder_failure_after_valid_pgn_is_not_silent_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            executable = self.decoder(folder,
                'import sys\nsys.stdout.write(' + repr(PGN) + ')\n'
                'sys.stdout.flush()\nsys.stderr.write("truncated frame")\nsys.exit(7)\n')
            with patch.object(curriculum, '_zstd_backend', return_value=('external', executable)):
                with self.assertRaisesRegex(ValueError, 'exit 7.*truncated frame'):
                    with curriculum.pgn_stream(folder / 'broken.pgn.zst') as stream:
                        self.assertEqual(stream.read(), PGN)

    def test_bounded_external_read_terminates_decoder_without_false_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            executable = self.decoder(folder,
                'import sys\nwhile True:\n sys.stdout.write("x" * 65536)\n sys.stdout.flush()\n')
            with patch.object(curriculum, '_zstd_backend', return_value=('external', executable)):
                with curriculum.pgn_stream(folder / 'large.pgn.zst') as stream:
                    self.assertEqual(stream.read(1), 'x')

    def test_complete_concatenated_and_partial_download_frames(self):
        try:
            curriculum.verify_zstd_runtime()
        except ValueError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            complete = folder / 'complete.pgn.zst'
            complete.write_bytes(FRAME + FRAME)
            _validate_archive(complete)
            with curriculum.pgn_stream(complete) as stream:
                self.assertEqual(stream.read(), PGN * 2)
            partial = folder / 'download.pgn.zst.part'
            partial.write_bytes(FRAME)
            _validate_archive(partial)
            for data in (FRAME[:4], FRAME[:-1], FRAME + FRAME[:-1], FRAME + b'corrupt'):
                partial.write_bytes(data)
                with self.subTest(frame_bytes=len(data)), self.assertRaisesRegex(ValueError, 'corrupt zstd archive'):
                    _validate_archive(partial)

    @unittest.skipUnless(shutil.which('zstd'), 'external zstd not installed')
    def test_external_decoder_verifies_truncated_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.pgn.zst'
            with patch.object(curriculum, '_stdlib_zstd', return_value=None):
                path.write_bytes(FRAME)
                _validate_archive(path)
                path.write_bytes(FRAME[:-1])
                with self.assertRaisesRegex(ValueError, 'corrupt zstd archive'):
                    _validate_archive(path)


if __name__ == '__main__':
    unittest.main()
