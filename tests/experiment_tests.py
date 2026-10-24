#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Frozen training sources remain independent of checkout changes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from inez.experiment import (SNAPSHOT_MANIFEST, create_snapshot, training_sources,
                             verify_snapshot, verify_accepted_artifact)


class ExperimentTests(unittest.TestCase):
    def sources(self, root):
        files = {'training/inez/trainer.py': '# optimizer\n',
                 'training/requirements.txt': 'torch==2.14.1\n',
                 'tools/train_inezita.py': '# entry\n',
                 'tools/benchmark_inezita.py': '# benchmark\n',
                 'hpc/train_inezita.slurm': '# train\n',
                 'hpc/prepare_inezita.slurm': '# prepare\n',
                 'configs/models/inezita.json': '{}\n',
                 'configs/train/custom.json': '{}\n',
                 'tests/fixtures/independent_tactics.json': '[]\n',
                 'data/openings/mco_openings.tsv': 'name\tmoves\n'}
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return files

    def test_snapshot_contains_current_sources_and_survives_checkout_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checkout'
            files = self.sources(root)
            initial = training_sources(root)
            snapshot = create_snapshot(root, root / 'results/run/source')
            metadata = verify_snapshot(snapshot)
            self.assertEqual(set(metadata['sources']), set(files))
            self.assertIsNone(metadata['source_revision'])
            self.assertEqual(training_sources(snapshot), initial)
            self.assertNotIn('tools/benchmark_inezita.py', initial)
            (root / 'training/inez/trainer.py').write_text('# changed optimizer\n')
            self.assertNotEqual(training_sources(root), initial)
            self.assertEqual(training_sources(snapshot), initial)
            verify_snapshot(snapshot)
            with self.assertRaises(FileExistsError):
                create_snapshot(root, snapshot)

    def test_generated_logs_and_bytecode_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checkout'
            files = self.sources(root)
            for name in ('hpc/logs/train.out', 'hpc/generated.err',
                         'training/inez/__pycache__/trainer.cpython.pyc'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('generated output')
            snapshot = create_snapshot(root, Path(tmp) / 'frozen')
            self.assertEqual(set(verify_snapshot(snapshot)['sources']), set(files))
            self.assertFalse((snapshot / 'hpc/logs').exists())
            (root / 'hpc/logs/train.out').write_text('new output')
            self.assertEqual(training_sources(root), training_sources(snapshot))
            (snapshot / 'hpc/logs').mkdir()
            (snapshot / 'hpc/logs/new.out').write_text('unrelated runtime log')
            verify_snapshot(snapshot)

    def test_verification_rejects_edits_deleted_and_extra_sources(self):
        for mutation in ('edit', 'delete', 'extra', 'symlink'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'checkout'
                self.sources(root)
                snapshot = create_snapshot(root, Path(tmp) / 'frozen')
                target = snapshot / 'training/inez/trainer.py'
                if mutation == 'edit':
                    target.write_text('# modified\n')
                elif mutation == 'delete':
                    target.unlink()
                elif mutation == 'extra':
                    (snapshot / 'training/inez/extra.py').write_text('# surprise import\n')
                else:
                    target.unlink()
                    target.symlink_to(root / 'training/inez/trainer.py')
                with self.assertRaises(ValueError):
                    verify_snapshot(snapshot)

    def test_unsafe_paths_and_symlink_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checkout'
            self.sources(root)
            with self.assertRaisesRegex(ValueError, 'outside'):
                create_snapshot(root, root / 'training/frozen')
            link = root / 'tools/linked.py'
            link.symlink_to(root / 'training/inez/trainer.py')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                create_snapshot(root, Path(tmp) / 'frozen')
            link.unlink()
            snapshot = create_snapshot(root, Path(tmp) / 'frozen')
            metadata_path = snapshot / SNAPSHOT_MANIFEST
            metadata = json.loads(metadata_path.read_text())
            digest = next(iter(metadata['sources'].values()))
            for unsafe in ('../outside.py', '/tmp/outside.py', 'tools/../outside.py', 'tools\\outside.py'):
                metadata['sources'] = {unsafe: digest}
                metadata_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'relative POSIX'):
                    verify_snapshot(snapshot)

    def test_snapshot_copy_rejects_a_concurrent_source_edit(self):
        import inez.experiment as experiment
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checkout'
            self.sources(root)
            original_copy = experiment.shutil.copy2
            def changing_copy(source, target):
                result = original_copy(source, target)
                Path(source).write_text('# concurrent edit\n')
                return result
            destination = Path(tmp) / 'frozen'
            with patch.object(experiment.shutil, 'copy2', side_effect=changing_copy):
                with self.assertRaisesRegex(ValueError, 'while snapshotting'):
                    create_snapshot(root, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(tmp).glob('.inezita-source-*')), [])

    def test_snapshot_of_snapshot_preserves_original_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checkout'
            self.sources(root)
            snapshot = create_snapshot(root, Path(tmp) / 'first')
            manifest = snapshot / SNAPSHOT_MANIFEST
            metadata = json.loads(manifest.read_text())
            metadata.update(source_revision='a' * 40, source_worktree_status=' M training/inez/trainer.py\n')
            manifest.write_text(json.dumps(metadata))
            second = create_snapshot(snapshot, Path(tmp) / 'second')
            self.assertEqual(verify_snapshot(second)['source_revision'], 'a' * 40)
            self.assertEqual(verify_snapshot(second)['source_worktree_status'], metadata['source_worktree_status'])


class AcceptedArtifactTests(unittest.TestCase):
    def artifact(self, run, name):
        import torch
        from inez.model import ChessTransformer, ModelConfig, save_checkpoint
        torch.set_num_threads(1)
        config = ModelConfig(width=16, layers=1, heads=2, dynamic_rank=2, policy_width=4)
        return save_checkpoint(ChessTransformer(config), run / name).resolve()

    def test_run_pointer_and_historical_artifacts_relocate_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'copied-inezita'
            first = self.artifact(run, 'teacher/candidate-0001000')
            latest = self.artifact(run, 'teacher/candidate-0002000')
            history = ['/cluster/results/copied-inezita/teacher/candidate-0001000',
                       '/cluster/results/copied-inezita/teacher/candidate-0002000']
            pointer = run / 'accepted.json'
            original = json.dumps({'artifact': history[-1], 'history': history})
            pointer.write_text(original)
            self.assertEqual(verify_accepted_artifact(run), latest)
            self.assertEqual(verify_accepted_artifact(pointer), latest)
            self.assertEqual(verify_accepted_artifact(first), first)
            self.assertEqual(pointer.read_text(), original)
            rejected = self.artifact(run, 'teacher/candidate-0003000')
            with self.assertRaisesRegex(ValueError, 'accepted history'):
                verify_accepted_artifact(rejected)

    def test_missing_evidence_and_corrupt_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            artifact = self.artifact(run, 'initial')
            with self.assertRaisesRegex(ValueError, 'accepted history'):
                verify_accepted_artifact(artifact)
            pointer = run / 'accepted.json'
            pointer.write_text(json.dumps({'artifact': 'initial', 'history': ['initial']}))
            self.assertEqual(verify_accepted_artifact(run), artifact)
            (artifact / 'weights.pt').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_accepted_artifact(run)

    def test_invalid_acceptance_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            self.artifact(run, 'initial')
            pointer = run / 'accepted.json'
            pointer.write_text(json.dumps({'artifact': 'initial', 'history': ['another']}))
            with self.assertRaisesRegex(ValueError, 'invalid accepted'):
                verify_accepted_artifact(run)


if __name__ == '__main__':
    unittest.main()
