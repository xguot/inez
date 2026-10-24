#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Exercise the real submission shell without launching scheduler jobs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'hpc/submit_inezita.sh'


class SlurmSubmissionTests(unittest.TestCase):
    def test_batch_logs_use_existing_checkout_log_directory(self):
        # Slurm opens logs before running the script, so mkdir inside it is too late.
        self.assertTrue((ROOT / 'hpc/logs/.gitkeep').is_file())
        scripts = [WRAPPER, *sorted((ROOT / 'hpc').glob('*.slurm'))]
        self.assertEqual(len(scripts), 5)
        for script in scripts:
            with self.subTest(script=script.name):
                directives = dict(line.removeprefix('#SBATCH --').split('=', 1)
                                  for line in script.read_text().splitlines()
                                  if line.startswith(('#SBATCH --output=', '#SBATCH --error=')))
                for option, suffix in (('output', '.out'), ('error', '.err')):
                    path = Path(directives[option])
                    self.assertEqual(path.parent, Path('hpc/logs'))
                    self.assertIn('%j', path.name)
                    self.assertTrue(path.name.endswith(suffix))
                    self.assertTrue((ROOT / path.parent).is_dir())

    def fixture(self, directory):
        root = directory / 'checkout with spaces'
        (root / 'tools').mkdir(parents=True)
        (root / 'tools/submit_training.py').write_text('# dispatcher fixture\n')
        executable = root / 'python with spaces'
        executable.write_text('#!' + sys.executable + "\n" +
            'import json, os, sys\n' +
            'print(json.dumps({"argv": sys.argv, "cwd": os.getcwd(), '
            '"unbuffered": os.environ.get("PYTHONUNBUFFERED"), '
            '"no_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE")}))\n')
        executable.chmod(0o755)
        env = {key: value for key, value in os.environ.items()
               if key not in ('INEZ_ROOT', 'SLURM_SUBMIT_DIR', 'INEZ_PYTHON')}
        env['INEZ_PYTHON'] = str(executable)
        return root, executable, env

    def run_wrapper(self, script, env, cwd, *arguments):
        return subprocess.run(['/bin/bash', str(script), *arguments], env=env, cwd=cwd,
                              capture_output=True, text=True, timeout=10)

    def test_slurm_spool_uses_submission_checkout_and_forwards_arguments_literally(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root, _, env = self.fixture(directory)
            spool = directory / 'slurm-spool'
            spool.mkdir()
            copied = spool / 'slurm_script'
            shutil.copyfile(WRAPPER, copied)
            env['SLURM_SUBMIT_DIR'] = str(root)
            arguments = ['scratch', '--dry-run', 'literal $(touch unexpected) ; argument']
            proc = self.run_wrapper(copied, env, spool, *arguments)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result['argv'][1], str(root.resolve() / 'tools/submit_training.py'))
            self.assertEqual(result['argv'][2:], arguments)
            self.assertEqual(result['cwd'], str(root.resolve()))
            self.assertEqual(result['unbuffered'], '1')
            self.assertEqual(result['no_bytecode'], '1')
            self.assertFalse((root / 'unexpected').exists())

    def test_explicit_root_wins_over_stale_slurm_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root, _, env = self.fixture(directory)
            env.update(INEZ_ROOT=str(root), SLURM_SUBMIT_DIR=str(directory / 'missing'))
            proc = self.run_wrapper(WRAPPER, env, directory, 'resume')
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(json.loads(proc.stdout)['argv'][2:], ['resume'])

    def test_home_prefix_and_python_command_on_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root, executable, env = self.fixture(directory)
            env.update(INEZ_ROOT=str(root), HOME=str(directory),
                       INEZ_PYTHON='HOME/' + str(executable.relative_to(directory)))
            proc = self.run_wrapper(WRAPPER, env, directory)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            command = root / 'python-command'
            shutil.copyfile(executable, command)
            command.chmod(0o755)
            env.update(INEZ_PYTHON=command.name, PATH=str(root) + os.pathsep + os.environ.get('PATH', ''))
            proc = self.run_wrapper(WRAPPER, env, directory)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_invalid_explicit_interpreter_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root, _, env = self.fixture(directory)
            env.update(INEZ_ROOT=str(root), INEZ_PYTHON=str(directory / 'missing-python'))
            proc = self.run_wrapper(WRAPPER, env, directory)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('INEZ_PYTHON is not an executable', proc.stderr)
            self.assertEqual(proc.stdout, '')
            env['INEZ_PYTHON'] = str(root)
            proc = self.run_wrapper(WRAPPER, env, directory)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('INEZ_PYTHON is not an executable', proc.stderr)

    def test_invalid_submission_directory_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _, _, env = self.fixture(directory)
            env['SLURM_SUBMIT_DIR'] = str(directory / 'missing-checkout')
            proc = self.run_wrapper(WRAPPER, env, directory)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('Submit from the checkout root or set INEZ_ROOT', proc.stderr)
            self.assertEqual(proc.stdout, '')

    def test_login_shell_resolves_checkout_from_script_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _, _, env = self.fixture(directory)
            proc = self.run_wrapper(WRAPPER, env, directory, '--dry-run')
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(json.loads(proc.stdout)['argv'][1], str(ROOT.resolve() / 'tools/submit_training.py'))


if __name__ == '__main__':
    unittest.main()
