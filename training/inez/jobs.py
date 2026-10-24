# SPDX-License-Identifier: GPL-3.0-only
"""Validate frozen training job inputs before scheduler work consumes them."""
import json
import platform
from pathlib import Path

from .experiment import verify_snapshot
from .curriculum import sha256


def verify_job_spec(path, *, source_root=None, runtime=False, resume_mode='exact'):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if type(spec.get('job_version')) is not int or spec.get('job_version') != 1 or spec.get('mode') not in ('continue', 'scratch'):
        raise ValueError('invalid training job specification')
    frozen = Path(spec['source_root']).resolve()
    if source_root is not None and frozen != Path(source_root).resolve():
        raise ValueError('job must execute from its frozen source checkout')
    metadata = verify_snapshot(frozen)
    if sha256(frozen / '.inezita-source.json') != spec['source_manifest_sha256']:
        raise ValueError('frozen source manifest changed')
    if spec['config'] not in spec.get('input_files', {}):
        raise ValueError('job configuration is not pinned by its specification')
    for name, expected in spec['input_files'].items():
        if sha256(name) != expected:
            raise ValueError(f'frozen training input changed: {name}')
    artifact = spec.get('initial_artifact')
    if (spec['mode'] == 'continue') != bool(artifact):
        raise ValueError('job initialization mode and initial artifact disagree')
    if artifact:
        directory = Path(artifact)
        manifest = json.loads((directory / 'manifest.json').read_text())
        actual = sha256(directory / 'weights.pt')
        if (actual != spec['initial_weights_sha256'] or actual != manifest['weights_sha256']
                or sha256(directory / 'manifest.json') != spec['initial_manifest_sha256']):
            raise ValueError('initial artifact changed since job submission')
    if runtime:
        import chess
        import torch
        actual = {'python': platform.python_version(), 'torch': str(torch.__version__),
                  'chess': chess.__version__}
        if actual != spec.get('runtime') and resume_mode != 'portable':
            raise ValueError('Python/PyTorch/chess changed since submission; restore the environment or explicitly use RESUME_MODE=portable')
    return spec
