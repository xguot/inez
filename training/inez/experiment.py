# SPDX-License-Identifier: GPL-3.0-only
"""Pin the executable sources used by a training experiment.

A Slurm script is copied by the scheduler, but its imported Python modules and
configuration files otherwise remain live. A snapshot keeps those inputs beside
the run, independent of later checkout changes. Hashes detect accidental edits;
they do not replace filesystem access control.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile


SNAPSHOT_MANIFEST = '.inezita-source.json'
SNAPSHOT_VERSION = 1
SNAPSHOT_DIRECTORIES = ('training', 'tools', 'hpc', 'configs', 'tests/fixtures', 'data/openings')
_IGNORED_NAMES = frozenset(('__pycache__', '.DS_Store', '.pytest_cache'))


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('snapshot paths must be normalized relative POSIX paths')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError('snapshot paths must be normalized relative POSIX paths')
    return path


def _reject_symlink(path, root):
    """Reject links in every component rather than following external inputs."""
    relative = path.relative_to(root)
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f'symlinks are not supported in training sources: {current}')


def _ignored(relative):
    return (any(part in _IGNORED_NAMES for part in relative.parts)
            or relative.suffix in ('.pyc', '.pyo')
            or (relative.parts[0] == 'hpc' and
                (relative.parts[1:2] == ('logs',) or relative.suffix in ('.out', '.err'))))


def _source_files(root):
    root = Path(root)
    files = []
    for name in SNAPSHOT_DIRECTORIES:
        directory = root / name
        _reject_symlink(directory, root)
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError(f'expected training source directory: {directory}')
        for parent, directories, names in os.walk(directory, followlinks=False):
            parent = Path(parent)
            # Prune output directories before visiting their potentially large
            # contents. Their files are unrelated to the submitted code.
            retained = []
            for name in sorted(directories):
                child = parent / name
                if _ignored(child.relative_to(root)):
                    continue
                _reject_symlink(child, root)
                retained.append(name)
            directories[:] = retained
            for name in sorted(names):
                path = parent / name
                if _ignored(path.relative_to(root)):
                    continue
                _reject_symlink(path, root)
                if not path.is_file():
                    raise ValueError(f'unsupported training source file: {path}')
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def training_sources(root):
    """Return deterministic hashes for training and resume behavior.

    Configuration values and the selected tactical fixture are also bound by the
    run request; this map binds their implementations and model definitions.
    Changes to unrelated benchmark scripts do not change optimizer semantics.
    """
    root = Path(root).resolve()
    sources = [path for path in _source_files(root)
               if ((path.relative_to(root).as_posix().startswith('training/inez/') and path.suffix == '.py')
                   or path.relative_to(root).as_posix() in (
                       'training/requirements.txt', 'tools/train_inezita.py',
                       'hpc/train_inezita.slurm', 'hpc/prepare_inezita.slurm')
                   or (path.relative_to(root).as_posix().startswith('configs/models/') and path.suffix == '.json'))]
    if not sources:
        raise ValueError(f'no training sources found: {root}')
    return {path.relative_to(root).as_posix(): _hash_file(path) for path in sources}


def _git_context(root):
    # A snapshot nested below a checkout has no .git of its own. Do not mistake
    # that enclosing checkout for the revision that produced the snapshot.
    existing = root / SNAPSHOT_MANIFEST
    if existing.is_file():
        metadata = verify_snapshot(root)
        return metadata['source_revision'], metadata['source_worktree_status']
    if not (root / '.git').exists():
        return None, None
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                          '--untracked-files=normal'], text=True,
                                         stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, status


def create_snapshot(root, destination):
    """Atomically create a source snapshot at a previously unused destination.

    Both committed changes and current working files are copied. The recorded
    revision and status provide context; the file hashes define the actual code.
    """
    root = Path(root).resolve()
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise ValueError('snapshot destination cannot be a symlink')
    resolved_destination = destination.resolve()
    for name in SNAPSHOT_DIRECTORIES:
        if resolved_destination.is_relative_to(root / name):
            raise ValueError('snapshot destination must be outside its source directories')
    if destination.exists():
        raise FileExistsError(destination)
    sources = _source_files(root)
    if not sources:
        raise ValueError(f'no snapshot sources found: {root}')
    revision, status = _git_context(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.inezita-source-', dir=destination.parent))
    try:
        hashes = {}
        for source in sources:
            relative = source.relative_to(root)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = _hash_file(source)
            shutil.copy2(source, target)
            after = _hash_file(source)
            if before != after or _hash_file(target) != before:
                raise ValueError(f'training source changed while snapshotting: {source}')
            hashes[relative.as_posix()] = before
        # Detect a new/deleted file as well as edits while the copy was made.
        current = {path.relative_to(root).as_posix(): _hash_file(path) for path in _source_files(root)}
        if current != hashes:
            raise ValueError('training sources changed while snapshotting; retry with a stable checkout')
        metadata = {'snapshot_version': SNAPSHOT_VERSION, 'source_revision': revision,
                    'source_worktree_status': status, 'sources': hashes}
        with (staging / SNAPSHOT_MANIFEST).open('w', encoding='utf-8') as stream:
            json.dump(metadata, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # rename never overwrites a populated snapshot directory.
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def verify_snapshot(root):
    """Check every frozen source and reject missing, changed, or extra files."""
    root = Path(root).resolve()
    manifest_path = root / SNAPSHOT_MANIFEST
    if manifest_path.is_symlink():
        raise ValueError('snapshot manifest cannot be a symlink')
    metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_keys = {'snapshot_version', 'source_revision', 'source_worktree_status', 'sources'}
    if (not isinstance(metadata, dict) or set(metadata) != expected_keys
            or type(metadata['snapshot_version']) is not int
            or metadata['snapshot_version'] != SNAPSHOT_VERSION
            or not isinstance(metadata['sources'], dict) or not metadata['sources']):
        raise ValueError('invalid training source snapshot manifest')
    for key in ('source_revision', 'source_worktree_status'):
        if metadata[key] is not None and not isinstance(metadata[key], str):
            raise ValueError('invalid training source revision metadata')
    for name, expected in metadata['sources'].items():
        relative = _safe_relative(name)
        path = root.joinpath(*relative.parts)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in '0123456789abcdef' for char in expected)):
            raise ValueError('invalid training source checksum')
        _reject_symlink(path, root)
        if not path.is_file() or _hash_file(path) != expected:
            raise ValueError(f'training source snapshot changed: {name}')
    actual = {path.relative_to(root).as_posix() for path in _source_files(root)}
    if actual != set(metadata['sources']):
        raise ValueError('training source snapshot file set changed')
    return metadata


def _accepted_record(pointer):
    record = json.loads(pointer.read_text(encoding='utf-8'))
    if (not isinstance(record, dict) or not isinstance(record.get('history'), list)
            or not record['history'] or any(not isinstance(name, str) or not name for name in record['history'])
            or not isinstance(record.get('artifact'), str) or record['artifact'] not in record['history']):
        raise ValueError(f'invalid accepted model history: {pointer}')
    return record


def _relocated_artifacts(value, run):
    saved = Path(value).expanduser()
    candidates = []
    if not saved.is_absolute():
        local = (run / saved).resolve()
        if not local.is_relative_to(run):
            raise ValueError('relative accepted artifact escapes its training run')
        candidates.append(local)
    # Results pulled from HPC retain the original absolute paths. Match the
    # enclosing run name and prefer its local artifact before the cluster path.
    for index in reversed(range(len(saved.parts))):
        if saved.parts[index] == run.name:
            local = run.joinpath(*saved.parts[index + 1:]).resolve()
            if local.is_relative_to(run):
                candidates.append(local)
            break
    if saved.is_absolute():
        candidates.append(saved.resolve())
    return list(dict.fromkeys(candidates))


def verify_accepted_artifact(value, *, expected_config=None):
    """Resolve a previously accepted artifact and verify its actual tensor bytes.

    Accepts a run directory, its accepted.json, or a particular member of its
    accepted history. A bare artifact copied without its acceptance evidence is
    insufficient for continuation. Copied HPC runs retain their original JSON.
    """
    path = Path(value).expanduser().resolve()
    if path.is_dir() and (path / 'manifest.json').is_file():
        artifact = path
        eligible = False
        for run in artifact.parents:
            pointer = run / 'accepted.json'
            if not pointer.is_file():
                continue
            record = _accepted_record(pointer)
            if any(artifact in _relocated_artifacts(name, run) for name in record['history']):
                eligible = True
                break
        if not eligible:
            raise ValueError('initial artifact must belong to a previous run\'s accepted history; rejected candidates cannot bypass promotion')
    else:
        pointer = path / 'accepted.json' if path.is_dir() else path
        if pointer.name != 'accepted.json' or not pointer.is_file():
            raise ValueError('initial artifact must name an accepted model artifact, training run, or accepted.json')
        record = _accepted_record(pointer)
        artifact = next((candidate for candidate in _relocated_artifacts(record['artifact'], pointer.parent)
                         if (candidate / 'manifest.json').is_file()), None)
        if artifact is None:
            raise ValueError('accepted artifact is missing; copy its complete weights and manifest with the training run')
    # The strict loader checks metadata, architecture, encoding, checksum,
    # finite tensor contents, shapes and dtypes. This runs before GPU allocation.
    from .model import load_checkpoint
    verified = load_checkpoint(artifact, device='cpu', expected_config=expected_config)
    del verified
    return artifact
