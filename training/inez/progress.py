# SPDX-License-Identifier: GPL-3.0-only
"""Checksummed, atomic game-boundary progress for long curriculum jobs."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def restore_random_state(rng, value):
    def tuples(item):
        return tuple(tuples(child) for child in item) if isinstance(item, list) else item
    try:
        rng.setstate(tuples(value))
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('invalid saved game random state') from error


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate progress JSON key')
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, ValueError) as error:
        raise ValueError(f'invalid progress file: {path}') from error


def _atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(canonical_json(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class GameProgress:
    """Immutable request plus one atomic chunk per attempted game.

    The curriculum's outer run lock must cover generation. An interrupted game
    is replayed from the preceding committed boundary, including its RNG state.
    """

    def __init__(self, path, request):
        self.path = Path(path) if path is not None else None
        self.chunks = []
        self.request = json.loads(canonical_json(request))
        self.digest = content_digest(self.request)
        if self.path is None:
            return
        if not isinstance(request.get('identity'), dict) or not request['identity']:
            raise ValueError('persistent game progress requires a nonempty progress_identity object')
        self.path.mkdir(parents=True, exist_ok=True)
        header = self.path / 'request.json'
        files = sorted(self.path.glob('game-*.json'))
        expected = {'version': 1, 'request': self.request, 'sha256': self.digest}
        if header.exists():
            if _read(header) != expected:
                raise ValueError('game progress request or identity changed')
        elif files:
            raise ValueError('game progress chunks have no request metadata')
        else:
            _atomic_json(header, expected)
        for index, path in enumerate(files):
            if path.name != f'game-{index:07d}.json':
                raise ValueError('game progress chunks are not contiguous')
            chunk = _read(path)
            if (not isinstance(chunk, dict)
                    or set(chunk) != {'version', 'request_sha256', 'index', 'payload', 'sha256'}):
                raise ValueError('invalid game progress chunk structure')
            signed = {key: value for key, value in chunk.items() if key != 'sha256'}
            if (type(chunk['version']) is not int or chunk['version'] != 1
                    or type(chunk['index']) is not int or chunk['index'] != index
                    or chunk['request_sha256'] != self.digest
                    or chunk['sha256'] != content_digest(signed)):
                raise ValueError('game progress checksum or identity mismatch')
            self.chunks.append(chunk['payload'])

    def append(self, payload):
        index = len(self.chunks)
        if self.path is not None:
            destination = self.path / f'game-{index:07d}.json'
            if destination.exists():
                raise ValueError('game progress destination already exists; concurrent generation is unsafe')
            signed = {'version': 1, 'request_sha256': self.digest,
                      'index': index, 'payload': payload}
            _atomic_json(destination, {**signed, 'sha256': content_digest(signed)})
        self.chunks.append(payload)
