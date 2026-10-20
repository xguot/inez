# SPDX-License-Identifier: GPL-3.0-only
"""Bounded supervised training with exact RNG/optimizer resume and honest gates."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import tempfile

import chess
import torch
from torch.nn import functional as F

from .curriculum import atomic_json, sha256
from .encoding import encode_position, legal_policy_indices, move_to_policy_index
from .model import ChessTransformer, ModelConfig, save_checkpoint
from .records import load_records, validate_record


def dataset_records(directory, split):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    path = directory / (split + '.jsonl')
    if sha256(path) != manifest['splits'][split]['sha256']:
        raise ValueError(f'dataset checksum mismatch: {path}')
    records = load_records([path])
    if len(records) != manifest['splits'][split]['records']:
        raise ValueError('dataset manifest record count mismatch')
    return records


class RecordBatches:
    """Cache only minibatch encodings; do not materialize a corpus on the GPU."""
    def __init__(self, records):
        if not records:
            raise ValueError('empty training/validation records')
        self.records = records
        # Legal game histories are validated by dataset_records/load_records.

    def batch(self, indices, device):
        xs, masks, policies, wdls = [], [], [], []
        for index in indices:
            record = self.records[index]
            board = chess.Board(record['initial_fen'])
            for move in record['moves'][:record['ply']]:
                board.push_uci(move)
            xs.append(encode_position(board))
            mask = torch.zeros(4288, dtype=torch.bool)
            mask[legal_policy_indices(board)] = True
            policy = torch.zeros(4288)
            for move, probability in record['policy'].items():
                policy[move_to_policy_index(board, chess.Move.from_uci(move))] = probability
            masks.append(mask)
            policies.append(policy)
            wdls.append(record['wdl'])
        return (torch.tensor(xs, dtype=torch.float32, device=device).reshape(-1, 64, 119),
                torch.stack(masks).to(device), torch.stack(policies).to(device),
                torch.tensor(wdls, dtype=torch.float32, device=device))


def supervised_loss(model, batch):
    features, mask, policy_target, wdl_target = batch
    policy_logits, value_logits = model(features)
    # Finite masking avoids 0 * -inf on zero-probability policy entries.
    log_probs = F.log_softmax(policy_logits.masked_fill(~mask, -1e9), dim=-1)
    policy_rows = policy_target.sum(dim=-1) > 0
    policy_loss = -(policy_target * log_probs).sum(dim=-1)
    policy_loss = policy_loss[policy_rows].mean() if bool(policy_rows.any()) else policy_logits.sum() * 0
    value_loss = -(wdl_target * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
    return policy_loss + value_loss, policy_loss, value_loss


@torch.no_grad()
def validation_loss(model, records, batch_size, device, limit=256):
    model.eval()
    dataset = RecordBatches(records)
    indices = random.Random(0).sample(range(len(records)), min(limit, len(records)))
    total, count = 0.0, 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        loss, _, _ = supervised_loss(model, dataset.batch(batch_indices, device))
        total += float(loss.item()) * len(batch_indices)
        count += len(batch_indices)
    return total / count


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class StopRequested:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        for signum in (signal.SIGTERM, signal.SIGUSR1):
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self.handle)
        return self

    def handle(self, signum, frame):
        self.requested = True

    def __exit__(self, *args):
        for signum, previous in self.previous.items():
            handler = signal.SIG_DFL if previous is None else previous
            try:
                signal.signal(signum, handler)
            except (TypeError, ValueError, OSError):
                pass


def fingerprint(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(',', ':')).encode())
    return digest.hexdigest()


def train_stage(model, train, validation, output, *, config, stage, gate, stop=None):
    """Every exported candidate is evaluated; the gate owns champion selection.

    last.pt is a training-state checkpoint, never an accepted playing model.
    Resume validates data/config and restores the next minibatch RNG state.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    steps, batch_size = config['steps'], config['batch_size']
    if steps < 1 or batch_size < 1 or config['checkpoint_every'] < 1:
        raise ValueError('training steps, batch size and checkpoint interval must be positive')
    if any(record['label_kind'] != 'game_result' for record in train) and stage == 'human':
        raise ValueError('human stage expects game-result initialization labels')
    if stage == 'teacher' and any(record['label_kind'] != 'teacher_wdl' for record in train):
        raise ValueError('teacher stage must replace both human policy and value targets')
    if stage == 'selfplay' and any(record['label_kind'] not in ('teacher_wdl', 'search') for record in train):
        raise ValueError('self-play updates may replay teacher labels, never human imitation labels')
    dataset = RecordBatches(train)
    data_id = fingerprint(train) + ':' + fingerprint(validation)
    signature = {'stage': stage, 'settings': config, 'model': model.metadata(), 'data': data_id}
    generator = torch.Generator().manual_seed(config['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    start_step = 0
    checkpoint = output / 'last.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['signature'] != signature:
            raise ValueError('resume configuration or dataset changed; use a new run directory')
        expected_gpu = torch.cuda.get_device_name(device) if device.type == 'cuda' else None
        if (state['context']['device'] != str(device) or state['context']['gpu_name'] != expected_gpu
                or state['context']['torch'] != str(torch.__version__)):
            raise ValueError('exact resume requires the same device type/model and PyTorch version; use a new run with --init-artifact to change hardware')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        if device.type == 'cuda' and state['cuda_rng']:
            if len(state['cuda_rng']) != torch.cuda.device_count():
                raise ValueError('exact resume requires the same visible CUDA device count')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_step = state['step']
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    context = {'hostname': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               'device': str(device), 'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               'python': sys.executable, 'torch': str(torch.__version__), 'git_revision': revision}

    def save(step):
        atomic_checkpoint(checkpoint, {'format_version': 1, 'signature': signature,
            'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'sampler_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], 'context': context})

    # Complete a pending candidate evaluation after a signal or failed gate run.
    def evaluate_step(step):
        candidate = output / f'candidate-{step:07d}'
        report = output / f'gate-{step:07d}.json'
        if not candidate.exists():
            save_checkpoint(model, candidate)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type == 'cuda' else []):
            gate(candidate, report)

    if start_step and (start_step % config['checkpoint_every'] == 0 or start_step == steps):
        evaluate_step(start_step)
    if start_step >= steps:
        return True
    for step in range(start_step + 1, steps + 1):
        model.train()
        indices = torch.randint(len(train), (batch_size,), generator=generator).tolist()
        batch = dataset.batch(indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, policy_loss, value_loss = supervised_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % config.get('log_every', 20) == 0:
            print(json.dumps({'stage': stage, 'step': step, 'loss': float(loss.item()),
                              'policy_loss': float(policy_loss.item()), 'value_loss': float(value_loss.item())}), flush=True)
        due = step % config['checkpoint_every'] == 0 or step == steps
        if due or stop is not None and stop.requested:
            save(step)
        if stop is not None and stop.requested:
            print('Signal received; atomic training checkpoint saved. Resume with the same run directory.', flush=True)
            return False
        if due:
            value = validation_loss(model, validation, batch_size, device)
            if not math.isfinite(value):
                raise ValueError('nonfinite validation loss')
            atomic_json(output / f'validation-{step:07d}.json', {'step': step, 'stage': stage,
                        'loss': value, 'records': min(256, len(validation)), 'context': context})
            evaluate_step(step)
    return True
