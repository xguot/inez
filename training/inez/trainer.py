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
    """Cache decoded board tensors in host RAM; never materialize a corpus on the GPU."""
    def __init__(self, records, *, max_cache=65536):
        if not records:
            raise ValueError('empty training/validation records')
        self.records = records
        self.max_cache = max_cache
        self._cache = {}
        # Legal game histories are validated by dataset_records/load_records.

    def _encode_one(self, index):
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        record = self.records[index]
        board = chess.Board(record['initial_fen'])
        for move in record['moves'][:record['ply']]:
            board.push_uci(move)
        x = torch.tensor(encode_position(board), dtype=torch.float32).reshape(64, 119)
        mask = torch.zeros(4288, dtype=torch.bool)
        mask[legal_policy_indices(board)] = True
        policy = torch.zeros(4288, dtype=torch.float32)
        for move, probability in record['policy'].items():
            policy[move_to_policy_index(board, chess.Move.from_uci(move))] = probability
        wdl = torch.tensor(record['wdl'], dtype=torch.float32)
        item = (x, mask, policy, wdl)
        if len(self._cache) < self.max_cache:
            self._cache[index] = item
        return item

    def batch(self, indices, device):
        items = [self._encode_one(index) for index in indices]
        return (torch.stack([item[0] for item in items]).to(device),
                torch.stack([item[1] for item in items]).to(device),
                torch.stack([item[2] for item in items]).to(device),
                torch.stack([item[3] for item in items]).to(device))


def supervised_loss(model, batch):
    features, mask, policy_target, wdl_target = batch
    policy_logits, value_logits = model(features)
    # Finite masking avoids 0 * -inf on zero-probability policy entries.
    log_probs = F.log_softmax(policy_logits.masked_fill(~mask, -1e9), dim=-1)
    policy_rows = policy_target.sum(dim=-1) > 0
    policy_loss = -(policy_target * log_probs).sum(dim=-1)
    policy_loss = policy_loss[policy_rows].mean() if bool(policy_rows.any()) else policy_logits.sum() * 0
    wdl_log_probs = F.log_softmax(value_logits, dim=-1)
    wdl_probs = wdl_log_probs.exp()
    expected_diff = (wdl_probs[:, 0] - wdl_probs[:, 2]) - (wdl_target[:, 0] - wdl_target[:, 2])
    value_loss = -(wdl_target * wdl_log_probs).sum(dim=-1).mean() + expected_diff.square().mean()
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


def resume_environment(device):
    return {'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'torch': str(torch.__version__),
            'cuda_device_count': torch.cuda.device_count() if device.type == 'cuda' else 0}


def restore_checkpoint_rng(state, device, *, resume_mode, seed, checkpoint):
    """Validate exact resumes; explicitly reseed CUDA when migrating environments."""
    if resume_mode not in ('exact', 'portable'):
        raise ValueError('resume mode must be exact or portable')
    current = resume_environment(device)
    saved = {**state['context'], 'cuda_device_count': len(state['cuda_rng'])}
    differences = {key: {'checkpoint': saved.get(key), 'current': value}
                   for key, value in current.items() if saved.get(key) != value}
    if differences and resume_mode == 'exact':
        details = '; '.join(f'{key}: checkpoint={values["checkpoint"]!r}, current={values["current"]!r}'
                            for key, values in differences.items())
        raise ValueError(f'exact resume environment mismatch ({details}). '
                         'Use --resume-mode portable (Slurm: RESUME_MODE=portable) to continue '
                         'this run without bitwise reproducibility, or request the original environment.')
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    cuda_seed = None
    if device.type == 'cuda':
        if differences:
            # CUDA RNG state is specific to the saved numerical environment.
            # Retain the sampler order but start a documented new CUDA stream.
            cuda_seed = (seed + state['step']) % (2**64)
            torch.cuda.manual_seed_all(cuda_seed)
        else:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    event = {'checkpoint': str(checkpoint), 'step': state['step'],
             'mode': resume_mode, 'differences': differences, 'cuda_reseed': cuda_seed}
    print(json.dumps({'checkpoint_resume': event}), flush=True)
    return event


def _priority_sampling_table(records):
    """Build a deterministic 30% uniform + 70% capped policy-surprise/phase table (Braun 2026 Sec 4.2.4)."""
    table = []
    for index, record in enumerate(records):
        surprise = 1.0
        policy = record.get('policy')
        moves = record.get('moves', ())
        ply = record.get('ply', 0)
        if policy and ply < len(moves) and record.get('label_kind') in ('teacher_wdl', 'search'):
            p_played = float(policy.get(moves[ply], 0.0))
            surprise = min(2.0, -math.log(max(p_played, 0.05)))
        if ply >= 56 or str(record.get('source_id', '')).startswith('mco-'):
            surprise = min(2.0, surprise * 1.25)
        weight = 0.30 + 0.70 * surprise
        slots = max(1, min(6, int(round(weight * 2.0))))
        table.extend([index] * slots)
    return table


def _apply_history_mask(features, step, seed):
    """Mask a random suffix of history frames with 5% probability (Chessformer ICLR 2026 Sec E)."""
    masked = None
    batch_size = features.shape[0]
    for b in range(batch_size):
        h = ((step * 6364136223846793005) + (b * 1442695040888963407) + seed) & 0xFFFFFFFFFFFFFFFF
        if (h % 20) == 0:
            if masked is None:
                masked = features.clone()
            keep_frames = 1 + ((h >> 8) % 7)
            masked[b, :, keep_frames * 14:112] = 0.0
            if keep_frames == 1:
                masked[b, :, 12] = 0.0
    return masked if masked is not None else features


def train_stage(model, train, validation, output, *, config, stage, gate, stop=None,
                resume_mode='exact'):
    """Every exported candidate is evaluated; the gate owns champion selection.

    last.pt is a training-state checkpoint, never an accepted playing model.
    Resume validates data/config and restores the next minibatch RNG state.
    """
    output = Path(output)
    if resume_mode not in ('exact', 'portable'):
        raise ValueError('resume mode must be exact or portable')
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
    resume_history = []
    checkpoint = output / 'last.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['signature'] != signature:
            raise ValueError('resume configuration or dataset changed; use a new run directory')
        event = restore_checkpoint_rng(state, device, resume_mode=resume_mode,
                                       seed=config['seed'], checkpoint=checkpoint)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        resume_history = list(state.get('resume_history', []))
        if event['differences']:
            resume_history.append(event)
        start_step = state['step']
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    context = {'hostname': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               **resume_environment(device), 'python': sys.executable, 'git_revision': revision}

    def save(step):
        atomic_checkpoint(checkpoint, {'format_version': 1, 'signature': signature,
            'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'sampler_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'context': context, 'resume_history': resume_history})

    # Complete a pending candidate evaluation after a signal or failed gate run.
    def evaluate_step(step):
        candidate = output / f'candidate-{step:07d}'
        report = output / f'gate-{step:07d}.json'
        if not candidate.exists():
            save_checkpoint(model, candidate)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type == 'cuda' else []):
            gate(candidate, report)

    print(json.dumps({'stage': stage, 'status': 'evaluating_final_checkpoint' if start_step >= steps else 'training',
                      'completed_steps': start_step, 'target_steps': steps,
                      'remaining_steps': max(0, steps - start_step)}), flush=True)
    if start_step and (start_step % config['checkpoint_every'] == 0 or start_step == steps):
        evaluate_step(start_step)
    if start_step >= steps:
        return True
    base_lr = float(config['lr'])
    warmup_steps = min(500, max(10, steps // 20)) if steps >= 100 else 0
    priority_table = _priority_sampling_table(train) if steps >= 100 else None
    for step in range(start_step + 1, steps + 1):
        if steps >= 100:
            if step <= warmup_steps:
                current_lr = base_lr * (step / warmup_steps)
            else:
                progress = (step - warmup_steps) / max(1, steps - warmup_steps)
                current_lr = base_lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = current_lr
        model.train()
        if priority_table is not None:
            draws = torch.randint(len(priority_table), (batch_size,), generator=generator).tolist()
            indices = [priority_table[d] for d in draws]
        else:
            indices = torch.randint(len(train), (batch_size,), generator=generator).tolist()
        batch = dataset.batch(indices, device)
        if steps >= 100:
            batch = (_apply_history_mask(batch[0], step, config['seed']), batch[1], batch[2], batch[3])
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
