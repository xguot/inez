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
from .model import ChessTransformer, ModelConfig, load_checkpoint, save_checkpoint
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
def validation_metrics(model, records, batch_size, device, limit=2048, seed=0):
    """Deterministic held-out metrics against the original, unsmoothed targets.

    WDL distances measure agreement with the supplied target distributions;
    teacher targets alone cannot establish calibration against real game outcomes.
    """
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError('validation batch size and sample limit must be positive')
    model.eval()
    dataset = RecordBatches(records)
    indices = random.Random(seed).sample(range(len(records)),
        len(records) if limit is None else min(limit, len(records)))
    sums = dict(policy_loss=0.0, wdl_cross_entropy=0.0, expected_value_mse=0.0,
                expected_value_mae=0.0, wdl_squared_distance=0.0,
                policy_top1_agreement=0.0, wdl_top1_agreement=0.0)
    policy_count = 0
    for start in range(0, len(indices), batch_size):
        features, mask, policy_target, wdl_target = dataset.batch(indices[start:start + batch_size], device)
        policy_logits, value_logits = model(features)
        policy_logits = policy_logits.masked_fill(~mask, -1e9)
        log_policy = F.log_softmax(policy_logits, dim=-1)
        log_wdl = F.log_softmax(value_logits, dim=-1)
        wdl = log_wdl.exp()
        rows = policy_target.sum(dim=-1) > 0
        policy_count += int(rows.sum().item())
        expected_diff = wdl[:, 0] - wdl[:, 2] - wdl_target[:, 0] + wdl_target[:, 2]
        values = {
            'policy_loss': -(policy_target * log_policy).sum(dim=-1)[rows].sum(),
            'wdl_cross_entropy': -(wdl_target * log_wdl).sum(),
            'expected_value_mse': expected_diff.square().sum(),
            'expected_value_mae': expected_diff.abs().sum(),
            'wdl_squared_distance': (wdl - wdl_target).square().sum(),
            'policy_top1_agreement': (policy_logits.argmax(-1)[rows] == policy_target.argmax(-1)[rows]).sum(),
            'wdl_top1_agreement': (wdl.argmax(-1) == wdl_target.argmax(-1)).sum(),
        }
        for key, value in values.items():
            sums[key] += float(value.item())
    metrics = {key: value / (max(1, policy_count) if key.startswith('policy_') else len(indices))
               for key, value in sums.items()}
    metrics['value_loss'] = metrics['wdl_cross_entropy'] + metrics['expected_value_mse']
    metrics['loss'] = metrics['policy_loss'] + metrics['value_loss']
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError('nonfinite validation metrics')
    return {**metrics, 'records': len(indices), 'policy_records': policy_count,
            'sampling_seed': seed, 'sample_indices_sha256': hashlib.sha256(
                json.dumps(indices, separators=(',', ':')).encode()).hexdigest()}


def validation_loss(model, records, batch_size, device, limit=256):
    """Compatibility helper for callers that only need the combined objective."""
    return validation_metrics(model, records, batch_size, device, limit)['loss']


def sample_batch_indices(primary_size, batch_size, generator, *, primary_table=None,
                         replay_size=0, replay_table=None, replay_fraction=None):
    """Draw fresh examples from each full pool, then shuffle with the saved RNG.

    Stochastic rounding makes the requested fraction unbiased even for small
    batches; priority weights only affect selection *within* each pool.
    """
    if primary_size < 1 or batch_size < 1:
        raise ValueError('empty primary pool or minibatch')
    if replay_fraction is None:
        replay_count = 0
        if replay_size:
            raise ValueError('replay fraction is required for a replay pool')
    else:
        if replay_size < 1 or not math.isfinite(replay_fraction) or not 0 < replay_fraction < 1:
            raise ValueError('replay requires a nonempty pool and fraction between zero and one')
        expected = batch_size * replay_fraction
        replay_count = math.floor(expected)
        if float(torch.rand((), generator=generator)) < expected - replay_count:
            replay_count += 1
    def draw(size, count, table):
        selected = torch.randint(len(table) if table is not None else size,
                                 (count,), generator=generator).tolist()
        return [table[index] for index in selected] if table is not None else selected
    indices = draw(primary_size, batch_size - replay_count, primary_table)
    if replay_count:
        indices += [primary_size + index for index in draw(replay_size, replay_count, replay_table)]
        order = torch.randperm(batch_size, generator=generator).tolist()
        indices = [indices[index] for index in order]
    return indices, replay_count


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
        if policy and ply < len(moves) and record.get('label_kind') == 'teacher_wdl':
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


def _hl_gauss_wdl(wdl_target, blend=0.15):
    """Project expected value onto 3 ordinal WDL bins via HL-Gauss (Farebrother et al. 2024 Listing 2)."""
    mean = wdl_target[:, 0] - wdl_target[:, 2]
    inv_s = 1.0 / (math.sqrt(2.0) * 0.45)
    c_m15 = torch.erf((-1.5 - mean) * inv_s)
    c_m05 = torch.erf((-0.5 - mean) * inv_s)
    c_p05 = torch.erf((0.5 - mean) * inv_s)
    c_p15 = torch.erf((1.5 - mean) * inv_s)
    z = (c_p15 - c_m15).clamp(min=1e-6)
    hl_probs = torch.stack(((c_p15 - c_p05) / z, (c_p05 - c_m05) / z, (c_m05 - c_m15) / z), dim=-1)
    smoothed = (1.0 - blend) * wdl_target + blend * hl_probs
    return smoothed / smoothed.sum(dim=-1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def _update_ema_state(ema_state, model, step, *, period=16, max_decay=0.992):
    """Update shadow EMA weights outside the optimizer loop (Morales-Brotons et al. 2024 Sec 4.1)."""
    t = max(1, step // period)
    alpha = min(max_decay, (t + 1.0) / (t + 10.0))
    for key, param in model.state_dict().items():
        if param.is_floating_point():
            ema_state[key].mul_(alpha).add_(param.detach(), alpha=1.0 - alpha)
        else:
            ema_state[key].copy_(param)


def training_signature(model, train, validation, config, stage, *, replay=None, replay_fraction=None):
    """Bind both ordinary resumes and completed-stage skips to identical inputs."""
    if not train or not validation:
        raise ValueError('empty training or validation records')
    if any(record['label_kind'] != 'game_result' for record in train) and stage == 'human':
        raise ValueError('human stage expects game-result initialization labels')
    if stage == 'teacher' and any(record['label_kind'] != 'teacher_wdl' for record in train):
        raise ValueError('teacher stage must replace both human policy and value targets')
    if stage == 'selfplay' and any(record['label_kind'] not in ('teacher_wdl', 'search') for record in train):
        raise ValueError('self-play updates may replay teacher labels, never human imitation labels')
    if replay is not None:
        if stage != 'selfplay' or not replay or any(r['label_kind'] != 'teacher_wdl' for r in replay):
            raise ValueError('self-play replay must contain the complete teacher training pool')
        if any(r['label_kind'] != 'search' for r in train):
            raise ValueError('mixed-pool self-play expects search records in its primary pool')
        if replay_fraction is None or not math.isfinite(replay_fraction) or not 0 < replay_fraction < 1:
            raise ValueError('invalid teacher replay fraction')
    elif replay_fraction is not None:
        raise ValueError('replay fraction requires replay records')
    data_id = fingerprint(train) + ':' + fingerprint(validation)
    signature = {'stage': stage, 'settings': config, 'model': model.metadata(), 'data': data_id}
    if replay is not None:
        signature['replay'] = {'data': fingerprint(replay), 'fraction': replay_fraction,
                               'sampler': 'full-corpus-mixture-v1'}
    return signature


def train_stage(model, train, validation, output, *, config, stage, gate, stop=None,
                resume_mode='exact', replay=None, replay_fraction=None):
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
    signature = training_signature(model, train, validation, config, stage,
                                   replay=replay, replay_fraction=replay_fraction)
    combined = list(train) + list(replay) if replay is not None else train
    dataset = RecordBatches(combined)
    generator = torch.Generator().manual_seed(config['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    start_step = 0
    resume_history = []
    sampler_counts = {'primary': 0, 'teacher_replay': 0}
    ema_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                 if steps >= 100 else None)
    checkpoint = output / 'last.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['signature'] != signature:
            raise ValueError('resume configuration or dataset changed; use a new run directory')
        event = restore_checkpoint_rng(state, device, resume_mode=resume_mode,
                                       seed=config['seed'], checkpoint=checkpoint)
        model.load_state_dict(state['model'])
        if ema_state is not None:
            saved_ema = state.get('ema_model', state['model'])
            ema_state = {k: v.to(device).detach().clone() for k, v in saved_ema.items()}
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        resume_history = list(state.get('resume_history', []))
        if event['differences']:
            resume_history.append(event)
        start_step = state['step']
        sampler_counts = dict(state.get('sampler_counts', sampler_counts))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    context = {'hostname': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               **resume_environment(device), 'python': sys.executable, 'git_revision': revision}

    def save(step):
        payload = {'format_version': 1, 'signature': signature,
            'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'sampler_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'context': context, 'resume_history': resume_history, 'sampler_counts': sampler_counts}
        if ema_state is not None:
            payload['ema_model'] = ema_state
        atomic_checkpoint(checkpoint, payload)

    # Complete a pending candidate evaluation after a signal or failed gate run.
    def evaluate_step(step):
        candidate = output / f'candidate-{step:07d}'
        report = output / f'gate-{step:07d}.json'
        if not candidate.exists():
            if ema_state is not None:
                raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state)
                try:
                    save_checkpoint(model, candidate)
                finally:
                    model.load_state_dict(raw_state)
            else:
                save_checkpoint(model, candidate)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type == 'cuda' else []):
            # Load the immutable artifact: this also validates its actual checksum.
            # The raw optimizer model is never replaced by EMA weights.
            exported = load_checkpoint(candidate, device=device, expected_config=model.config)
            metrics = validation_metrics(exported, validation, batch_size, device,
                limit=config.get('validation_limit', 2048), seed=config.get('validation_seed', 0))
            del exported
            manifest = json.loads((candidate / 'manifest.json').read_text())
            atomic_json(output / f'validation-{step:07d}.json', {
                'step': step, 'stage': stage, **metrics, 'context': context,
                'weights_kind': 'ema' if ema_state is not None else 'raw',
                'candidate': str(candidate.resolve()), 'weights_sha256': manifest['weights_sha256'],
                'validation_data_sha256': fingerprint(validation), 'sampler_counts': sampler_counts})
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
    replay_table = _priority_sampling_table(replay) if replay is not None and steps >= 100 else None
    hl_blend = 0.20 if stage == 'human' else 0.08
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
        indices, replay_count = sample_batch_indices(len(train), batch_size, generator,
            primary_table=priority_table, replay_size=len(replay) if replay is not None else 0,
            replay_table=replay_table, replay_fraction=replay_fraction)
        sampler_counts['primary'] += batch_size - replay_count
        sampler_counts['teacher_replay'] += replay_count
        batch = dataset.batch(indices, device)
        if steps >= 100:
            batch = (
                _apply_history_mask(batch[0], step, config['seed']),
                batch[1],
                batch[2],
                _hl_gauss_wdl(batch[3], blend=hl_blend),
            )
        optimizer.zero_grad(set_to_none=True)
        loss, policy_loss, value_loss = supervised_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
        optimizer.step()
        if ema_state is not None and (step % 16 == 0 or step == steps or step % config['checkpoint_every'] == 0):
            _update_ema_state(ema_state, model, step)
        if step == 1 or step % config.get('log_every', 20) == 0:
            print(json.dumps({'stage': stage, 'step': step, 'loss': float(loss.item()),
                              'policy_loss': float(policy_loss.item()), 'value_loss': float(value_loss.item()),
                              'sampler_counts': sampler_counts}), flush=True)
        due = step % config['checkpoint_every'] == 0 or step == steps
        if due or stop is not None and stop.requested:
            save(step)
        if stop is not None and stop.requested:
            print('Signal received; atomic training checkpoint saved. Resume with the same run directory.', flush=True)
            return False
        if due:
            evaluate_step(step)
    return True
