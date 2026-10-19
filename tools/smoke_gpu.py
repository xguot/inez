#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Verify selected-device execution, small-batch overfitting, and v1 artifact export."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import chess
import chess.pgn
import torch
from torch.nn import functional as F

from inez.encoding import (
    POLICY_SIZE,
    encode_position,
    legal_policy_indices,
    move_to_policy_index,
    wdl_from_result,
)
from inez.model import (
    ChessTransformer,
    ModelConfig,
    legal_masked_policy,
    load_checkpoint,
    save_checkpoint,
    wdl_probabilities_and_value,
)


def build_diagnostic_batch(root_dir: Path):
    pgn_path = root_dir / "games" / "recorded" / "2026-10-19-inez-vs-xiyuan-001.pgn"
    positions = []
    if pgn_path.exists():
        with pgn_path.open("r", encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
        if game is not None:
            result = game.headers.get("Result", "0-1")
            board = game.board()
            for move in list(game.mainline_moves())[:16]:
                positions.append((board.copy(stack=True), move, result))
                board.push(move)

    if not positions:
        board = chess.Board()
        for uci, res in [("e2e4", "1-0"), ("e7e5", "0-1"), ("g1f3", "1-0"), ("b8c6", "1/2-1/2")]:
            move = chess.Move.from_uci(uci)
            positions.append((board.copy(stack=True), move, res))
            board.push(move)

    features = []
    legal_masks = []
    target_moves = []
    target_wdls = []
    for board, move, result in positions:
        features.append(encode_position(board))
        mask = [False] * POLICY_SIZE
        for idx in legal_policy_indices(board):
            mask[idx] = True
        legal_masks.append(mask)
        target_moves.append(move_to_policy_index(board, move))
        target_wdls.append(wdl_from_result(result, board.turn))

    x = torch.tensor(features, dtype=torch.float32).reshape(len(positions), 64, 119)
    mask_t = torch.tensor(legal_masks, dtype=torch.bool)
    moves_t = torch.tensor(target_moves, dtype=torch.long)
    wdl_t = torch.tensor(target_wdls, dtype=torch.float32)
    return x, mask_t, moves_t, wdl_t


def compute_loss(model: ChessTransformer, x, mask_t, moves_t, wdl_t):
    policy_logits, wdl_logits = model(x)
    masked_logits = policy_logits.masked_fill(~mask_t, -1e9)
    policy_loss = F.cross_entropy(masked_logits, moves_t)
    wdl_log_probs = F.log_softmax(wdl_logits, dim=-1)
    value_loss = -(wdl_t * wdl_log_probs).sum(dim=-1).mean()
    return policy_loss + value_loss, policy_loss, value_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="directory for smoke output and artifact")
    parser.add_argument("--device", default="cuda", help="target torch device (cuda, mps, or cpu)")
    parser.add_argument("--steps", type=int, default=40, help="number of optimization steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--seed", type=int, default=20261019, help="random seed")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA device, but torch.cuda.is_available() is False")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    run_context = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "requested_device": args.device,
    }
    print(f"Smoke execution: {json.dumps(run_context, sort_keys=True)}", flush=True)
    root_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ModelConfig()
    model = ChessTransformer(config).to(device=device, dtype=torch.float32)
    device = next(model.parameters()).device
    param_count = sum(p.numel() for p in model.parameters())

    x, mask_t, moves_t, wdl_t = build_diagnostic_batch(root_dir)
    x = x.to(device)
    mask_t = mask_t.to(device)
    moves_t = moves_t.to(device)
    wdl_t = wdl_t.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start = time.monotonic()
    with torch.no_grad():
        initial_total, initial_policy, initial_value = [
            float(v.item()) for v in compute_loss(model, x, mask_t, moves_t, wdl_t)
        ]

    final_total = initial_total
    final_policy = initial_policy
    final_value = initial_value
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, p_loss, v_loss = compute_loss(model, x, mask_t, moves_t, wdl_t)
        loss.backward()
        optimizer.step()
        final_total = float(loss.item())
        final_policy = float(p_loss.item())
        final_value = float(v_loss.item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.monotonic() - start

    model.eval()
    with torch.no_grad():
        policy_logits, wdl_logits = model(x)
        probs = legal_masked_policy(policy_logits, mask_t)
        top_moves = probs.argmax(dim=-1)
        accuracy = float((top_moves == moves_t).to(torch.float32).mean().item())
        _, values = wdl_probabilities_and_value(wdl_logits)

    if not (final_total < initial_total * 0.5):
        raise SystemExit(
            f"Diagnostic overfit failed to halve loss: initial={initial_total:.4f}, final={final_total:.4f}"
        )

    artifact_dir = out_dir / "artifact"
    if artifact_dir.exists():
        import shutil
        shutil.rmtree(artifact_dir)
    save_checkpoint(model, artifact_dir)
    reloaded = load_checkpoint(artifact_dir, device=device, expected_config=config)
    with torch.no_grad():
        reload_policy, reload_wdl = reloaded(x)
        max_policy_diff = float((reload_policy - policy_logits).abs().max().item())
        max_wdl_diff = float((reload_wdl - wdl_logits).abs().max().item())
    if max_policy_diff > 1e-5 or max_wdl_diff > 1e-5:
        raise SystemExit(
            f"Reloaded artifact mismatch: policy_diff={max_policy_diff}, wdl_diff={max_wdl_diff}"
        )

    summary = {
        "run_context": run_context,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": param_count,
        "batch_size": int(x.shape[0]),
        "steps": args.steps,
        "elapsed_sec": round(elapsed_s, 4),
        "initial_loss": {
            "total": round(initial_total, 4),
            "policy": round(initial_policy, 4),
            "value": round(initial_value, 4),
        },
        "final_loss": {
            "total": round(final_total, 4),
            "policy": round(final_policy, 4),
            "value": round(final_value, 4),
        },
        "policy_top1_accuracy": round(accuracy, 4),
        "mean_predicted_value": round(float(values.mean().item()), 4),
        "artifact_dir": str(artifact_dir),
        "reload_max_abs_diff": {
            "policy": max_policy_diff,
            "wdl": max_wdl_diff,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
