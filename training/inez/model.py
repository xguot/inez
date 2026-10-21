"""An initial, chess-specific transformer and strict portable weight artifacts.

This is an Inez baseline, not an exact Chessformer reproduction. Square tokens
use pre-norm attention, learned two-dimensional relative geometry, and a shared
board-conditioned generator of per-head attention biases. Search, training and
playing-strength validation are separate steps; a random model is not a player.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoding import FEATURE_COUNT, POLICY_SIZE, encoding_metadata

ARCHITECTURE = "inez-square-transformer-v1"
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    layers: int = 6
    heads: int = 8
    ffn_ratio: float = 1.5
    dynamic_rank: int = 32
    policy_width: int = 64
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("width", "layers", "heads", "dynamic_rank", "policy_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.width % self.heads:
            raise ValueError("model width must be divisible by the head count")
        if (not isinstance(self.ffn_ratio, (float, int)) or isinstance(self.ffn_ratio, bool) or
                not math.isfinite(self.ffn_ratio) or self.ffn_ratio <= 0 or
                int(self.width * self.ffn_ratio) < 1):
            raise ValueError("ffn_ratio must be finite and create a nonempty FFN")
        if (not isinstance(self.dropout, (float, int)) or isinstance(self.dropout, bool) or
                not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or set(values) != {field.name for field in fields(cls)}:
            raise ValueError("model configuration fields do not match architecture v1")
        return cls(**values)


class GeometryBias(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.relative = nn.Parameter(torch.zeros(config.heads, 15 * 15))
        ranks = torch.arange(64) // 8
        files = torch.arange(64) % 8
        indices = ((ranks[:, None] - ranks[None, :] + 7) * 15 +
                   files[:, None] - files[None, :] + 7)
        self.register_buffer("relative_indices", indices, persistent=False)
        self.dynamic = nn.Sequential(
            nn.Linear(config.width, config.dynamic_rank), nn.GELU(),
            nn.Linear(config.dynamic_rank, config.heads * 64 * 64),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        relative = self.relative[:, self.relative_indices]
        dynamic = self.dynamic(tokens.mean(dim=1)).reshape(-1, self.heads, 64, 64)
        return dynamic + relative.unsqueeze(0)


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.dropout = config.dropout
        self.attention_norm = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)
        self.ffn_norm = nn.LayerNorm(config.width)
        hidden = int(config.width * config.ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(hidden, config.width),
        )

    def forward(self, tokens: Tensor, geometry: Tensor) -> Tensor:
        batch = tokens.shape[0]
        qkv = self.qkv(self.attention_norm(tokens)).reshape(
            batch, 64, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = F.scaled_dot_product_attention(
            query, key, value, attn_mask=geometry,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, 64, -1)
        tokens = tokens + F.dropout(self.output(attention), p=self.dropout, training=self.training)
        return tokens + F.dropout(self.ffn(self.ffn_norm(tokens)), p=self.dropout, training=self.training)


class ChessTransformer(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config
        self.input_projection = nn.Linear(FEATURE_COUNT, config.width)
        self.square_embedding = nn.Parameter(torch.empty(1, 64, config.width))
        nn.init.normal_(self.square_embedding, std=0.02)
        self.geometry = GeometryBias(config)
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width)
        self.policy_from = nn.Linear(config.width, config.policy_width, bias=False)
        self.policy_to = nn.Linear(config.width, config.policy_width, bias=False)
        self.underpromotions = nn.Linear(config.width, 192)
        self.value_head = nn.Sequential(
            nn.Linear(config.width, config.width), nn.GELU(), nn.Linear(config.width, 3),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or tuple(features.shape[1:]) != (64, FEATURE_COUNT):
            raise ValueError("features must have shape [batch, 64, 119]")
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite floating-point values")
        tokens = self.input_projection(features) + self.square_embedding
        geometry = self.geometry(tokens)
        for block in self.blocks:
            tokens = block(tokens, geometry)
        tokens = self.final_norm(tokens)
        query = self.policy_from(tokens)
        key = self.policy_to(tokens)
        from_to = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.config.policy_width)
        occupied = features[:, :, :12].sum(dim=-1, keepdim=True)
        occupied_mean = (tokens * occupied).sum(dim=1) / occupied.sum(dim=1).clamp(min=1.0)
        piece_diff = features[:, :, :6].sum(dim=1) - features[:, :, 6:12].sum(dim=1)
        piece_weights = self.input_projection.weight[:, :6] - self.input_projection.weight[:, 6:12]
        piece_balance = F.linear(piece_diff, piece_weights) * 4.0
        pooled = tokens.mean(dim=1) + occupied_mean + piece_balance
        promotion_pooled = pooled + tokens[:, 48:64].mean(dim=1)
        policy = torch.cat((from_to.flatten(start_dim=1), self.underpromotions(promotion_pooled)), dim=1)
        return policy, self.value_head(pooled)

    def metadata(self):
        return {
            "architecture": ARCHITECTURE,
            "config": asdict(self.config),
            "encoding": encoding_metadata(),
        }


def legal_masked_policy(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Mask illegal moves and normalize; no legal moves returns finite zeros.

    A forced-draw position can still have legal moves. The caller must check
    game outcome before asking search to use this policy.
    """
    if logits.ndim != 2 or logits.shape[1] != POLICY_SIZE:
        raise ValueError("policy logits must have shape [batch, 4288]")
    if legal_mask.shape != logits.shape or legal_mask.dtype != torch.bool:
        raise ValueError("legal mask must be boolean with the same shape as logits")
    if logits.device != legal_mask.device:
        raise ValueError("policy logits and mask must use the same device")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits[legal_mask]).all()):
        raise ValueError("legal policy logits must be finite floating-point values")
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    empty = ~legal_mask.any(dim=1, keepdim=True)
    masked = masked.masked_fill(empty, 0.0)
    return masked.softmax(dim=1) * legal_mask.to(dtype=logits.dtype)


def wdl_probabilities_and_value(logits: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("WDL logits must have shape [batch, 3]")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("WDL logits must be finite floating-point values")
    probabilities = logits.softmax(dim=-1)
    return probabilities, probabilities[:, 0] - probabilities[:, 2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(model: ChessTransformer, directory) -> Path:
    """Write a new portable artifact directory (model weights, not a resumable
    training checkpoint). Export canonical float32 weights and refuse to
    overwrite existing artifacts.
    """
    if not isinstance(model, ChessTransformer):
        raise TypeError("expected a ChessTransformer")
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("artifact directory already exists: {}".format(destination))
    staging = Path(tempfile.mkdtemp(prefix=".inez-artifact-", dir=str(destination.parent)))
    try:
        state = {key: value.detach().cpu().to(dtype=torch.float32).clone()
                 for key, value in model.state_dict().items()}
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("cannot export nonfinite model parameters")
        weights = staging / "weights.pt"
        torch.save(state, weights)
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            **model.metadata(),
            "weights": "weights.pt",
            "weights_sha256": _sha256(weights),
            "torch_version": str(torch.__version__),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Model artifacts become visible only after both weights and manifest exist.
        staging.rename(destination)
    except BaseException:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_checkpoint(directory, *, device="cpu", expected_config=None) -> ChessTransformer:
    """Load only a supported JSON manifest and a strict tensor state dictionary.
    The restricted weights-only loader does not permit arbitrary pickle objects.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    keys = {"artifact_version", "architecture", "config", "encoding", "weights",
            "weights_sha256", "torch_version"}
    if not isinstance(manifest, dict) or set(manifest) != keys:
        raise ValueError("artifact manifest fields do not match version 1")
    if type(manifest["artifact_version"]) is not int or manifest["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("unsupported artifact version")
    if manifest["architecture"] != ARCHITECTURE:
        raise ValueError("unsupported model architecture")
    # JSON comparison also distinguishes booleans from integer version values.
    if json.dumps(manifest["encoding"], sort_keys=True, allow_nan=False) != json.dumps(
            encoding_metadata(), sort_keys=True, allow_nan=False):
        raise ValueError("incompatible chess encoding contract")
    if not isinstance(manifest["torch_version"], str) or not manifest["torch_version"]:
        raise ValueError("artifact must record its exporting torch version")
    config = ModelConfig.from_dict(manifest["config"])
    if expected_config is not None and config != expected_config:
        raise ValueError("artifact configuration does not match expected configuration")
    if manifest["weights"] != "weights.pt":
        raise ValueError("unsupported weight filename")
    digest = manifest["weights_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid weight checksum")
    weights = directory / "weights.pt"
    if _sha256(weights) != digest:
        raise ValueError("model weight checksum mismatch")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(key, str) or not isinstance(value, Tensor)
                                          for key, value in state.items()):
        raise ValueError("weights must be a plain tensor state dictionary")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("model weights contain nonfinite values")
    model = ChessTransformer(config).to(dtype=torch.float32)
    for name, expected in model.state_dict().items():
        if name in state and (state[name].shape != expected.shape or state[name].dtype != expected.dtype):
            raise ValueError("model tensor shape or dtype mismatch for {}".format(name))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
