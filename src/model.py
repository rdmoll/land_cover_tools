"""U-Net model construction, normalization and checkpoints."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to rebuild a trained model and its inputs.

    Attributes
    ----------
    encoder : str
        smp encoder name, e.g. ``"resnet34"``.
    in_channels : int
        Number of image bands.
    class_ids : tuple of int
        Class ID for each output channel, in order.
    band_mean, band_std : tuple of float
        Per-band normalization statistics in raw image units.
    """

    encoder: str
    in_channels: int
    class_ids: tuple[int, ...]
    band_mean: tuple[float, ...]
    band_std: tuple[float, ...]


def resolve_device(name: str = "auto") -> torch.device:
    """Pick ``cuda``, then ``mps``, then ``cpu`` when ``name`` is ``auto``."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_unet(
    encoder: str, in_channels: int, num_classes: int, encoder_weights: str | None
) -> nn.Module:
    """Create a U-Net. With ImageNet weights, extra bands such as NIR get
    weights adapted from the RGB filters by smp."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )


def band_stats(
    image: np.ndarray, valid: np.ndarray
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-band mean and standard deviation over valid pixels."""
    pixels = image[:, valid].astype(np.float64)
    mean = pixels.mean(axis=1)
    std = np.maximum(pixels.std(axis=1), 1e-6)
    return tuple(float(m) for m in mean), tuple(float(s) for s in std)


def normalize(image: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """Standardize a ``(bands, h, w)`` image to float32."""
    mean = np.asarray(spec.band_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(spec.band_std, dtype=np.float32)[:, None, None]
    return (image.astype(np.float32) - mean) / std


def save_checkpoint(path: Path, model: nn.Module, spec: ModelSpec) -> None:
    """Save model weights together with their :class:`ModelSpec`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": asdict(spec), "state_dict": model.state_dict()}, path)


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, ModelSpec]:
    """Load a model saved by :func:`save_checkpoint`, in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    raw = ckpt["spec"]
    spec = ModelSpec(
        encoder=raw["encoder"],
        in_channels=int(raw["in_channels"]),
        class_ids=tuple(int(c) for c in raw["class_ids"]),
        band_mean=tuple(float(v) for v in raw["band_mean"]),
        band_std=tuple(float(v) for v in raw["band_std"]),
    )
    model = build_unet(spec.encoder, spec.in_channels, len(spec.class_ids), None)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, spec
