"""Train a U-Net on chips cut from user-labeled imagery."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.config import Config
from src.labels import (
    IGNORE_INDEX,
    chip_origins,
    encode_targets,
    pad_to_min_size,
    rasterize_labels,
    split_train_val,
)
from src.model import (
    ModelSpec,
    band_stats,
    build_unet,
    normalize,
    resolve_device,
    save_checkpoint,
)
from src.raster_io import read_raster

logger = logging.getLogger(__name__)


def augment(
    image: np.ndarray, target: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random one of the 8 rotations/flips of the square, which keep
    overhead imagery realistic."""
    k = int(rng.integers(4))
    image = np.rot90(image, k, axes=(1, 2))
    target = np.rot90(target, k, axes=(0, 1))
    if rng.random() < 0.5:
        image = image[:, :, ::-1]
        target = target[:, ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


class ChipDataset(Dataset):
    """Chips cut on the fly from one normalized image and its targets.

    Parameters
    ----------
    image : np.ndarray
        Normalized float32 ``(bands, H, W)`` image.
    targets : np.ndarray
        int64 ``(H, W)`` training indices, :data:`IGNORE_INDEX` = unlabeled.
    origins : np.ndarray
        ``(n, 2)`` chip upper-left corners.
    chip_size : int
        Chip edge length.
    jitter : int
        Maximum random shift of each chip, in pixels. 0 disables it.
    seed : int, optional
        Seed for jitter and augmentation; ``None`` disables augmentation.
    """

    def __init__(
        self,
        image: np.ndarray,
        targets: np.ndarray,
        origins: np.ndarray,
        chip_size: int,
        jitter: int = 0,
        seed: int | None = None,
    ) -> None:
        self.image = image
        self.targets = targets
        self.origins = origins
        self.chip_size = chip_size
        self.jitter = jitter
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = (int(v) for v in self.origins[idx])
        size = self.chip_size
        if self.rng is not None and self.jitter > 0:
            h, w = self.targets.shape
            row = int(
                np.clip(
                    row + self.rng.integers(-self.jitter, self.jitter + 1), 0, h - size
                )
            )
            col = int(
                np.clip(
                    col + self.rng.integers(-self.jitter, self.jitter + 1), 0, w - size
                )
            )
        image = self.image[:, row : row + size, col : col + size]
        target = self.targets[row : row + size, col : col + size]
        if self.rng is not None:
            image, target = augment(image, target, self.rng)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(
            np.ascontiguousarray(target)
        )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass over ``loader``; trains when an optimizer is given.

    Returns
    -------
    loss, accuracy : float
        Mean loss per batch and accuracy over labeled pixels.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, batches, correct, labeled = 0.0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            mask = targets != IGNORE_INDEX
            if not mask.any():
                continue
            logits = model(images)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            preds = logits.argmax(dim=1)
            correct += int((preds[mask] == targets[mask]).sum())
            labeled += int(mask.sum())
    if batches == 0:
        return math.nan, math.nan
    return total_loss / batches, correct / max(labeled, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    on_improved: Callable[[nn.Module], None],
) -> float:
    """Train with AdamW and cross-entropy that ignores unlabeled pixels.

    ``on_improved`` is called whenever the monitored loss (validation if
    available, otherwise training) reaches a new minimum.

    Returns
    -------
    float
        Best monitored loss.
    """
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best = math.inf
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        scheduler.step()
        monitored = train_loss
        msg = f"epoch {epoch}/{epochs}: train loss {train_loss:.4f} acc {train_acc:.3f}"
        if val_loader is not None:
            val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
            monitored = val_loss
            msg += f" | val loss {val_loss:.4f} acc {val_acc:.3f}"
        if monitored < best:
            best = monitored
            on_improved(model)
            msg += " (saved)"
        logger.info(msg)
    return best


def train_model(cfg: Config) -> ModelSpec:
    """Train a U-Net from the configured imagery and label polygons.

    The best checkpoint is written to ``cfg.training.model_path``.
    """
    t = cfg.training
    image, grid = read_raster(cfg.imagery.output)
    valid = image[0] != 0
    labels = rasterize_labels(
        cfg.labels.path, cfg.labels.class_field, grid, cfg.class_ids
    )
    labels[~valid] = 0
    missing = [c for c in cfg.class_ids if not (labels == c).any()]
    if missing:
        logger.warning("No labeled pixels for class ids %s", missing)

    mean, std = band_stats(image, valid)
    spec = ModelSpec(t.encoder, image.shape[0], cfg.class_ids, mean, std)
    norm = pad_to_min_size(normalize(image, spec), t.chip_size)
    targets = pad_to_min_size(
        encode_targets(labels, cfg.class_ids), t.chip_size, IGNORE_INDEX
    )

    origins = chip_origins(
        np.where(targets == IGNORE_INDEX, 0, 1),
        t.chip_size,
        t.chip_stride,
        t.min_labeled_fraction,
    )
    if len(origins) == 0:
        raise RuntimeError(
            "No chips have enough labeled pixels; draw more polygons or lower "
            "training.min_labeled_fraction"
        )
    rng = np.random.default_rng(t.seed)
    train_idx, val_idx = split_train_val(len(origins), t.val_fraction, rng)
    logger.info("Chips: %d train, %d validation", len(train_idx), len(val_idx))

    torch.manual_seed(t.seed)
    train_ds = ChipDataset(
        norm, targets, origins[train_idx], t.chip_size, t.chip_stride // 2, t.seed
    )
    sampler = None
    if t.samples_per_epoch:
        sampler = RandomSampler(
            train_ds, replacement=True, num_samples=t.samples_per_epoch
        )
    train_loader = DataLoader(
        train_ds, batch_size=t.batch_size, shuffle=sampler is None, sampler=sampler
    )
    val_loader = None
    if len(val_idx):
        val_ds = ChipDataset(norm, targets, origins[val_idx], t.chip_size)
        val_loader = DataLoader(val_ds, batch_size=t.batch_size)

    device = resolve_device(t.device)
    logger.info("Training on %s", device)
    model = build_unet(
        t.encoder, spec.in_channels, len(spec.class_ids), t.encoder_weights
    )
    fit(
        model,
        train_loader,
        val_loader,
        t.epochs,
        t.learning_rate,
        device,
        on_improved=lambda m: save_checkpoint(t.model_path, m, spec),
    )
    logger.info("Best model saved to %s", t.model_path)
    return spec
