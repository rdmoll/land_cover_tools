"""Run a trained U-Net over a full image and write the land-cover GeoTIFF."""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import Config
from src.labels import decode_predictions
from src.model import load_checkpoint, normalize, resolve_device
from src.raster_io import hex_to_rgba, read_raster, write_raster

logger = logging.getLogger(__name__)


def predict_indices(
    model: nn.Module,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    batch_size: int = 4,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Predict a class index for every pixel with overlapping tiles.

    Each tile keeps only its central ``tile_size - 2 * overlap`` pixels, so
    every output pixel comes from a prediction with at least ``overlap``
    pixels of context on all sides. The image is reflect-padded at the edges.

    Parameters
    ----------
    model : nn.Module
        Segmentation model mapping ``(n, bands, s, s)`` to ``(n, classes, s, s)``.
    image : np.ndarray
        Normalized float32 ``(bands, H, W)`` image.
    tile_size : int
        Tile edge length passed to the model (multiple of 32 for U-Net).
    overlap : int
        Context margin discarded on each side of a tile.
    batch_size : int
        Tiles per forward pass.
    device : torch.device or str
        Device to run on.

    Returns
    -------
    np.ndarray
        int64 ``(H, W)`` array of class indices.
    """
    _, height, width = image.shape
    core = tile_size - 2 * overlap
    n_rows, n_cols = math.ceil(height / core), math.ceil(width / core)
    padded = np.pad(
        image,
        (
            (0, 0),
            (overlap, n_rows * core - height + overlap),
            (overlap, n_cols * core - width + overlap),
        ),
        mode="reflect",
    )
    out = np.zeros((n_rows * core, n_cols * core), dtype=np.int64)
    origins = [(r * core, c * core) for r in range(n_rows) for c in range(n_cols)]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(origins), batch_size):
            batch = origins[start : start + batch_size]
            tiles = np.stack(
                [padded[:, r : r + tile_size, c : c + tile_size] for r, c in batch]
            )
            logits = model(torch.from_numpy(tiles).to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            for (r, c), pred in zip(batch, preds):
                out[r : r + core, c : c + core] = pred[
                    overlap : overlap + core, overlap : overlap + core
                ]
            logger.debug("Predicted %d/%d tiles", start + len(batch), len(origins))
    return out[:height, :width]


def write_legend(path: Path, cfg: Config) -> None:
    """Write a CSV listing each class ID, name and color."""
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "name", "color"])
        writer.writerow([0, "no_data", ""])
        for c in cfg.classes:
            writer.writerow([c.class_id, c.name, c.color or ""])


def predict_landcover(cfg: Config) -> Path:
    """Classify the configured imagery and write the land-cover GeoTIFF.

    The output is on the same 1 m grid as the input imagery. Each pixel holds
    its class ID as an unsigned integer (uint8, or uint16 for IDs above 255);
    0 marks pixels with no imagery. A ``.csv`` legend is written next to it.

    Returns
    -------
    Path
        Path of the written GeoTIFF.
    """
    p = cfg.prediction
    device = resolve_device(p.device)
    model, spec = load_checkpoint(cfg.training.model_path, device)
    if set(spec.class_ids) != set(cfg.class_ids):
        raise ValueError(
            f"model was trained on class ids {list(spec.class_ids)} but the "
            f"config defines {list(cfg.class_ids)}; retrain the model"
        )
    image, grid = read_raster(cfg.imagery.output)
    if image.shape[0] != spec.in_channels:
        raise ValueError(
            f"model expects {spec.in_channels} bands, image has {image.shape[0]}"
        )
    valid = image[0] != 0
    logger.info("Predicting %d x %d px on %s", grid.width, grid.height, device)
    indices = predict_indices(
        model, normalize(image, spec), p.tile_size, p.overlap, p.batch_size, device
    )
    classes = decode_predictions(indices, spec.class_ids)
    classes[~valid] = 0

    colormap = None
    if all(c.color for c in cfg.classes):
        colormap = {0: (0, 0, 0, 0)} | {
            c.class_id: hex_to_rgba(c.color) for c in cfg.classes
        }
    write_raster(
        p.output,
        classes,
        grid,
        nodata=0,
        colormap=colormap,
        descriptions=("land_cover",),
    )
    legend = p.output.with_suffix(".csv")
    write_legend(legend, cfg)
    counts = {c.name: int((classes == c.class_id).sum()) for c in cfg.classes}
    logger.info("Pixels per class: %s", counts)
    logger.info("Legend written to %s", legend)
    return p.output
