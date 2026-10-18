"""Turn user-drawn label polygons into training chips."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from src.raster_io import GeoGrid

logger = logging.getLogger(__name__)

UNLABELED = 0
IGNORE_INDEX = -1


def rasterize_labels(
    path: Path, class_field: str, grid: GeoGrid, class_ids: Sequence[int]
) -> np.ndarray:
    """Burn label polygons onto the image grid.

    Parameters
    ----------
    path : Path
        Vector file (GeoPackage, GeoJSON, Shapefile, ...) with polygons.
    class_field : str
        Attribute holding each polygon's integer class ID.
    grid : GeoGrid
        Grid of the imagery the labels were drawn on.
    class_ids : sequence of int
        Allowed class IDs from the config.

    Returns
    -------
    np.ndarray
        uint16 array of shape ``(height, width)``; 0 where nothing is labeled.
        Where polygons overlap, the one later in the file wins.
    """
    gdf = gpd.read_file(path)
    if class_field not in gdf.columns:
        raise ValueError(
            f"{path} has no '{class_field}' column; found {list(gdf.columns)}"
        )
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS; save it with the image's CRS")
    gdf = gdf.to_crs(grid.crs)
    values = gdf[class_field].astype(int).to_numpy()
    unknown = sorted(int(v) for v in set(values) - set(class_ids))
    if unknown:
        raise ValueError(
            f"labels use class ids {unknown} that are not defined in the config "
            f"(defined: {list(class_ids)})"
        )
    labels = rasterize(
        zip(gdf.geometry, values),
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=UNLABELED,
        dtype="uint16",
    )
    counts = {int(c): int((labels == c).sum()) for c in class_ids}
    logger.info("Labeled pixels per class: %s", counts)
    return labels


def encode_targets(labels: np.ndarray, class_ids: Sequence[int]) -> np.ndarray:
    """Map class IDs to contiguous training indices.

    Class ``class_ids[i]`` becomes index ``i``; unlabeled pixels become
    :data:`IGNORE_INDEX` so the loss skips them.
    """
    lookup = np.full(int(max(class_ids)) + 1, IGNORE_INDEX, dtype=np.int64)
    lookup[np.asarray(class_ids)] = np.arange(len(class_ids))
    return lookup[labels]


def decode_predictions(indices: np.ndarray, class_ids: Sequence[int]) -> np.ndarray:
    """Map training indices back to class IDs as the smallest fitting uint."""
    dtype = np.uint8 if max(class_ids) <= 255 else np.uint16
    return np.asarray(class_ids, dtype=dtype)[indices]


def pad_to_min_size(array: np.ndarray, size: int, value: float = 0) -> np.ndarray:
    """Pad the last two axes at the bottom/right up to ``size`` if smaller."""
    pad_h = max(0, size - array.shape[-2])
    pad_w = max(0, size - array.shape[-1])
    if pad_h == 0 and pad_w == 0:
        return array
    pad = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
    return np.pad(array, pad, constant_values=value)


def chip_origins(
    labels: np.ndarray, chip_size: int, stride: int, min_labeled_fraction: float
) -> np.ndarray:
    """Find chip windows that contain enough labeled pixels.

    Windows are laid on a regular grid with the given stride; the last row
    and column are shifted so chips always end at the image edge.

    Parameters
    ----------
    labels : np.ndarray
        ``(height, width)`` label array, 0 meaning unlabeled. Must be at
        least ``chip_size`` in both dimensions.
    chip_size, stride : int
        Chip edge length and spacing in pixels.
    min_labeled_fraction : float
        Minimum share of labeled pixels a chip needs to be kept.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` int array of ``(row, col)`` upper-left corners.
    """
    height, width = labels.shape
    if height < chip_size or width < chip_size:
        raise ValueError("labels are smaller than chip_size; pad them first")

    def starts(n: int) -> np.ndarray:
        s = np.arange(0, n - chip_size + 1, stride)
        return np.unique(np.append(s, n - chip_size))

    rows, cols = starts(height), starts(width)
    # Integral image gives every window's labeled count without a Python loop.
    labeled = (labels != UNLABELED).astype(np.int64)
    integral = np.pad(labeled.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r0, c0 = np.meshgrid(rows, cols, indexing="ij")
    r1, c1 = r0 + chip_size, c0 + chip_size
    counts = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    keep = counts >= min_labeled_fraction * chip_size * chip_size
    keep &= counts > 0
    return np.stack([r0[keep], c0[keep]], axis=1)


def split_train_val(
    n: int, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split ``n`` chip indices into train and validation sets.

    At least one chip always stays in training; validation is empty when
    there are fewer than two chips.
    """
    order = rng.permutation(n)
    n_val = round(n * val_fraction) if n >= 2 else 0
    n_val = min(n_val, n - 1)
    return np.sort(order[n_val:]), np.sort(order[:n_val])
