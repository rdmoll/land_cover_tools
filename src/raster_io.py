"""Read and write GeoTIFFs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoGrid:
    """The georeferencing of a raster: CRS, affine transform and size."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` in CRS units."""
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (self.width, self.height)
        return (left, bottom, right, top)


def read_raster(path: Path) -> tuple[np.ndarray, GeoGrid]:
    """Read every band of a GeoTIFF.

    Returns
    -------
    data : np.ndarray
        Array of shape ``(bands, height, width)``.
    grid : GeoGrid
        Georeferencing of the raster.
    """
    with rasterio.open(path) as src:
        data = src.read()
        grid = GeoGrid(src.crs, src.transform, src.width, src.height)
    return data, grid


def hex_to_rgba(color: str) -> tuple[int, int, int, int]:
    """Convert ``#RRGGBB`` to an RGBA tuple with full opacity."""
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"color {color!r} is not in #RRGGBB form")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)


def write_raster(
    path: Path,
    data: np.ndarray,
    grid: GeoGrid,
    nodata: float | None = None,
    colormap: dict[int, tuple[int, int, int, int]] | None = None,
    descriptions: tuple[str, ...] | None = None,
) -> None:
    """Write an array to a tiled, compressed GeoTIFF.

    Parameters
    ----------
    path : Path
        Destination file; parent directories are created.
    data : np.ndarray
        Array of shape ``(height, width)`` or ``(bands, height, width)``.
    grid : GeoGrid
        Georeferencing. Its size must match ``data``.
    nodata : float, optional
        Nodata value stored in the file.
    colormap : dict, optional
        Color table for band 1, mapping pixel value to RGBA.
    descriptions : tuple of str, optional
        Per-band descriptions.
    """
    if data.ndim == 2:
        data = data[np.newaxis]
    if data.shape[1:] != (grid.height, grid.width):
        raise ValueError(
            f"data shape {data.shape[1:]} does not match grid "
            f"({grid.height}, {grid.width})"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "count": data.shape[0],
        "height": grid.height,
        "width": grid.width,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    if data.shape[0] > 1:
        profile["interleave"] = "pixel"
    if colormap:
        profile["photometric"] = "palette"
    with rasterio.open(path, "w", **profile) as dst:
        # The color table must be set before any pixels are written.
        if colormap:
            dst.write_colormap(1, colormap)
        dst.write(data)
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)
    logger.info("Wrote %s (%d x %d, %s)", path, grid.width, grid.height, data.dtype)
