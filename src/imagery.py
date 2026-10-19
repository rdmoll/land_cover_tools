"""Download imagery from Microsoft Planetary Computer onto a 1 m grid."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence

import numpy as np
import planetary_computer
import pystac
import pystac_client
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.warp import reproject, transform_bounds
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window, from_bounds

from src.config import ImageryConfig
from src.raster_io import GeoGrid, write_raster

logger = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing a point."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def target_grid(
    bbox: tuple[float, float, float, float], resolution: float = 1.0
) -> GeoGrid:
    """Build a north-up UTM grid covering a lon/lat bounding box.

    The grid edges are snapped to whole multiples of ``resolution`` so that
    pixel ``(row, col)`` always covers the same square on the ground.

    Parameters
    ----------
    bbox : tuple of float
        ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
    resolution : float
        Pixel size in meters.

    Returns
    -------
    GeoGrid
        Grid in the UTM zone of the bbox center.
    """
    lon_c = (bbox[0] + bbox[2]) / 2
    lat_c = (bbox[1] + bbox[3]) / 2
    crs = CRS.from_epsg(utm_epsg(lon_c, lat_c))
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    left = math.floor(left / resolution) * resolution
    bottom = math.floor(bottom / resolution) * resolution
    right = math.ceil(right / resolution) * resolution
    top = math.ceil(top / resolution) * resolution
    width = round((right - left) / resolution)
    height = round((top - bottom) / resolution)
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
    return GeoGrid(crs, transform, width, height)


def centered_grid(
    lon: float, lat: float, size_m: float, resolution: float = 1.0
) -> GeoGrid:
    """Build a square north-up UTM grid centered on a lon/lat point.

    Parameters
    ----------
    lon, lat : float
        Center in WGS84 degrees.
    size_m : float
        Edge length of the square in meters, rounded to whole pixels.
    resolution : float
        Pixel size in meters.

    Returns
    -------
    GeoGrid
        Grid in the UTM zone of the center, with edges snapped to multiples
        of ``resolution``.
    """
    crs = CRS.from_epsg(utm_epsg(lon, lat))
    (x,), (y,) = transform_coords("EPSG:4326", crs, [lon], [lat])
    n = round(size_m / resolution)
    left = round((x - n * resolution / 2) / resolution) * resolution
    top = round((y + n * resolution / 2) / resolution) * resolution
    affine = Affine(resolution, 0.0, left, 0.0, -resolution, top)
    return GeoGrid(crs, affine, n, n)


def grid_for_config(cfg: ImageryConfig) -> GeoGrid:
    """The output grid described by an imagery config."""
    if cfg.bbox is not None:
        return target_grid(cfg.bbox, cfg.resolution)
    lon, lat = cfg.center
    return centered_grid(lon, lat, cfg.size_m, cfg.resolution)


def wgs84_bounds(grid: GeoGrid) -> tuple[float, float, float, float]:
    """Lon/lat bounding box that encloses a grid."""
    return transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)


def item_year(item: pystac.Item) -> int:
    """Acquisition year of a STAC item."""
    if "naip:year" in item.properties:
        return int(item.properties["naip:year"])
    if item.datetime is not None:
        return item.datetime.year
    return int(str(item.properties["start_datetime"])[:4])


def latest_year_items(items: Iterable[pystac.Item]) -> list[pystac.Item]:
    """Keep only the items from the most recent acquisition year.

    Mixing years would give a patchwork of seasons and sensors, which hurts
    classification, so one year is mosaicked at a time.
    """
    items = list(items)
    if not items:
        return []
    year = max(item_year(i) for i in items)
    return [i for i in items if item_year(i) == year]


def search_items(
    collection: str,
    bbox: tuple[float, float, float, float],
    datetime: str | None = None,
) -> list[pystac.Item]:
    """Search Planetary Computer for items intersecting a lon/lat bbox."""
    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )
    search = catalog.search(
        collections=[collection], bbox=list(bbox), datetime=datetime
    )
    items = list(search.items())
    logger.info("Found %d %s items for bbox %s", len(items), collection, bbox)
    return items


def merge_first(
    dest: np.ndarray, dest_valid: np.ndarray, src: np.ndarray, src_valid: np.ndarray
) -> None:
    """Fill still-empty pixels of ``dest`` from ``src`` in place."""
    fill = src_valid & ~dest_valid
    dest[:, fill] = src[:, fill]
    dest_valid |= fill


def _warp_item(
    href: str, grid: GeoGrid, bands: Sequence[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the part of one remote COG that overlaps the grid and warp it."""
    with rasterio.open(href) as src:
        src_bounds = transform_bounds(grid.crs, src.crs, *grid.bounds, densify_pts=21)
        window = from_bounds(*src_bounds, transform=src.transform)
        window = Window(
            math.floor(window.col_off) - 2,
            math.floor(window.row_off) - 2,
            math.ceil(window.width) + 4,
            math.ceil(window.height) + 4,
        )
        try:
            window = window.intersection(Window(0, 0, src.width, src.height))
        except WindowError:
            return None
        data = src.read(list(bands), window=window)
        src_transform = src.window_transform(window)
        src_crs = src.crs

    out = np.zeros((len(bands), grid.height, grid.width), dtype=data.dtype)
    reproject(
        data,
        out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.average,
    )
    coverage = np.zeros((grid.height, grid.width), dtype=np.uint8)
    reproject(
        np.ones(data.shape[1:], dtype=np.uint8),
        coverage,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.nearest,
    )
    return out, coverage.astype(bool)


def download_imagery(cfg: ImageryConfig) -> GeoGrid:
    """Download, mosaic and resample imagery for the configured area.

    The result is written to ``cfg.output`` as a uint8 GeoTIFF on a UTM grid
    with ``cfg.resolution`` meter pixels. Pixels with no coverage are 0 in
    every band (the file's nodata value); covered pixels are at least 1.

    Returns
    -------
    GeoGrid
        Georeferencing of the written image.
    """
    grid = grid_for_config(cfg)
    bbox = wgs84_bounds(grid)
    logger.info(
        "Target grid: %s, %d x %d px at %.2f m",
        grid.crs,
        grid.width,
        grid.height,
        cfg.resolution,
    )
    items = latest_year_items(search_items(cfg.collection, bbox, cfg.datetime))
    if not items:
        raise RuntimeError(
            f"No {cfg.collection} imagery found for bbox {bbox}. NAIP only "
            "covers the contiguous United States."
        )
    logger.info("Using %d items from %d", len(items), item_year(items[0]))

    mosaic = np.zeros((len(cfg.bands), grid.height, grid.width), dtype=np.uint8)
    valid = np.zeros((grid.height, grid.width), dtype=bool)
    for item in items:
        logger.info("Reading %s", item.id)
        warped = _warp_item(item.assets["image"].href, grid, cfg.bands)
        if warped is None:
            continue
        data, coverage = warped
        merge_first(mosaic, valid, data.astype(np.uint8), coverage)
        if valid.all():
            break

    if not valid.any():
        raise RuntimeError("Imagery items were found but none covered the area")
    logger.info("Coverage: %.1f%% of the grid", 100 * valid.mean())
    mosaic[:, valid] = np.maximum(mosaic[:, valid], 1)
    write_raster(cfg.output, mosaic, grid, nodata=0)
    return grid
