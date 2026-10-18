import numpy as np
import pytest

from src.raster_io import hex_to_rgba, read_raster, write_raster


def test_roundtrip_with_colormap(tmp_path, grid):
    data = (
        np.random.default_rng(0)
        .integers(0, 4, (grid.height, grid.width))
        .astype(np.uint8)
    )
    path = tmp_path / "out" / "classes.tif"
    write_raster(
        path,
        data,
        grid,
        nodata=0,
        colormap={1: (255, 0, 0, 255)},
        descriptions=("land_cover",),
    )
    back, back_grid = read_raster(path)
    np.testing.assert_array_equal(back[0], data)
    assert back_grid == grid


def test_shape_mismatch_rejected(tmp_path, grid):
    with pytest.raises(ValueError, match="does not match"):
        write_raster(tmp_path / "x.tif", np.zeros((3, 3), np.uint8), grid)


def test_hex_to_rgba():
    assert hex_to_rgba("#1f78b4") == (31, 120, 180, 255)
    with pytest.raises(ValueError):
        hex_to_rgba("red")
