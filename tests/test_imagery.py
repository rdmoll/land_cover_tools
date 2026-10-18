from datetime import datetime

import numpy as np
import pystac
import pytest

from src.imagery import latest_year_items, merge_first, target_grid, utm_epsg


@pytest.mark.parametrize(
    ("lon", "lat", "epsg"),
    [
        (-122.4, 47.6, 32610),
        (-75.0, 40.0, 32618),
        (151.2, -33.9, 32756),
        (179.9, 0, 32660),
    ],
)
def test_utm_epsg(lon, lat, epsg):
    assert utm_epsg(lon, lat) == epsg


def test_target_grid_is_one_meter_and_snapped():
    grid = target_grid((-122.42, 47.655, -122.405, 47.665), 1.0)
    assert grid.crs.to_epsg() == 32610
    assert (grid.transform.a, grid.transform.e) == (1.0, -1.0)
    left, bottom, right, top = grid.bounds
    assert all(float(v).is_integer() for v in (left, bottom, right, top))
    # ~1.1 km on each side
    assert 1100 < grid.width < 1200
    assert 1100 < grid.height < 1200


def make_item(item_id: str, year: int) -> pystac.Item:
    return pystac.Item(
        item_id, None, None, datetime(year, 6, 1), {"naip:year": str(year)}
    )


def test_latest_year_items_keeps_newest_year():
    items = [make_item("a", 2019), make_item("b", 2021), make_item("c", 2021)]
    assert [i.id for i in latest_year_items(items)] == ["b", "c"]
    assert latest_year_items([]) == []


def test_merge_first_only_fills_empty_pixels():
    dest = np.array([[[5, 0, 0]]], dtype=np.uint8)
    dest_valid = np.array([[True, False, False]])
    src = np.array([[[9, 7, 3]]], dtype=np.uint8)
    src_valid = np.array([[True, True, False]])
    merge_first(dest, dest_valid, src, src_valid)
    assert dest.tolist() == [[[5, 7, 0]]]
    assert dest_valid.tolist() == [[True, True, False]]
