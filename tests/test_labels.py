import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from src.labels import (
    IGNORE_INDEX,
    chip_origins,
    decode_predictions,
    encode_targets,
    pad_to_min_size,
    rasterize_labels,
    split_train_val,
)


def write_polygons(path, grid, polygons, crs=None):
    left, _, _, top = grid.bounds
    geoms = [
        box(left + c0, top - r1, left + c1, top - r0)
        for _, (r0, c0, r1, c1) in polygons
    ]
    gdf = gpd.GeoDataFrame(
        {"class_id": [c for c, _ in polygons]}, geometry=geoms, crs=grid.crs
    )
    if crs is not None:
        gdf = gdf.to_crs(crs)
    gdf.to_file(path)


def test_rasterize_labels_burns_classes(tmp_path, grid):
    path = tmp_path / "labels.gpkg"
    write_polygons(path, grid, [(2, (0, 0, 10, 20)), (7, (20, 30, 30, 40))])
    labels = rasterize_labels(path, "class_id", grid, [2, 7])
    assert labels.dtype == np.uint16
    assert (labels[0:10, 0:20] == 2).all()
    assert (labels[20:30, 30:40] == 7).all()
    assert (labels == 0).sum() == grid.width * grid.height - 200 - 100


def test_rasterize_labels_reprojects(tmp_path, grid):
    path = tmp_path / "labels.geojson"
    write_polygons(path, grid, [(1, (10, 10, 30, 30))], crs="EPSG:4326")
    labels = rasterize_labels(path, "class_id", grid, [1])
    assert abs(int((labels == 1).sum()) - 400) <= 40


def test_rasterize_labels_rejects_unknown_class(tmp_path, grid):
    path = tmp_path / "labels.gpkg"
    write_polygons(path, grid, [(9, (0, 0, 5, 5))])
    with pytest.raises(ValueError, match=r"\[9\]"):
        rasterize_labels(path, "class_id", grid, [1, 2])


def test_rasterize_labels_requires_field(tmp_path, grid):
    path = tmp_path / "labels.gpkg"
    write_polygons(path, grid, [(1, (0, 0, 5, 5))])
    with pytest.raises(ValueError, match="no 'cls' column"):
        rasterize_labels(path, "cls", grid, [1])


def test_encode_decode_roundtrip():
    class_ids = [3, 10, 300]
    labels = np.array([[0, 3], [10, 300]], dtype=np.uint16)
    targets = encode_targets(labels, class_ids)
    assert targets.tolist() == [[IGNORE_INDEX, 0], [1, 2]]
    decoded = decode_predictions(np.array([[0, 1], [2, 0]]), class_ids)
    assert decoded.dtype == np.uint16
    assert decoded.tolist() == [[3, 10], [300, 3]]
    assert decode_predictions(np.array([0]), [1, 255]).dtype == np.uint8


def test_pad_to_min_size():
    a = np.ones((2, 3, 5))
    padded = pad_to_min_size(a, 4, value=-1)
    assert padded.shape == (2, 4, 5)
    assert (padded[:, 3] == -1).all()
    assert pad_to_min_size(a, 2) is a


def brute_force_origins(labels, size, stride, frac):
    h, w = labels.shape
    rows = sorted(set(range(0, h - size + 1, stride)) | {h - size})
    cols = sorted(set(range(0, w - size + 1, stride)) | {w - size})
    out = []
    for r in rows:
        for c in cols:
            n = (labels[r : r + size, c : c + size] != 0).sum()
            if n > 0 and n >= frac * size * size:
                out.append((r, c))
    return out


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.5])
def test_chip_origins_matches_brute_force(frac):
    rng = np.random.default_rng(1)
    labels = np.zeros((70, 90), dtype=np.uint16)
    labels[5:30, 10:50] = 1
    labels[rng.random(labels.shape) < 0.02] = 2
    got = chip_origins(labels, 32, 16, frac)
    assert [tuple(o) for o in got] == brute_force_origins(labels, 32, 16, frac)


def test_chip_origins_requires_padding():
    with pytest.raises(ValueError, match="pad"):
        chip_origins(np.ones((10, 10)), 32, 16, 0.0)


def test_split_train_val():
    rng = np.random.default_rng(0)
    train, val = split_train_val(10, 0.2, rng)
    assert len(train) == 8 and len(val) == 2
    assert sorted(np.concatenate([train, val]).tolist()) == list(range(10))
    train, val = split_train_val(1, 0.5, rng)
    assert train.tolist() == [0] and len(val) == 0
