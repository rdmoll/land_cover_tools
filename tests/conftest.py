import pytest
from affine import Affine
from rasterio.crs import CRS

from src.raster_io import GeoGrid


@pytest.fixture
def grid() -> GeoGrid:
    """A 64 x 48 px, 1 m UTM grid."""
    return GeoGrid(CRS.from_epsg(32610), Affine(1, 0, 500000, 0, -1, 5000048), 64, 48)
