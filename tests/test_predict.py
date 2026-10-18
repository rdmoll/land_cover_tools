import numpy as np
import pytest
from torch import nn

from src.predict import predict_indices


class BandArgmax(nn.Module):
    """Logits equal to the input bands, so the prediction is a per-pixel argmax."""

    def forward(self, x):
        return x


@pytest.mark.parametrize(("height", "width"), [(100, 130), (20, 20), (128, 128)])
def test_predict_indices_covers_every_pixel(height, width):
    image = np.random.default_rng(0).random((3, height, width)).astype(np.float32)
    got = predict_indices(BandArgmax(), image, tile_size=64, overlap=8, batch_size=3)
    assert got.shape == (height, width)
    np.testing.assert_array_equal(got, image.argmax(axis=0))


def test_predict_indices_without_overlap():
    image = np.random.default_rng(1).random((2, 70, 50)).astype(np.float32)
    got = predict_indices(BandArgmax(), image, tile_size=32, overlap=0)
    np.testing.assert_array_equal(got, image.argmax(axis=0))
