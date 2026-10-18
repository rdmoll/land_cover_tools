import numpy as np
import torch

from src.model import (
    ModelSpec,
    band_stats,
    build_unet,
    load_checkpoint,
    normalize,
    save_checkpoint,
)


def test_band_stats_and_normalize_ignore_invalid_pixels():
    image = np.zeros((2, 4, 4), dtype=np.uint8)
    image[0, :2] = 10
    image[0, 2:] = 20
    image[1] = 5
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    image[:, 0, 0] = 0
    mean, std = band_stats(image, valid)
    assert mean[1] == 5 and std[1] > 0
    spec = ModelSpec("resnet18", 2, (1, 2), mean, std)
    norm = normalize(image, spec)
    assert norm.dtype == np.float32
    assert abs(norm[0][valid].mean()) < 1e-5


def test_checkpoint_roundtrip(tmp_path):
    spec = ModelSpec("resnet18", 4, (1, 5, 9), (1.0,) * 4, (2.0,) * 4)
    model = build_unet(spec.encoder, spec.in_channels, len(spec.class_ids), None).eval()
    x = torch.randn(1, 4, 64, 64)
    expected = model(x)
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, spec)
    loaded, loaded_spec = load_checkpoint(path)
    assert loaded_spec == spec
    torch.testing.assert_close(loaded(x), expected)
