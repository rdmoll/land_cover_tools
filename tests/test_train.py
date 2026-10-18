import numpy as np
import torch
from torch.utils.data import DataLoader

from src.labels import IGNORE_INDEX
from src.model import build_unet, resolve_device
from src.train import ChipDataset, augment, fit


def test_augment_keeps_image_and_target_aligned():
    rng = np.random.default_rng(0)
    target = np.arange(16).reshape(4, 4)
    image = np.stack([target, target * 2]).astype(np.float32)
    for _ in range(10):
        img, tgt = augment(image, target, rng)
        np.testing.assert_array_equal(img[0], tgt)
        np.testing.assert_array_equal(img[1], tgt * 2)


def test_chip_dataset_stays_in_bounds():
    image = np.zeros((3, 40, 50), dtype=np.float32)
    targets = np.zeros((40, 50), dtype=np.int64)
    origins = np.array([[0, 0], [8, 18]])
    ds = ChipDataset(image, targets, origins, chip_size=32, jitter=16, seed=0)
    for _ in range(20):
        for i in range(len(ds)):
            img, tgt = ds[i]
            assert img.shape == (3, 32, 32) and tgt.shape == (32, 32)


def test_fit_learns_a_trivial_task():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(3, 64, 64)).astype(np.float32)
    targets = (image[0] > 0).astype(np.int64)
    targets[:8] = IGNORE_INDEX
    ds = ChipDataset(image, targets, np.array([[0, 0], [32, 32], [0, 32]]), 32)
    loader = DataLoader(ds, batch_size=3)
    torch.manual_seed(0)
    model = build_unet("resnet18", 3, 2, None)
    saved = []
    best = fit(model, loader, loader, 30, 1e-3, resolve_device("cpu"), saved.append)
    assert saved, "on_improved was never called"
    assert best < 0.6
