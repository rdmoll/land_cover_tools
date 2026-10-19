from pathlib import Path

import pytest
import yaml

from src.config import load_config, parse_config


def raw_config() -> dict:
    return yaml.safe_load(Path("configs/default.yaml").read_text())


def test_default_config_loads():
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.class_ids == (1, 2, 3, 4, 5, 6)
    assert cfg.imagery.resolution == 1.0
    assert cfg.imagery.bands == (1, 2, 3, 4)
    assert cfg.training.model_path.is_absolute()


def test_relative_paths_resolve_against_base_dir(tmp_path):
    cfg = parse_config(raw_config(), base_dir=tmp_path)
    assert cfg.imagery.output == tmp_path / "data/imagery.tif"


def test_classes_accept_plain_names():
    raw = raw_config()
    raw["classes"] = {2: "trees", 1: "water"}
    cfg = parse_config(raw)
    assert [(c.class_id, c.name, c.color) for c in cfg.classes] == [
        (1, "water", None),
        (2, "trees", None),
    ]


@pytest.mark.parametrize("class_id", [0, -1, 70000])
def test_class_id_out_of_range_rejected(class_id):
    raw = raw_config()
    raw["classes"] = {class_id: "bad"}
    with pytest.raises(ValueError, match="between 1 and 65535"):
        parse_config(raw)


def test_bad_bbox_rejected():
    raw = raw_config()
    raw["imagery"]["bbox"] = [1, 1, 0, 2]
    with pytest.raises(ValueError, match="bbox"):
        parse_config(raw)


def test_chip_size_must_fit_unet():
    raw = raw_config()
    raw["training"]["chip_size"] = 100
    with pytest.raises(ValueError, match="multiple of 32"):
        parse_config(raw)


def test_overlap_must_leave_a_core():
    raw = raw_config()
    raw["prediction"]["overlap"] = 256
    with pytest.raises(ValueError, match="overlap"):
        parse_config(raw)
