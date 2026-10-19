"""Load and validate the YAML run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LandCoverClass:
    """One user-defined land-cover category.

    Attributes
    ----------
    class_id : int
        Unsigned integer written to the output raster. Must be >= 1; 0 is
        reserved for "unlabeled" in training labels and "no data" in outputs.
    name : str
        Human-readable name.
    color : str or None
        Optional ``#RRGGBB`` color used for the GeoTIFF color table.
    """

    class_id: int
    name: str
    color: str | None = None


@dataclass(frozen=True, kw_only=True)
class ImageryConfig:
    """Settings for downloading imagery from Planetary Computer.

    The area is given either as ``bbox`` (lon/lat) or as ``center`` (lon/lat)
    plus ``size_m``, a square edge length in meters on the UTM grid.
    """

    output: Path
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    size_m: float | None = None
    collection: str = "naip"
    datetime: str | None = None
    resolution: float = 1.0
    bands: tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class LabelsConfig:
    """Where the user-drawn label polygons live."""

    path: Path
    class_field: str = "class_id"


@dataclass(frozen=True)
class TrainingConfig:
    """U-Net training hyperparameters."""

    model_path: Path
    chip_size: int = 256
    chip_stride: int = 128
    min_labeled_fraction: float = 0.05
    val_fraction: float = 0.2
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    samples_per_epoch: int | None = None
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class PredictionConfig:
    """Tiled inference and output settings."""

    output: Path
    tile_size: int = 512
    overlap: int = 64
    batch_size: int = 4
    device: str = "auto"


@dataclass(frozen=True)
class Config:
    """Full run configuration."""

    imagery: ImageryConfig
    classes: tuple[LandCoverClass, ...]
    labels: LabelsConfig
    training: TrainingConfig
    prediction: PredictionConfig
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def class_ids(self) -> tuple[int, ...]:
        """Class IDs in ascending order."""
        return tuple(sorted(c.class_id for c in self.classes))


def _parse_classes(raw: dict[Any, Any]) -> tuple[LandCoverClass, ...]:
    if not raw:
        raise ValueError("config must define at least one class under 'classes'")
    classes = []
    for key, value in raw.items():
        class_id = int(key)
        if class_id < 1 or class_id > 65535:
            raise ValueError(f"class id {class_id} must be between 1 and 65535")
        if isinstance(value, str):
            classes.append(LandCoverClass(class_id, value))
        else:
            classes.append(LandCoverClass(class_id, value["name"], value.get("color")))
    return tuple(sorted(classes, key=lambda c: c.class_id))


def parse_config(raw: dict[str, Any], base_dir: Path | None = None) -> Config:
    """Build a :class:`Config` from a parsed YAML mapping.

    Parameters
    ----------
    raw : dict
        Mapping as loaded from YAML.
    base_dir : Path, optional
        Directory that relative paths are resolved against. Defaults to the
        current working directory.

    Returns
    -------
    Config
        Validated configuration.
    """
    base = base_dir or Path.cwd()

    def path(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else base / p

    img = dict(raw["imagery"])
    has_bbox = img.get("bbox") is not None
    has_center = img.get("center") is not None or img.get("size_m") is not None
    if has_bbox == has_center:
        raise ValueError("imagery needs either bbox, or center and size_m (not both)")
    if has_bbox:
        bbox = tuple(float(v) for v in img["bbox"])
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(
                "imagery.bbox must be [min_lon, min_lat, max_lon, max_lat]"
            )
        img["bbox"] = bbox
    else:
        center = tuple(float(v) for v in img.get("center") or ())
        if len(center) != 2 or not (
            -180 <= center[0] <= 180 and -90 <= center[1] <= 90
        ):
            raise ValueError("imagery.center must be [lon, lat] in degrees")
        if img.get("size_m") is None or float(img["size_m"]) <= 0:
            raise ValueError("imagery.size_m must be a positive number of meters")
        img["center"] = center
        img["size_m"] = float(img["size_m"])
    img["output"] = path(img["output"])
    if "bands" in img:
        img["bands"] = tuple(int(b) for b in img["bands"])
    imagery = ImageryConfig(**img)

    lbl = dict(raw["labels"])
    lbl["path"] = path(lbl["path"])
    labels = LabelsConfig(**lbl)

    trn = dict(raw["training"])
    trn["model_path"] = path(trn["model_path"])
    training = TrainingConfig(**trn)
    if training.chip_size % 32 != 0:
        raise ValueError("training.chip_size must be a multiple of 32")

    prd = dict(raw["prediction"])
    prd["output"] = path(prd["output"])
    prediction = PredictionConfig(**prd)
    if prediction.tile_size % 32 != 0:
        raise ValueError("prediction.tile_size must be a multiple of 32")
    if not 0 <= 2 * prediction.overlap < prediction.tile_size:
        raise ValueError("prediction.overlap must be less than half of tile_size")

    known = {"imagery", "classes", "labels", "training", "prediction"}
    return Config(
        imagery=imagery,
        classes=_parse_classes(raw["classes"]),
        labels=labels,
        training=training,
        prediction=prediction,
        extra={k: v for k, v in raw.items() if k not in known},
    )


def load_config(path: Path) -> Config:
    """Read a YAML config file.

    Relative paths inside the file are resolved against the current working
    directory, matching how the CLI is run from the project root.
    """
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return parse_config(raw)
