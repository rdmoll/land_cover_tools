"""Command-line entry point.

Typical workflow::

    python -m src.main download --config configs/default.yaml
    # draw label polygons over data/imagery.tif in QGIS, save as data/labels.gpkg
    python -m src.main train --config configs/default.yaml
    python -m src.main predict --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from src.config import load_config

logger = logging.getLogger("src")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Build a 1 m land-cover map from NAIP imagery with a U-Net.",
    )
    parser.add_argument(
        "command",
        choices=["download", "train", "predict", "train-predict"],
        help="download: fetch imagery; train: fit the U-Net on your labels; "
        "predict: write the land-cover GeoTIFF; train-predict: both",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="YAML config"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    # Third-party libraries log at WARNING; this package logs at INFO/DEBUG.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    cfg = load_config(args.config)

    # Imported lazily so `download` doesn't pay for loading torch.
    if args.command == "download":
        from src.imagery import download_imagery

        download_imagery(cfg.imagery)
        logger.info(
            "Next: draw polygons over %s with an integer '%s' field, save them to "
            "%s, then run `train`",
            cfg.imagery.output,
            cfg.labels.class_field,
            cfg.labels.path,
        )
    if args.command in ("train", "train-predict"):
        from src.train import train_model

        train_model(cfg)
    if args.command in ("predict", "train-predict"):
        from src.predict import predict_landcover

        predict_landcover(cfg)


if __name__ == "__main__":
    main()
