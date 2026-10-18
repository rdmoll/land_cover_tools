import pytest

from src.main import build_parser


def test_parser_accepts_commands():
    args = build_parser().parse_args(["train", "--config", "x.yaml"])
    assert args.command == "train" and str(args.config) == "x.yaml"


def test_parser_rejects_unknown_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bogus"])
