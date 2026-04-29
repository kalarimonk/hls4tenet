from pathlib import Path

from hls_tenet.cli import tenet_args


def test_tenet_args_defaults():
    args = tenet_args(["/tmp/data"])
    assert args.directory_path == Path("/tmp/data")
    assert args.csim is False
    assert args.build is False
    assert args.pack is False
    assert args.top_iso_ttn is False


def test_tenet_args_flags():
    args = tenet_args(["/tmp/data", "-c", "-b", "--top-iso-ttn"])
    assert args.csim is True
    assert args.build is True
    assert args.pack is False
    assert args.top_iso_ttn is True
