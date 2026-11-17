from pathlib import Path

import numpy as np
import pytest

from hls4tenet.cli import main, run_generation, tenet_args
from hls4tenet.processor import TTNProcessor, matches_node_pattern, process_directory
from hls4tenet.tool_configuration import ToolConfiguration


def test_tenet_args_defaults():
    args = tenet_args(["/tmp/model.npz", "ttn"])
    assert args.model_path == Path("/tmp/model.npz")
    assert args.tn == "ttn"
    assert args.csim is False
    assert args.build is False
    assert args.pack is False
    assert args.tb_data is None


def test_tenet_args_flags():
    args = tenet_args(["/tmp/model.npz", "mps", "-c", "-tb", "/tmp/input.bin", "-b", "-p"])
    assert args.csim is True
    assert args.build is True
    assert args.pack is True
    assert args.tb_data == Path("/tmp/input.bin")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.0", True),
        ("12.4", True),
        ("root", False),
        ("1", False),
        ("1.2.3", False),
    ],
)
def test_matches_node_pattern(value, expected):
    assert matches_node_pattern(value) is expected


def test_process_directory_writes_bin_and_txt(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    np.save(input_dir / "sample.npy", np.array([[1.0, 2.5]], dtype=np.float32))

    process_directory(input_dir, output_dir)

    assert np.fromfile(output_dir / "sample.bin", dtype=np.float64).tolist() == [1.0, 2.5]
    assert (output_dir / "sample.txt").read_text(encoding="utf-8").splitlines() == ["1.0", "2.5"]


def test_ttn_processor_reads_npz_nodes_and_ignores_other_keys(tmp_path):
    model_path = tmp_path / "model.npz"
    np.savez(
        model_path,
        **{
            "0.0": np.arange(8).reshape(2, 2, 2),
            "1.0": np.ones((2, 3, 1, 4)),
            "metadata": np.array([99]),
        },
    )

    configuration = ToolConfiguration()
    configuration.model_files_path = model_path
    processor = TTNProcessor(configuration)

    assert processor.get_num_node() == 2
    assert processor.get_total_weights() == 8 + 24
    assert processor.get_max_bond_dimension() == 4
    assert processor.get_num_classes() == 2
    assert processor.get_tensor_metadata() == ["{2, 2, 2, 0, 0}", "{2, 3, 4, 8, 1}"]


def test_run_generation_ttn_writes_expected_hls_files(tmp_path):
    model_path = tmp_path / "model.npz"
    np.savez(model_path, **{"0.0": np.arange(8).reshape(2, 2, 2)})

    configuration = ToolConfiguration()
    configuration.tensor_network_structure = "TTN"
    configuration.model_files_path = model_path
    configuration.output_dir = tmp_path / "generated_hls"
    configuration.output_dir.mkdir()
    configuration.tb_data_path = tmp_path / "input.bin"
    configuration.vitis_configfile_path = configuration.output_dir / "hls_config.cfg"

    processor = run_generation(configuration)

    assert processor.get_num_node() == 1
    assert (configuration.output_dir / "hls_tenet.cpp").exists()
    assert (configuration.output_dir / "hls_dispatcher.cpp").exists()
    assert "#define NUM_FEATURES 2" in (configuration.output_dir / "hls_tensor.h").read_text(
        encoding="utf-8"
    )
    assert "const tensor_t w1d[8]" in (configuration.output_dir / "hls_weights.h").read_text(
        encoding="utf-8"
    )


def test_main_rejects_tb_data_without_csim(tmp_path, capsys, monkeypatch):
    # main() writes to ./generated_hls: run in a temp dir so the real one is never touched
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "model.npz"
    tb_path = tmp_path / "input.bin"
    np.savez(model_path, **{"0.0": np.arange(8).reshape(2, 2, 2)})
    tb_path.write_bytes(b"")
    previous_output = tmp_path / "generated_hls" / "keep_me.txt"
    previous_output.parent.mkdir()
    previous_output.write_text("previous run", encoding="utf-8")

    assert main([str(model_path), "ttn", "-tb", str(tb_path)]) == 1

    assert "only relevant if C-Simulation is enabled" in capsys.readouterr().out
    # invalid arguments must not wipe the previous output
    assert previous_output.read_text(encoding="utf-8") == "previous run"


def test_ttn_template_resources_are_packaged():
    import importlib.resources as resources

    template_root = resources.files("hls4tenet").joinpath("templates", "ttn")

    assert template_root.joinpath("tenet.cpp.in").is_file()
    assert template_root.joinpath("dispatcher.cpp.in").is_file()
