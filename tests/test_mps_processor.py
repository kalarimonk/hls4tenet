import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hls_tenet import mps_generators
from hls_tenet.processor import MPSProcessor, export_mps_npz
from hls_tenet.tool_configuration import ToolConfiguration

REPO_ROOT = Path(__file__).resolve().parents[1]
MPS_N4_PKL = REPO_ROOT / "examples" / "mps" / "dataset" / "MPS_poly_N4.pkl"
MPS_TEMPLATES = REPO_ROOT / "src" / "hls_tenet" / "templates" / "mps"

STATE_ATTRS = (
    "num_nodes",
    "num_features",
    "max_bond_dim",
    "label_site_id",
    "phy_bond_dim",
    "classes",
    "tensor_metadata",
    "tensor_values",
)


def _processor(model_path: Path) -> MPSProcessor:
    configuration = ToolConfiguration()
    configuration.model_files_path = model_path
    return MPSProcessor(configuration)


def _state(processor: MPSProcessor) -> dict:
    return {attr: getattr(processor, attr) for attr in STATE_ATTRS}


def _render(processor: MPSProcessor, out_dir: Path) -> dict:
    out_dir.mkdir()
    mps_generators.generate_mpsweight_file(
        MPS_TEMPLATES / "weights.h.in", out_dir / "hls_weights.h", processor
    )
    mps_generators.generate_mpsmacro_header(
        MPS_TEMPLATES / "tensor.h.in", out_dir / "hls_tensor.h", processor
    )
    mps_generators.generate_top_module(
        MPS_TEMPLATES / "tenet.cpp.in", out_dir / "hls_tenet.cpp", processor
    )
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(out_dir.iterdir())}


def _synthetic_mps(path: Path, label_site: int = 1) -> list:
    rng = np.random.default_rng(0)
    shapes = [(1, 3, 3), (3, 9, 3), (9, 3, 3), (3, 1, 3)]
    tensors = []
    for i, shape in enumerate(shapes):
        if i == label_site:
            shape = shape + (5,)
        tensors.append(rng.standard_normal(shape))
    np.savez(path, **{f"site{i}": t for i, t in enumerate(tensors)})
    return tensors


def test_mps_npz_matches_pkl(tmp_path):
    pytest.importorskip("tn4ml")
    npz_path = tmp_path / "MPS_poly_N4.npz"
    export_mps_npz(MPS_N4_PKL, npz_path)

    from_pkl = _processor(MPS_N4_PKL)
    from_npz = _processor(npz_path)

    assert _state(from_npz) == _state(from_pkl)
    assert _render(from_npz, tmp_path / "npz") == _render(from_pkl, tmp_path / "pkl")


def test_mps_npz_synthetic_model(tmp_path):
    npz_path = tmp_path / "model.npz"
    tensors = _synthetic_mps(npz_path, label_site=1)

    processor = _processor(npz_path)

    assert processor.num_nodes == processor.num_features == 4
    assert processor.max_bond_dim == 9
    assert processor.label_site_id == 1
    assert processor.classes == 5
    assert processor.phy_bond_dim == 3
    assert processor.tensor_metadata == [tuple(t.shape) for t in tensors]
    for values, tensor in zip(processor.tensor_values, tensors):
        np.testing.assert_array_equal(np.array(values), tensor)


def test_mps_npz_site_order_is_numeric(tmp_path):
    # site10 must come after site9, not after site1 (lexicographic order would break this)
    npz_path = tmp_path / "long.npz"
    tensors = {f"site{i}": np.full((2, 2, 3), float(i)) for i in range(12)}
    tensors["site5"] = np.full((2, 2, 3, 4), 5.0)
    np.savez(npz_path, **tensors)

    processor = _processor(npz_path)

    assert [np.array(v).flat[0] for v in processor.tensor_values] == [float(i) for i in range(12)]
    assert processor.label_site_id == 5


def test_mps_npz_does_not_import_tn4ml(tmp_path):
    npz_path = tmp_path / "model.npz"
    _synthetic_mps(npz_path)
    code = (
        "import sys\n"
        "from hls_tenet.processor import MPSProcessor\n"
        "from hls_tenet.tool_configuration import ToolConfiguration\n"
        "c = ToolConfiguration()\n"
        f"c.model_files_path = {str(npz_path)!r}\n"
        "MPSProcessor(c)\n"
        "heavy = sorted({m.split('.')[0] for m in sys.modules} & {'tn4ml', 'jax', 'jaxlib', 'quimb'})\n"
        "assert not heavy, heavy\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_mps_npz_without_sites_raises(tmp_path):
    npz_path = tmp_path / "bad.npz"
    np.savez(npz_path, weights=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="no 'site<i>' arrays"):
        _processor(npz_path)


def test_mps_npz_bad_rank_raises(tmp_path):
    npz_path = tmp_path / "bad_rank.npz"
    np.savez(npz_path, site0=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unexpected tensor rank 2"):
        _processor(npz_path)
