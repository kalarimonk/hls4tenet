# hls4tenet

[![CI](https://github.com/kalarimonk/hls4tenet/actions/workflows/ci.yml/badge.svg)](https://github.com/kalarimonk/hls4tenet/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)

**TENET** generates High-Level Synthesis (HLS) C++ for tensor-network machine-learning models and
drives AMD Vitis HLS (C-simulation, synthesis, IP packaging) from a single command, without opening
the Vitis GUI. It supports **Matrix Product States (MPS)** and **Tree Tensor Networks (TTN)**, and
is used to deploy jet-substructure classifiers for High Energy Physics on FPGAs.

```bash
pip install "hls4tenet @ git+https://github.com/kalarimonk/hls4tenet.git@dev-mps"
tenet model.npz mps -w 16 -i 6
```

## Features

- Reads trained tensor-network models: MPS from [tn4ml](https://github.com/bsc-quantic/tn4ml)
  (`.pkl`) or numpy (`.npz`), TTN from numpy (`.npz`).
- Generates model-specific HLS sources (weights, dimensions, contraction kernels, testbench) in
  configurable `ap_fixed` precision.
- Optionally runs Vitis HLS C-simulation, synthesis and RTL IP packaging.

## Requirements

- Python 3.10 – 3.14
- AMD Vitis / Vivado: only needed to run C-simulation, synthesis or packaging. Generating HLS
  sources works without it.
- Optional: [tn4ml](https://github.com/bsc-quantic/tn4ml), only for reading MPS models saved as
  tn4ml `.pkl` files. It needs JAX and therefore a CPU with AVX support. `.npz` models need only
  numpy (see [Model formats](#model-formats)).

## Installation

> **Pre-release:** hls4tenet is not yet on PyPI and has no tagged release yet. Install the current
> development branch from GitHub.

```bash
python -m venv .venv
source .venv/bin/activate
pip install "hls4tenet @ git+https://github.com/kalarimonk/hls4tenet.git@dev-mps"
tenet --help
```

To also read MPS models saved as tn4ml `.pkl` files, install the `tn4ml` extra:

```bash
pip install "hls4tenet[tn4ml] @ git+https://github.com/kalarimonk/hls4tenet.git@dev-mps"
```

Once released on PyPI:

```bash
pip install hls4tenet            # TTN and MPS .npz models
pip install "hls4tenet[tn4ml]"   # also MPS tn4ml .pkl models
```

The package is installed and imported as `hls4tenet`, and provides the command-line tool `tenet`
(also available as `python -m hls4tenet`).

## Quick start

Generate HLS sources for the bundled 4-site MPS example:

```bash
git clone https://github.com/kalarimonk/hls4tenet.git
cd hls4tenet
tenet examples/mps/dataset/MPS_poly_N4.npz mps -w 16 -i 6
```

The sources are written to `./generated_hls/`. To also run C-simulation, source Vitis and pass
the testbench input:

```bash
source /tools/Xilinx/Vitis/<version>/settings64.sh
tenet examples/mps/dataset/MPS_poly_N4.npz mps -w 16 -i 6 -c \
      -tb examples/mps/dataset/X_N4_embedded.bin
```

## Usage

```text
tenet [-h] [-w W] [-i I] [-clock CLOCK] [-c] [-tb TB_DATA] [-b] [-p] model_path {mps,ttn}
```

| Argument | Description | Default |
|---|---|---|
| `model_path` | Trained model file (`.pkl` or `.npz`) | required |
| `{mps,ttn}` | Tensor-network structure | required |
| `-w` | Total bit width of the `ap_fixed` type | `16` |
| `-i` | Integer bits of the `ap_fixed` type | `2` |
| `-clock` | Target clock period in ns | `4.0` |
| `-c`, `--csim` | Run Vitis C-simulation (requires `-tb`) | off |
| `-tb`, `--tb_data` | Testbench input, raw `float64` `.bin` | – |
| `-b`, `--build` | Run synthesis | off |
| `-p`, `--pack` | Export an RTL IP package (runs synthesis) | off |

For example, `-w 16 -i 6` produces `ap_fixed<16, 6>`.

### Model formats

| Network | Format | Contents |
|---|---|---|
| MPS | `.pkl` | A tn4ml model saved with `tn4ml.models.model.save_model`. Needs the `tn4ml` extra and an AVX-capable CPU. |
| MPS | `.npz` | One array per site, keys `site0`, `site1`, … in chain order. Each site has shape `(left, right, physical)`, and the label site has an extra class axis. |
| TTN | `.npz` | One array per node, keys `"<layer>.<index>"`, root `"0.0"`. |

To convert an MPS `.pkl` to `.npz` (on a machine where tn4ml runs):

```python
from hls4tenet.processor import export_mps_npz
export_mps_npz("model.pkl", "model.npz")
```

Both formats produce identical HLS sources.

### Outputs

| Path | Contents |
|---|---|
| `generated_hls/` | Generated HLS sources, testbench, `hls_config.cfg` and, after C-simulation, `results.txt`. **Recreated on every run.** |
| `tenet_workspace/tenet_comp/` | Vitis workspace and component: reports, logs and the IP package `run_tenet.zip`. Can be opened in the Vitis GUI. |

Both are created in the current working directory.

### Using the sources in the Vitis GUI

Run `tenet` without `-c`, `-b` or `-p`. Then add the files in `generated_hls/` to an HLS
component, with `tenet_tb.cpp` as the testbench and `run_tenet` as the top function.

## Examples

| Example | Description |
|---|---|
| [`examples/mps`](examples/mps/README.md) | 4-site MPS classifier, testbench data, hardware results and SW/HW comparison notebook |
| [`examples/ttn`](examples/ttn/README.md) | Symmetric and asymmetric TTN classifiers, testbench data, hardware results and notebook |

## Troubleshooting

| Problem | Fix |
|---|---|
| `vitis executable not found in PATH` | Source `settings64.sh` from your Vitis installation in the same shell. |
| Vitis reports the workspace is in use | Delete `tenet_workspace/` and run again. |
| `jaxlib was built using AVX instructions` | The CPU does not expose AVX. Use the `.npz` model format, or run on an AVX-capable machine. |
| `Loading .pkl MPS models requires tn4ml` | tn4ml is not installed or failed to import. Run `pip install "hls4tenet[tn4ml]"`, or use an `.npz` model. |

## Development

```bash
git clone https://github.com/kalarimonk/hls4tenet.git
cd hls4tenet
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest              # unit tests
ruff check src tests
python -m build     # sdist + wheel in dist/
```

The package version comes from git tags (`hatch-vcs`). Pushing a tag `vX.Y.Z` publishes to PyPI
through the release workflow. Release-candidate tags (`vX.Y.ZrcN`) are not published and are
meant for installing from GitHub.

## Citation

If you use hls4tenet in your work, please cite it. The citation metadata is in
[CITATION.cff](CITATION.cff), and GitHub's **Cite this repository** button exports it as
APA or BibTeX. A paper describing TENET is in preparation.

## License

Copyright 2026 Prateek Gupta and contributors. Licensed under the [Apache License, Version 2.0](LICENSE); see
also [NOTICE](NOTICE).
