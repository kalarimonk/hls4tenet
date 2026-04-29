# High Level Synthesis (HLS) based Hardware Implementation of Tree Tensor Networks (TTN)

This project provides a CLI-first workflow to process TTN datasets/models, generate HLS source files, and optionally run Vitis HLS tasks (C simulation, synthesis, packaging) without opening the GUI.

## Prerequisites

- Python 3.9+
- Vitis/Vivado (only required when running Vitis tasks)

## Install

### End users (recommended)

```bash
pipx install hls-tenet
```

### Developers

```bash
python -m pip install -e .[dev]
```



## CLI usage

Canonical entrypoints:

```bash
tenet --help
# or
python -m hls_tenet --help
```

```
### Example

tenet ./datasets/mps_data/MPS_poly_N4.pkl ./datasets/mps_data/ mps -c --tb_data ./testbench_data/X_N4_embedded.bin

```
```
Options:

tenet -h
usage: tenet [-h] [-c] [-tb TB_DATA] [-b] [-p] model_path directory_path {mps,ttn}

Process dataset/model files and generate HLS artifacts for TENET

positional arguments:
  model_path            Path to model file (e.g. .pkl, .npz)
  directory_path        Path to data directory
  {mps,ttn}             Tensor Network Structure

optional arguments:
  -h, --help            show this help message and exit
  -c, --csim            Run C-Simulation
  -tb TB_DATA, --tb_data TB_DATA
                        Path to testbench data file for C-sim (e.g. test.bin)
  -b, --build           Run synthesis build
  -p, --pack            Create RTL IP package
```


## Running Vitis tasks requires sourced environment

```bash
source /tools/Xilinx/Vitis/<version>/settings64.sh
```

Then run TENET commands as above.

## CI/CD

- `CI` workflow validates lint, tests, CLI smoke test, and build artifacts on pushes/PRs.
- `Release` workflow publishes to PyPI on git tags matching `v*`.

## Notes

- If Vitis reports a locked workspace, use a different `--workspace` path or remove the stale lock/workspace directory.
- `hls_config.cfg` should include valid absolute paths for `csim.argv` inputs/outputs.
