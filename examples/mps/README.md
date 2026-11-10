# MPS example dataset (1000 test samples)

A 4-site Matrix Product State (MPS) classifier with 1000 test samples. The MPS
C-simulation testbench reads exactly this many (`NUM_SAMPLES 1000`).

```
mps/
├── dataset/               model, embedded inputs, software outputs and labels
├── hw_results/            hardware inference outputs, checked against the software results
└── mps_sw_hw_check.ipynb  SW/HW comparison, accuracy and ROC-AUC
```

## Model

| Property | Value |
|---|---|
| Sites / features | 4 |
| Physical (feature map) dimension | 3, polynomial embedding |
| Max bond dimension | 9 |
| Classes | 5, read out from the label index on site 1 |
| Tensor shapes | `(1, 3, 3)`, `(3, 9, 3, 5)`, `(9, 3, 3)`, `(3, 1, 3)` |

## `dataset/`

| File | Contents |
|---|---|
| `MPS_poly_N4.pkl` | Trained model, a pickled [tn4ml](https://github.com/bsc-quantic/tn4ml) `TensorNetwork`. This is the file passed to the tool. Loading it requires `tn4ml`. |
| `X_N4_embedded.bin` | Embedded inputs as raw float64, shape (1000, 4, 3), row-major, no header. This is the C-sim testbench input. |
| `X_N4_embedded.txt` | Same values as the `.bin`, one per line. |
| `software_full_precision_results.txt` | Software (floating-point) model outputs, 1000 × 5 values, one per line, sample by sample. Use this as the C-sim reference. |
| `y_N4.npy` | Ground-truth labels, one-hot float64, shape (1000, 5). |

## `hw_results/`

Hardware (FPGA) inference outputs, one value per line and 5 per sample, in the same order as
the software results. Both runs cover the first 992 samples.

| File | Fixed-point format | Fractional bits |
|---|---|---|
| `mps4_166_hwresults.txt` | `ap_fixed<16,6>` | 10 |
| `mps4_168_hwresults.txt` | `ap_fixed<16,8>` | 8 |

## Differences from the TTN example

- **Model format:** a tn4ml `.pkl`, not a `.npz` of node tensors.
- **Input embedding:** 3 values per feature (4 × 3 per sample), while TTN uses 2 (16 × 2).
- **Outputs:** MPS outputs are raw class scores. The predicted class is the **largest output**.
  TTN outputs are amplitudes, and the predicted class is the largest *squared* output.
- **Saturation:** some software outputs are large (up to about 54), beyond what the
  hardware fixed-point range can represent. Those values saturate in hardware, which
  shows up as outliers in the SW/HW comparison. This barely affects accuracy.

## Usage

Run from the `hls_tenet/` folder:

```bash
# MPS, 16-bit fixed point with 6 integer bits, run C-simulation
tenet examples/mps/dataset/MPS_poly_N4.pkl mps -w 16 -i 6 -c \
      -tb examples/mps/dataset/X_N4_embedded.bin
```

Compare the C-sim results file with `dataset/software_full_precision_results.txt`.
Both are 5 values per sample, in the same order.

## Checking hardware results

`mps_sw_hw_check.ipynb` compares the outputs in `hw_results/` with the software results,
then reports accuracy and one-vs-rest ROC-AUC. Run it from this folder. To check a new run,
set `HW_RESULTS` to your results file. Class probabilities for the ROC curves are the
softmax of the raw outputs.
