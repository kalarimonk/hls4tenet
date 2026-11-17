# MPS example dataset (1000 test samples)

A 4-site Matrix Product State (MPS) classifier with 1000 test samples. The MPS
C-simulation testbench reads exactly this many (`NUM_SAMPLES 1000`).

```
mps/
├── dataset/               MPS model (4 features, bond dim up to 9, 5 classes), inputs, outputs, labels
├── hw_results/            hardware and C-sim inference outputs, checked against the software results
└── mps_sw_hw_check.ipynb  SW/HW comparison, accuracy and ROC-AUC
```

The dataset folder contains:

| File | Contents |
|---|---|
| `MPS_poly_N4.npz` | Trained model. One tensor per site, with keys `site0` … `site3` in chain order. This can be exported from the tn4ml `.pkl` version as explained below |
| `MPS_poly_N4.pkl` | The same model as a pickled [tn4ml](https://github.com/bsc-quantic/tn4ml) `TensorNetwork`. Also accepted by the tool, but loading it needs `tn4ml`. |
| `X_N4_embedded.bin` | Embedded inputs as raw float64, shape (1000, 4, 3), row-major, no header. This is the C-sim testbench input. |
| `X_N4_embedded.txt` | Same values as the `.bin`, one per line. |
| `outputs_N4_norm_false.txt` | Software (floating-point) model outputs, 1000 × 5 values, one per line, sample by sample. Use this as the C-sim reference. |
| `outputs_N4_norm_false.npy` | Same outputs as an array of shape (1000, 5). |
| `y_N4.npy` | Ground-truth labels, one-hot float64, shape (1000, 5). |
| `contraction.py` | Script that computed the software outputs by contracting the model in numpy (`normalize=False`), and suggests an integer bit count for the fixed-point type. Needs `tn4ml`. |

## Differences from the TTN example

- **Model format:** both use `.npz`, with different keys. MPS has one tensor per site
  (`site<i>`), with shape `(left_bond, right_bond, physical)`. The label site (site 1) has an
  extra class axis: `(3, 9, 3, 5)`. TTN has one tensor per tree node (`"<layer>.<index>"`).
  The MPS `.npz` and `.pkl` hold identical values and produce identical HLS sources.
- **Input embedding:** 3-dimensional polynomial feature map per feature (4 × 3 values per
  sample). Both TTN sets use 2 dimensions (16 × 2).
- **Outputs:** MPS outputs are raw class scores, and the predicted class is the **largest
  output**. TTN outputs are amplitudes, and the predicted class is the largest *squared* output.
- **Fixed-point range:** some software outputs are large (up to about 53), so the MPS needs more
  integer bits than the TTN. `ap_fixed<18,8>` (range ±128) represents them without saturation.

## Usage

```bash
# MPS, 18-bit fixed point with 8 integer bits, run C-simulation
tenet examples/mps/dataset/MPS_poly_N4.npz mps -w 18 -i 8 -c \
      -tb examples/mps/dataset/X_N4_embedded.bin
```

Compare the C-sim results file (`generated_hls/results.txt`) with `outputs_N4_norm_false.txt`.
Both are 5 values per sample, in the same order.

To use your own tn4ml MPS, convert it to `.npz` once on a machine where tn4ml runs:

```python
from hls_tenet.processor import export_mps_npz
export_mps_npz("MY_MODEL.pkl", "MY_MODEL.npz")
```

## Checking hardware results

`mps_sw_hw_check.ipynb` compares the outputs in `hw_results/` with the software results,
then reports accuracy and one-vs-rest ROC-AUC. Run it from this folder. To check a new run,
set `HW_RESULTS` to your results file. MPS outputs are raw scores, so the predicted class is
the largest output, and class probabilities for the ROC curves are their softmax.

| File | Source | Fixed-point format | Samples |
|---|---|---|---|
| `mps4_188_hls_csim_results.txt` | Vitis C-sim | `ap_fixed<18,8>` | 1000 |
