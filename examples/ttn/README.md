# TTN example datasets (800 test samples)

Two Tree Tensor Network (TTN) models, each with the first 800 test samples. The
TTN C-simulation testbench reads exactly this many (`NUM_SAMPLES 800`).

```
ttn/
├── dataset_symm/          symmetric TTN   (16 features, bond dim up to 10, 5 classes)
├── dataset_asymm/         asymmetric TTN  (16 features, bond dim up to 40, 5 classes)
├── hw_results/            hardware inference outputs, checked against the software results
└── ttn_sw_hw_check.ipynb  SW/HW comparison, accuracy and ROC-AUC for both models
```

Each dataset folder contains:

| File | Contents |
|---|---|
| `TTN_*.npz` | Trained model. One tensor per node, with keys `"<layer>.<index>"` (root is `"0.0"`). This is the file passed to the tool. |
| `X_test_embedded.bin` | Embedded inputs as raw float64, shape (800, 16, 2), row-major, no header. This is the C-sim testbench input. |
| `X_test_embedded.txt` | Same values as the `.bin`, one per line. |
| `software_inference_results.txt` | Software (floating-point) model outputs, 800 × 5 values, one per line, sample by sample. Use this as the C-sim reference. |
| `software_inference_results.npy` | Same outputs as an array of shape (800, 5). |
| `y_test.npy` | Ground-truth labels, one-hot float64, shape (800, 5). |

## Differences from the MPS example

The two TTN datasets came from different training pipelines, so a few details differ:

- **Labels:** `y_test.npy` is one-hot float64 with shape `(800, 5)` in both folders, the same encoding as the MPS example. The asymmetric labels were originally class indices and were converted to one-hot.
- **Software outputs:** the symmetric model's outputs were saved as float32 (fewer digits), the asymmetric model's as float64.
- **Model format:** TTN models are `.npz` files, not `.pkl`. The asymmetric root tensor `"0.0"` has shape `(40, 14, 1, 5)`, and the tool squeezes out the singleton axis.
- **Input embedding:** both TTN sets use a 2-dimensional feature map per feature (16 × 2 values per sample). The MPS example uses 3 dimensions (4 × 3). The embedding was applied upstream, so these inputs are ready to use.

## Usage

```bash
# symmetric TTN, 16-bit fixed point with 2 integer bits, run C-simulation
tenet examples/ttn/dataset_symm/TTN_symm.npz ttn -w 16 -i 2 -c \
      -tb examples/ttn/dataset_symm/X_test_embedded.bin

# asymmetric TTN
tenet examples/ttn/dataset_asymm/TTN_asymm.npz ttn -w 16 -i 4 -c \
      -tb examples/ttn/dataset_asymm/X_test_embedded.bin
```

Compare the C-sim results file with `software_inference_results.txt`.
Both are 5 values per sample, in the same order.

## Checking hardware results

`ttn_sw_hw_check.ipynb` compares the outputs in `hw_results/` with the software results,
then reports accuracy and one-vs-rest ROC-AUC. Run it from this folder. To check a new run,
set `HW_RESULTS` in that model's section to your results file. TTN outputs are amplitudes,
so the predicted class is the largest **squared** output.
