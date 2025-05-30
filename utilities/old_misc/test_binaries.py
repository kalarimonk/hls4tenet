import numpy as np

data = np.fromfile("/home/pgupta/tenet_hls/datasets/set1/axi_weights.bin")

print("Weights Data:")
print(len(data))
