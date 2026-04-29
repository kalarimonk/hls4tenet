import re
import struct
from pathlib import Path
import os
import tn4ml 
from tn4ml.models.model import *

import numpy as np

NODE_KEY_PATTERN = re.compile(r"^\d+\.\d+$")


def matches_node_pattern(value: str) -> bool:
    return bool(NODE_KEY_PATTERN.match(value))

def save_npy_as_bin(npy_file: Path, output_file: Path) -> None:
    print(f"Processing .npy file: {npy_file}")
    array_data = np.load(npy_file).flatten()
    array_data.astype(np.float64).tofile(output_file)
    print(f"Saved .npy as binary: {output_file}")

def save_npy_as_dat(npy_file: Path, results_dat: Path) -> None:
    data = np.load(npy_file).flatten()
    np.savetxt(results_dat, data, fmt="%s")
    
def process_directory(input_dir: Path, output_dir: Path) -> None:
        input_dir = Path(input_dir)

        for npy_file in sorted(input_dir.glob("*.npy")):
            output_file = output_dir / f"{npy_file.stem}.bin"
            save_npy_as_bin(npy_file, output_file)

            txt_file = output_dir / f"{npy_file.stem}.txt"
            save_npy_as_dat(npy_file, txt_file)

class TTNProcessor:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.is_top_iso = False
        self.input_map_dim = 2
        self.reset_state()
        self.parse_software_model(self.model_path)

    def reset_state(self) -> None:
        self.num_nodes = 0
        self.num_features = 0
        self.max_bond_dim = 0
        self.num_weights = 0
        self.classes = 0
        self.height = 0
        self.tensor_values = []
        self.tensor_metadata = []
        self.dispatch_config = []
    
    def parse_software_model(self, model_path: Path) -> None:
        
        self.reset_state()
        
        print(f"Parsing software TTN model: {model_path}")

        # output_dir.mkdir(parents=True, exist_ok=True)  
        # output_file = output_dir.resolve().joinpath("weights.bin")
        
        weight_tensors = np.load(model_path)

        config_ids = {}
        # with output_file.open("wb") as fp:
            # fp.write(struct.pack("i", len(weight_tensors)))
        counter = 0

        for key in sorted(weight_tensors.keys()):
            if not matches_node_pattern(key):
                continue

            metadata = []
            weights = weight_tensors[key]
            if len(weights.shape) == 4:
                print("Before squeezing:", key, weights.shape)
                weights = weights.squeeze(axis=2)
                print("After squeezing:", key, weights.shape)

            a, b, c = weights.shape
            print("Processing weight tensor:", key, (a, b, c))

            config_key = f"{a}x{b}x{c}"
            metadata.extend([a, b, c, self.num_weights])
            

            if config_key not in config_ids:
                config_ids[config_key] = counter
                metadata.append(counter)
                counter += 1
                self.dispatch_config.append(metadata)
            else:
                metadata.append(config_ids[config_key])
                
            print("Metadata for tensor", key, metadata)

            self.tensor_metadata.append("{" + ", ".join(str(x) for x in metadata) + "}")

            self.num_nodes += 1
            self.max_bond_dim = max(a, b, c, self.max_bond_dim)
            self.num_weights += a * b * c

            if key == "0.0":
                self.classes = min(a, b, c)

            # fp.write(struct.pack("iii", a, b, c))
            flat_weights = weights.flatten()

            if key == "0.0" and self.is_top_iso:
                norm = np.linalg.norm(flat_weights)
                if norm != 0:
                    flat_weights = flat_weights / norm

            # fp.write(flat_weights.astype(np.float64))
            self.tensor_values.extend(flat_weights.tolist())

        self.height = int(np.log2(self.num_nodes + 1))
        self.num_features = 2 ** self.height

    def get_max_bond_dimension(self):
        return self.max_bond_dim

    def get_num_node(self):
        return self.num_nodes

    def get_total_weights(self):
        return self.num_weights

    def get_num_classes(self):
        return self.classes

    def get_tensor_values(self):
        return self.tensor_values

    def get_tensor_metadata(self):
        return self.tensor_metadata

    def get_tree_height(self):
        return self.height

    def get_num_features(self):
        return self.num_features

    def get_input_map_dim(self):
        return self.input_map_dim

    def get_dispatch_config(self):
        return self.dispatch_config



class MPSProcessor:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.reset_state()
        self.parse_software_model(self.model_path)

    def reset_state(self) -> None:
        self.num_nodes = 0
        self.num_features = 0
        self.max_bond_dim = 0
        self.label_site_id = None
        self.phy_bond_dim = None
        self.classes = 0    #label site dim
        self.bits_to_pack = 0
        self.tensor_values = []
        self.tensor_metadata = []
        
    def parse_software_model(self, model_path: Path) -> None:
        self.reset_state()
        
        print(f"Parsing software MPS model: {model_path.parent / model_path.stem}")

        model = load_model(f"{model_path.parent / model_path.stem}")
        
        self.num_nodes = model.L
        self.num_features = model.L
        self.max_bond_dim = model.max_bond()
        
        tensor_id = 0
        for tensor in model.tensors:
            # print("\nProcessing tensor with shape:", len(tensor.shape))

            if len(tensor.shape) == 3:
                a, b, c = tensor.shape
                self.tensor_metadata.append((a, b, c))
                self.tensor_values.append(tensor.data.tolist())
            elif len(tensor.shape) == 4:
                a, b, c, d = tensor.shape
                self.label_site_id = tensor_id
                self.classes = d
                self.tensor_metadata.append((a, b, c, d))
                self.tensor_values.append(tensor.data.tolist())           
            tensor_id += 1
        self.phy_bond_dim = c # The physical bond dimension is the last one in the shape of the tensors [bond_left, bond_right, phy_bond_dim]
    
    def calculate_bits_to_pack(self, bit_width: int) -> int:
        self.bits_to_pack = self.phy_bond_dim * self.num_features * bit_width
        return self.bits_to_pack
        