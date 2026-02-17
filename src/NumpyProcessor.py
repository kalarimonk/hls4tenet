import numpy as np
import struct
import re
from pathlib import Path

pattern = r'^(\d+\.\d+)*$'


def matches_node_pattern(s):
    return bool(re.match(pattern, s))


class NumpyFileProcessor:
    def __init__(self, base_dir, is_top_iso):
        self.base_dir = Path(base_dir)
        self.is_top_iso = is_top_iso
        self.num_nodes = 0
        self.num_features = 0
        self.max_bond_dim = 0
        self.num_weights = 0
        self.classes = 0
        self.input_map_dim = 2
        self.height = 0
        self.tensor_values = []
        self.tensor_metadata = [] # stored as list of string "{a,b,c, pos, config_id}"
        self.dispatch_config = [] # same as metadata but stored as a list of list

    def save_npy_as_bin(self, npy_file, output_file):
        """Convert a .npy file to a text file."""
        print(f"Processing .npy file: {npy_file}")
        array_data = (np.load(npy_file)).flatten()
        # print(len(array_data))
        array_data.astype(np.float64).tofile(output_file)
        # np.savetxt(output_file, array_data.flatten(), fmt='%s', delimiter=',')
        print(f"Saved .npy as text file: {output_file}")

    def save_npy_as_text(self, npy_file, output_file):
    	# Not used
        """Convert a .npy file to a text file."""
        print(f"Processing .npy file: {npy_file}")
        array_data = (np.load(npy_file)).flatten()
        print(len(array_data))
        array_data.astype(np.float64).tofile(output_file)
        # np.savetxt(output_file, array_data.flatten(), fmt='%s', delimiter=',')
        print(f"Saved .npy as text file: {output_file}")
        
    def save_npy_as_dat(self,npy_file, results_dat):
        data = np.load(npy_file)
         # Ensure the data is a 1D array
        data = data.flatten()
        np.savetxt(results_dat, data, fmt='%s')


    def save_npz_as_text(self, npz_file, output_dir):
        """Convert each array in a .npz file to separate text files."""
        print(f"Processing .npz file: {npz_file}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weight_tensors = np.load(npz_file)
        for key in weight_tensors:
            if matches_node_pattern(key):
                output_file = output_dir / f"{key}.txt"
                array_data = weight_tensors[key]
                print("Tensor Shape:", array_data.shape)
                np.savetxt(output_file, array_data.flatten(), fmt='%s', delimiter=',')
                print(f"Saved {key} from .npz as text file: {output_file}")
    
    def save_npz_as_bin(self, npz_file, output_dir):
        print(f"Processing .npz file: {npz_file}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'weights.bin'
        weight_tensors = np.load(npz_file)
        
        self.num_nodes = 0
        
        config_ids = {}
        
        with open( output_file , "wb") as f:
            f.write(struct.pack("i",len(weight_tensors)))
            # f.write(f'{len(weight_tensors)}')
            config_ids = {}
            counter = 0
            for key in list(weight_tensors.keys()):
                temp = []
                if matches_node_pattern(key):
                    
                    print(f"Processing weight tensor: {key}")  # Debugging message
                    # print(weight_tensors[key])  
                    weights = weight_tensors[key] 
                    
                    if(len(weights.shape) == 4 ):
                    	weights = weights.squeeze(axis=2)  # This is due to top tensor with a shape 40,14,1,5 in the asymmetrical dataset
                    	print("After Squeezing:",key, weights.shape)
                    a,b,c = weights.shape
                    print(a,b,c)
                   
                    
                    onew = np.ones(a*b*c)
                    
                    config_key = str(a)+str(b)+str(c)

                    # Tensor Metadata
                    temp.append(a)
                    temp.append(b)
                    temp.append(c)
                    temp.append(self.num_weights)
                    
                    if config_key not in config_ids.keys():
                    	config_ids[config_key] = counter
                    	temp.append(counter)
                    	counter+=1
                    	self.dispatch_config.append(temp)
                    else:
                    	temp.append(config_ids[config_key])
                    
                    
                    tensor_meta = "{"+', '.join(str(x) for x in temp)+"}"
                    self.tensor_metadata.append(tensor_meta)
                    

                    # Num of nodes, max bond dimension, total weights in the tensors 
                    self.num_nodes+=1
                    self.max_bond_dim = max(a,b,c,self.max_bond_dim)
                    self.num_weights+= (a*b*c)

                    # Num of classes
                    if(key == "0.0"):
                        self.classes = min(a,b,c) # or maybe just c is better?
                        
                    
                    
                    f.write(struct.pack("iii",a,b,c))
                    flat_weights = weights.flatten()
                    
                    if(key == "0.0" and self.is_top_iso):
                    	print("Pre-normalized:", flat_weights)
                    	norm = np.linalg.norm(flat_weights)
                    	print("Norm:", norm)
                    	flat_weights = flat_weights / norm
                    	print("Normalized Weights", flat_weights)
                    
                    f.write(flat_weights.astype(np.float64))
                    
                    
                    #print(f"Weights{weights}")
                    
                    # Weight values
                    self.tensor_values+=(flat_weights.tolist())
                    print(f"Saved .npz as binary")
        
        self.height = int(np.log2(self.num_nodes + 1))
        self.num_features = 2**(self.height)
                    
    def process_directory(self, input_dir, output_dir):
        """Process all .npy and .npz files in the input directory."""
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for npy_file in input_dir.glob("*.npy"):
            output_file = output_dir / f"{npy_file.stem}.bin"
            print("New File to process", output_file)
            self.save_npy_as_bin(npy_file, output_file)
            
            txt_file = output_dir / f"{npy_file.stem}.txt"
            print("Generating corresponding readable .txt file")
            self.save_npy_as_dat(npy_file, txt_file)
            
        for npz_file in input_dir.glob("*.npz"):
            npz_output_dir = output_dir / npz_file.stem
            self.save_npz_as_bin(npz_file, npz_output_dir)
    
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



