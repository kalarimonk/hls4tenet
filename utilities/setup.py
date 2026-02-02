import argparse
import re
import os
import numpy as np
import subprocess
from NumpyProcessor import NumpyFileProcessor
from utilities.ArgParser import get_tenet_args
# from run_vitis_simulation import run_vitis_operation



TEMPLATE_FILE = "./header_templates/tenet.h.in"
WEIGHT_TEMPLATE = "./header_templates/tensors.h.in"
DISPATCHER_TEMPLATE = "./header_templates/dispatcher.h.in"

def generate_dispatcher(template_path, output_path, processor):
    with open(template_path, 'r') as file:
        content = file.read()
	
    matched_str = re.search("__TEMPLATE_STR_START__(.*?)__TEMPLATE_STR_END__", content, re.DOTALL)
    
    if matched_str:
        template_str = matched_str.group(1)
    else:
        print("Process failed! No template string")
        return

    print("Generating Dispatcher cases")
	
    tensor_config = processor.get_dispatch_config()
    new_content_parts = []
    
    #print(tensor_config)
    
    for config in tensor_config:
        temp_str = template_str
        temp_str = temp_str.replace("__DIM1__",str(config[0]))
        temp_str = temp_str.replace("__DIM2__",str(config[1]))
        temp_str = temp_str.replace("__DIM3__",str(config[2]))
        temp_str = temp_str.replace("__CONFIG_ID__", str(config[4]))
        new_content_parts.append(temp_str)
        #print(new_content_parts)
    
    new_content = ''.join(new_content_parts)
    
    content = re.sub(r"__TEMPLATE_STR_START__(.*?)__TEMPLATE_STR_END__", str(new_content), content, flags = re.DOTALL)
    
    
    with open(output_path, 'w') as file:
        file.write(content)


def generate_header(template_path, output_path, replacements):
    with open(template_path, 'r') as file:
        content = file.read()
    
    for key, value in replacements.items():
        content = content.replace(key, str(value))
    
    with open(output_path, 'w') as file:
        file.write(content)

def generate_macro_header(template_path, output_path, processor):

    print("Generating tensor macros")
    input_spinorial_mapping = processor.get_input_map_dim()
    last_layer_count = int(np.pow(2, processor.get_tree_height() - 1))
    
    bit_width = 16
    bit_to_pack = processor.get_input_map_dim() * processor.get_num_features() * bit_width

    # Define replacements
    replacements = {
        "__FEATURES__": f"{processor.get_num_features()}",
        "__IN_DIM__": f"{input_spinorial_mapping}",
        "__MAX_BOND_DIM__": f"{processor.get_max_bond_dimension()}",
        "__NUM_NODE__": f"{processor.get_num_node()}",
        "__NUM_WEIGHTS__": f"{processor.get_total_weights()}",
        "__CLASSES__": f"{processor.get_num_classes()}",
        "__HEIGHT__": f"{processor.get_tree_height()}",
        "__LAST_LAYER_COUNT__": f"{last_layer_count}",
        "__BITS_PACKED__": f"{bit_to_pack}"
    }
    generate_header(template_path, output_path, replacements)

def generate_tensor_data(template_path, output_path, processor):
    print("Generating Tensor Values")
    replacements = {
        "__NUM_WEIGHTS__" : f"{processor.get_total_weights()}",
        "__NUM_NODES__" : f"{processor.get_num_node()}",
        "__tensor_values__" : f"{', '.join(str(x) for x in processor.get_tensor_values())}",
        "__tensor_metadata__" : f"{', '.join(str(y) for y in processor.get_tensor_metadata())}"
    }    
    generate_header(template_path, output_path, replacements)

def main(args):

    is_top_iso = args.top_iso_ttn
    print('\nc: ',args.csim, '\t b:', args.build, '\tp:', args.pack)
    
    base_directory = args.directory_path
    processor = NumpyFileProcessor(base_directory, is_top_iso)
    processor.process_directory(base_directory, base_directory + "/processed")
    
    print(f"Total number of nodes in the tree: {processor.get_num_node()}")
    print(f"Max Bond Dimensions: {processor.get_max_bond_dimension()}")
    print(f"Total number of weights: {processor.get_total_weights()}")
    print(f"Total number of classes: {processor.get_num_classes()}")
    #print(f"Tensor Values for this network: {(processor.get_tensor_values())}")
    
    print(f"Height: {processor.get_tree_height()}")
    print(f"Num features: {processor.get_num_features()}")
    
    generate_macro_header(TEMPLATE_FILE, "../hls_tensor.h", processor)
    generate_tensor_data(WEIGHT_TEMPLATE, "../hls_weights.h", processor)
    generate_dispatcher(DISPATCHER_TEMPLATE,"../hls_dispatcher.cpp", processor)


def launch_vitis_subprocess(args):
    arg_str = "-"
    if args.csim:
        arg_str+="c"
    if args.build:
        arg_str+="b"
    if args.pack:
        args_str+="p"

    print("Vitis arg string:", arg_str)

    subprocess.run("vitis -s run_vitis_simulation.py {arg_str}", shell=True)
    

if __name__ == "__main__":
    print("--------------------Initializing TENET--------------------")
    print("[INFO]:Tasks Scheduled:")
    print("[INFO]:TTN Model Processing and HLS Generation.")
    args = get_tenet_args()
    # main(args)
    launch_vitis_subprocess(args)
