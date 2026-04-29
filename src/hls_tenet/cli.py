import argparse
import os
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Optional

from hls_tenet.mps_generators import generate_mpsmacro_header, generate_mpsweight_file, generate_top_module

from .generators import generate_config, generate_dispatcher, generate_macro_header, generate_tensor_data, generate_immutables
from .processor import MPSProcessor, TTNProcessor, process_directory
from .tool_configuration import ToolConfiguration
  
    

def tenet_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Process dataset/model files and generate HLS artifacts for TENET"
    )
    
    # subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to execute")
    # parser_gen = subparsers.add_parser("generate", help="Generate HLS code and Vitis config")
    
    parser.add_argument("model_path", type=Path, help="Path to model file (e.g. .pkl, .npz)")
    parser.add_argument("directory_path",type=Path, help="Path to data directory")
    parser.add_argument("tn",choices=["mps","ttn"], help="Tensor Network Structure")
    
    parser.add_argument("-c", "--csim", action="store_true", help="Run C-Simulation ")
    parser.add_argument("-tb","--tb_data", type=Path, help="Path to testbench data file for C-sim (e.g. test.bin)")
    
    parser.add_argument("-b", "--build", action="store_true", help="Run synthesis build")
    parser.add_argument("-p", "--pack", action="store_true", help="Create RTL IP package")
    
    # parser.add_argument("--top-iso-ttn", action="store_true", help="Top-Isometrized TTN")
    return parser.parse_args(argv)


def _resolve_templates_dir(tn_model: str) -> Path:
    resource_root = files("hls_tenet").joinpath("templates"+os.sep+tn_model.lower())
    return Path(str(resource_root))


def run_generation(configuration: ToolConfiguration) -> TTNProcessor or MPSProcessor:
    
    # tb_data_dir = Path(os.getcwd()).resolve().joinpath("testbench_data")
    # tb_data_dir.mkdir(parents=True, exist_ok=True)
    
    # process_directory(configuration.dataset_directory, tb_data_dir) 
    
    if configuration.tensor_network_structure == "TTN":
        processor = TTNProcessor(configuration.model_files_path)
        
        print(f"Total number of nodes in the tree: {processor.get_num_node()}")
        print(f"Max bond dimensions: {processor.get_max_bond_dimension()}")
        print(f"Total number of weights: {processor.get_total_weights()}")
        print(f"Total number of classes: {processor.get_num_classes()}")
        print(f"Height: {processor.get_tree_height()}")
        print(f"Num features: {processor.get_num_features()}")
        templates_dir = _resolve_templates_dir(configuration.tensor_network_structure)

        generate_macro_header(templates_dir / "tensor.h.in", configuration.output_dir.joinpath("hls_tensor.h"), processor)
        generate_tensor_data(templates_dir / "weights.h.in", configuration.output_dir / "hls_weights.h", processor)
        generate_dispatcher(templates_dir / "dispatcher.cpp.in", configuration.output_dir / "hls_dispatcher.cpp", processor)

        generate_config(templates_dir/ "config.cfg.in", configuration.vitis_configfile_path, configuration.tb_data_path)

        generate_immutables(templates_dir/ "tenet.h.in", configuration.output_dir/"hls_tenet.h", processor)
        generate_immutables(templates_dir/ "tenet.cpp.in", configuration.output_dir/"hls_tenet.cpp", processor)
        generate_immutables(templates_dir/ "dispatcher.h.in", configuration.output_dir/"hls_dispatcher.h", processor)
        generate_immutables(templates_dir/ "contraction.h.in", configuration.output_dir/"hls_contraction.h", processor)
        
        generate_immutables(templates_dir/ "tenet_tb.cpp.in", configuration.output_dir/"tenet_tb.cpp", processor)
        
    
    if configuration.tensor_network_structure == "MPS":
        processor = MPSProcessor(configuration.model_files_path)
        
        print(f"\nTotal number of nodes in the network: {processor.num_features}")
        print(f"\nMax bond dimension: {processor.max_bond_dim}")
        print(f"\nPhysical bond dimension: {processor.phy_bond_dim}")

        templates_dir = _resolve_templates_dir(configuration.tensor_network_structure)

        generate_mpsmacro_header(templates_dir / "tensor.h.in", configuration.output_dir.joinpath("hls_tensor.h"), processor)
        generate_mpsweight_file(templates_dir / "weights.h.in", configuration.output_dir / "hls_weights.h", processor)
        generate_top_module(templates_dir / "tenet.cpp.in", configuration.output_dir / "hls_tenet.cpp", processor)
        generate_config(templates_dir/ "config.cfg.in", configuration.vitis_configfile_path, configuration.tb_data_path)

        generate_immutables(templates_dir/ "tenet.h.in", configuration.output_dir/"hls_tenet.h", processor)
        generate_immutables(templates_dir/ "contraction.h.in", configuration.output_dir/"hls_contraction.h", processor)
        generate_immutables(templates_dir/ "tenet_tb.cpp.in", configuration.output_dir/"tenet_tb.cpp", processor)
        
        
    
    return processor




def launch_vitis_subprocess(configuration: ToolConfiguration) -> None:
    if not (configuration.run_vitis_csim or configuration.run_vitis_build or configuration.run_vitis_pack):
        print("[TENET-INFO]: No Vitis tasks scheduled. Skipping Vitis launch.")
        return
    
    vitis_script = Path(__file__).with_name("vitis_runner.py")
    cmd = ["vitis", "-s", str(vitis_script)]
    
    if configuration.run_vitis_csim:
        cmd.append("-c")
    if configuration.run_vitis_build:
        cmd.append("-b")
    if configuration.run_vitis_pack:
        cmd.append("-p")

    cmd.extend(["--workspace", Path("tenet_workspace"), "--config", str(configuration.vitis_configfile_path)])

    print("--------------------Launching Vitis subprocess--------------------")
    subprocess.run(cmd, check=True)


def main(argv=None):
    args = tenet_args(argv)  
    
    configuration = ToolConfiguration()  
    configuration.tensor_network_structure = args.tn.upper()
    
    #Check for mandatory valid args
    if args.model_path.suffix.lower() not in [".pkl", ".npz"]:
        print(f"Unsupported model file type: {args.model_path.suffix}. Supported types are .pkl and .npz")
        return 1
    
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")
    
    dataset_dir = args.directory_path.resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    
    configuration.model_files_path = args.model_path.resolve()
    configuration.output_dir = Path(os.getcwd()+os.sep+"generated_hls")
    
    
    # Remove the directory and its contents if it exists
    if configuration.output_dir.exists() and configuration.output_dir.is_dir():
        shutil.rmtree(configuration.output_dir)  # Deletes the directory recursively.  Clean slate for generation.
    
    configuration.output_dir.mkdir(parents=True, exist_ok=True)
    configuration.dataset_directory = dataset_dir
    
    if args.csim and not args.tb_data:
        print(f"Testbench data file is required for C-Simulation. eg: --tb_data x_test.bin")
        return 1
    
    if args.tb_data and not args.csim :
        print(f"Testbench data file is only relevant if C-Simulation is enabled. Please provide --csim flag to enable C-Simulation.")
        return 1
    
    print("Testbench data file provided:", args.tb_data)
    
    if args.tb_data is not None:
        if args.tb_data.suffix.lower() not in [".bin"]:
            print(f"Unsupported testbench data file type: {args.tb_data.suffix}. Supported types are .bin")
            return 1
        configuration.tb_data_path = args.tb_data.resolve()
        
    configuration.run_vitis_csim = args.csim
    configuration.run_vitis_build = args.build
    configuration.run_vitis_pack = args.pack
    configuration.vitis_configfile_path = configuration.output_dir.joinpath("hls_config.cfg")

    configuration.display_configuration()
    
    print("\n\n--------------------Initializing TENET--------------------")
    
    run_generation(configuration)
        

    try:
        launch_vitis_subprocess(configuration)
    except FileNotFoundError:
        print("vitis executable not found in PATH. Ensure Vitis environment is sourced.")
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"Vitis task failed with exit code {exc.returncode}")
        return exc.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
