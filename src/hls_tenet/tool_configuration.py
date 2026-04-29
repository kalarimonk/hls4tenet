
from pathlib import Path



class ToolConfiguration:
    def __init__(self):
        self.tensor_network_structure: str = "MPS"
        self.model_files_path: Path = Path("")
        self.dataset_directory: Path = Path("")
        self.output_dir: Path = Path("")
        self.tb_data_path: Path = Path("")
        self.vitis_configfile_path: Path = Path("")
        self.run_vitis_csim: bool = False
        self.run_vitis_build: bool = False 
        self.run_vitis_pack: bool = False 
        self.generate_vitis_configfile: bool = False
    
    def display_configuration(self):
        print("\nTENET Configuration Summary:")
        print(f"  - Tensor Network Structure: {self.tensor_network_structure}")
        print(f"  - Model Files Path: {self.model_files_path}")
        print(f"  - Dataset Directory: {self.dataset_directory}")
        print(f"  - Output Directory: {self.output_dir}")
        print(f"  - Testbench Data Path: {self.tb_data_path}")
        if self.run_vitis_csim or self.run_vitis_build or self.run_vitis_pack:
            print(f"  - Run Vitis: Yes")
            print(f"    - Run C-Simulation: {'Yes' if self.run_vitis_csim else 'No'}")
            print(f"    - Run Synthesis Build: {'Yes' if self.run_vitis_build else 'No'}")
            print(f"    - Create RTL IP Package: {'Yes' if self.run_vitis_pack else 'No'}")
            print(f"    - Vitis Configuration File: {self.vitis_configfile_path if not self.generate_vitis_configfile else f'${self.vitis_configfile_path} (will be generated)'}")
        else :
            print(f"  - Run Vitis: No Vitis tasks scheduled.")

        
        