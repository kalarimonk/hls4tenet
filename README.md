
# High Level Synthesis (HLS) based Hardware Implementation of Tree Tensor Networks (TTN)

This project is developed to provide a HLS based framework to ease the design and hardware implementation of tensor networks on hardware accelerators like FPGAs with the primary application being classification of jet-substructure in the field of High Energy Physics (HEP). 


**Prerequisites**:  `Vitis`, `Vivado`, `Python3`

## Directory structure

The repository is organized as follows:

```
.
├── README.md
├── datasets/
├── results/
│    ...
├── hls_config.cfg
├── hls_contraction.h
├── hls_dispatcher.cpp
├── hls_dispatcher.h
├── hls_tenet.cpp
├── hls_tenet.h
├── hls_tensor.h
├── hls_weights.h
│    ...
│    ...
├── tenet_tb.cpp
├── src
│   ├── NumpyProcessor.py
│   ├── ArgParser.py
│   ├── run_vitis_simulation.py
│   └── setup.py
├── templates
│   |── dispatcher.h.in
│   |── tenet.h.in
│   └── tensors.h.in
├── utilities
│   ├── data_quantization.cpp
│   ├── old_misc/

```

The folders `datasets` and `results` are representative of where the datasets and results are placed. 

## TENET

The tool allows the user to process the dataset/model, generate HLS code dependent on the model and run Vitis tasks like C-Simulation, Synthesis and RTL Packaging all in a single execution without ever launching the Vitis GUI. The following is a short-guide on how to setup and run this tool

### HLS Configuration

The `hls_config.cfg` file maintains a list of parameters required by Vitis for the execution of its tasks. These include the FPGA part, clock and target files to name a few. Update this file according to your requirements.

The Vitis C-simulation uses the `tenet_tb.cpp` file as the test bench. The arguments to the test bench must be updated in the `hls_config.cfg` configuration file as follows.

```
csim.argv = <absolute_path_to_weights_file> <absolute_path_to_input_data_file> <absolute_path_to_output_results_file>

```

### Dataset and Tensor Networks based ML Model

The dataset files and the weights of the tensor network based ML model trained on the dataset are provided as `.npy` and `.npz` numpy formats respectively. These files are stored under the sub-folders `set1, set2, ...` under the datasets folder.

Note: The model is not trained in this project rather obtained from a sister-project for our hardware implementation

### Running the tool

Before running the tool, the user must source the Vitis tool by running the following command


```
Linux

$ source /tools/Xilinx/Vitis/<version>/settings64.sh
```


```
Windows

$ C:\Xilinx\Vitis\<version>\settings64.bat
```

Next move to the `src` directory of the project
```
$ cd src
```

In this section, I list the various ways to utilize the TENET tool. The file `setup.py` is the single execution entry point for all the tasks the user chooses to run.

```python
$ python setup.py --help
usage: setup.py [-h] [-c] [-b] [-p] [--top-iso-ttn] directory_path

Process the dataset to obtain binaries for HLS processing

positional arguments:
  directory_path  Path to the dataset to be processed

optional arguments:
  -h, --help      show this help message and exit
  -c, --csim      Run C-Simulation
  -b, --build     Run Synthesis build
  -p, --pack      Run Synthesis and Create RTL IP Package
  --top-iso-ttn   Top-Isometrized TTN

```

The user must specify the folder of the model files as it is the primary requirement of the HLS code generation. After specifying the dataset folder the user can specify the Vitis task they wish to execute. 

**Note**: However, do not specify the `-b` option alongside RTL packaging `-p`, since the tool internally runs the synthesis prior creating the RTL package. The result of the C-simultation will be stored at the output location specified in the configuration file under `csim.argv`


```python
Examples
1.python setup.py ../datasets/set1 -c  (Runs C-Simulation)

2.python setup.py ../datasets/set1 -b (Runs Synthesis)

3.python setup.py ../datasets/set1 -cp (Runs C-Simulation, RTL packaging)

```

All Vitis tasks require a workspace folder and a Vitis component to be executed. The `tenet_workspace` folder in the root of the repository provides this workspace. If the folder does not exist, the tool creates a folder with the same name. Under this workspace folder, a folder named `tenet_comp` exists which is the component folder for tasks to be operated upon.

The component folder is also the location of the simulation and synthesis reports, alongside the RTL package which is generated as `run_tenet.zip`. The zip file consists of all the HDL files needed by Vivado from Vitis to generate the bit-stream. Note, however there is still some in-Vivado work that needs to be done before generating the final bit-stream to be loaded into the FPGA.

After executing the tasks, the component `tenet_comp` can also be loaded using the Vitis GUI to view the reports, play around and re-run tasks as one would do if using just the Vitis GUI.

### Running only Vitis tasks

The user can run the entire set of tasks as specified in the above section. However after running `setup.py`, the user maynot want to run the dataset/model file processing step. In such case, one can execute just the vitis task using the `run_vitis_tasks.py` script with the same task options as above.

```
$ vitis -s run_vitis_tasks.py -h

usage: run_vitis_tasks.py [-h] [-c] [-b] [-p]

Arguments for Vitis tasks

optional arguments:
  -h, --help   show this help message and exit
  -c, --csim   Run C-Simulation
  -b, --build  Run Synthesis build
  -p, --pack   Create RTL IP
```


### Running as a model processor

The tool can also be used just to generate the HLS code and then the user can open Vitis GUI and continue operating as a standard user. For this purpose, after cloning the repository create a Vitis project and add the following files as sources and test bench. 

```
├── sources
│   ├── hls_contraction.h
│   ├── hls_dispatcher.cpp
│   ├── hls_dispatcher.h
│   ├── hls_tenet.cpp
│   ├── hls_tenet.h
│   ├── hls_tensor.h
│   ├── hls_weights.h
├── Test Bench
│   └── tenet_tb.cpp

```

The following command can be run to update the HLS files depending on the user's model/data location and the user can then continue tinkering with these in Vitis.

```python
$ python setup.py ../datasets/set2
```


## Common Errors

1. If the tool may throw an exception saying that the workspace is under use and fail execution. In such case just delete the `tenet_workspace` folder and re-run the tool. This should fix this error.

2. Forgetting to run the source command for vitis will also fail execution of the vitis tasks, always remember to run the source vitis command.




