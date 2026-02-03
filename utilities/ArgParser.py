import argparse

def tenet_args():
    parser = argparse.ArgumentParser(description = "Process the dataset to obtain binaries for HLS processing")
    parser.add_argument("-c","--csim", action="store_true", help="Run C-Simulation")
    parser.add_argument("-b","--build", action="store_true", help="Run Synthesis build")
    parser.add_argument("-p","--pack", action= "store_true", help="Create RTL IP")
    parser.add_argument("directory_path", type = str, help = "Path to the dataset to be processed")
    parser.add_argument("--top-iso-ttn", action = "store_true", help = "Top-Isometrized TTN")
    args = parser.parse_args()

    if(args.csim):
        print("[TENET-INFO]: HLS C-Simulation Task Scheduled")
    
    if(args.build):
        print("[TENET-INFO]: CSynthesis Task Scheduled")
    
    if(args.pack):
        print("[TENET-INFO]: Vivado RTL IP Packaging Scheduled")

    print("\n")
    return args


def vitis_args():
    parser = argparse.ArgumentParser(description = "Arguments for Vitis tasks")
    parser.add_argument("-c","--csim", action="store_true", help="Run C-Simulation")
    parser.add_argument("-b","--build", action="store_true", help="Run Synthesis build")
    parser.add_argument("-p","--pack", action= "store_true", help="Create RTL IP")
    args = parser.parse_args()

    return args