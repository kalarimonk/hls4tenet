import vitis
import os
from utilities.ArgParser import get_tenet_args
import argparse
import datetime

def run_vitis_operation(args):
    # Create unique workspace name per run to avoid lock conflicts
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_path = f"../tenet_workspace"

    # Ensure workspace directory exists
    os.makedirs(workspace_path, exist_ok=True)

    

    print("---------- Creating Vitis Client -------------")
    client = vitis.create_client()

    try:
        client.set_workspace(path=workspace_path)

        # Clean up component if it exists
        try:
            comp = client.get_component(name="tenet_comp")
            print("----------- Deleting existing component for CLI execution ------------")
            client.delete_component(name="tenet_comp")
        except Exception:
            # Component not found, safe to continue
            pass

        # Create new component
        comp = client.create_hls_component(
            name="tenet_comp",
            cfg_file=["C:/kalarimonk/hls4tenet/hls_config.cfg"],
            template="empty_hls_component"
        )

        if args.csim:
            print("\n----------- Running C-Simulation ------------")
            comp.run(operation="C_SIMULATION")

        if args.build:
            print("\n----------- Running Synthesis ------------")
            comp.run(operation="SYNTHESIS")
        
        if args.pack:
            print("\n----------- Creating IP ------------")
            comp.run(operation="SYNTHESIS")
            comp.run(operation="PACKAGE")

    finally:
        # Always dispose to release locks
        vitis.dispose()



if __name__ == "__main__":
    args = get_tenet_args()
    print("Args:",args)
    run_vitis_operation(args)