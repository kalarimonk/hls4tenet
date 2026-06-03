import argparse
import os
from pathlib import Path


def vitis_args(argv=None):
    parser = argparse.ArgumentParser(description="Arguments for Vitis tasks")
    parser.add_argument("-c", "--csim", action="store_true", help="Run C-Simulation")
    parser.add_argument("-b", "--build", action="store_true", help="Run synthesis build")
    parser.add_argument("-p", "--pack", action="store_true", help="Create RTL IP")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("tenet_workspace"),
        help="Workspace directory for Vitis runs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to hls_config.cfg",
    )
    return parser.parse_args(argv)


def run_vitis_operation(args) -> None:
    import vitis

    workspace_path = Path(args.workspace).resolve()
    
    os.makedirs(workspace_path, exist_ok=True)

    config_path = Path(args.config).resolve()
    print("--------------------Creating Vitis Client--------------------")

    client = vitis.create_client()

    try:
        print(f"Using Vitis workspace at: {workspace_path}")
        client.set_workspace(path=str(workspace_path))

        try:
            client.get_component(name="tenet_comp")
            print("--------------------Deleting existing component for CLI execution--------------------")
            client.delete_component(name="tenet_comp")
        except Exception:
            pass

        comp = client.create_hls_component(
            name="tenet_comp",
            cfg_file=[str(config_path)],
            template="empty_hls_component",
        )

        if args.csim:
            print("\n--------------------Running C-Simulation--------------------")
            comp.run(operation="C_SIMULATION")

        if args.build:
            print("\n--------------------Running Synthesis--------------------")
            comp.run(operation="SYNTHESIS")

        if args.pack:
            print("\n--------------------Creating IP Package--------------------")
            comp.run(operation="SYNTHESIS")
            comp.run(operation="PACKAGE")

    finally:
        print("Disposing Vitis")
        vitis.dispose()

    if args.pack:
        print("\n\n[TENET-INFO]: IP package created successfully. You can find run_tenet.zip in the Vitis workspace under tenet_workspace/tenet_comp/tenet_comp/")

def main(argv=None):
    args = vitis_args(argv)
    run_vitis_operation(args)


if __name__ == "__main__":
    main()
