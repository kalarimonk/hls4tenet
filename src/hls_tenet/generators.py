import re
from pathlib import Path

import numpy as np


def generate_dispatcher(template_path: Path, output_path: Path, processor) -> None:
    content = template_path.read_text(encoding="utf-8")
    matched_str = re.search("__TEMPLATE_STR_START__(.*?)__TEMPLATE_STR_END__", content, re.DOTALL)

    if not matched_str:
        raise ValueError("Process failed: template string markers not found")

    template_str = matched_str.group(1)
    print("Generating dispatcher cases")

    new_content_parts = []
    for config in processor.get_dispatch_config():
        temp_str = template_str
        temp_str = temp_str.replace("__DIM1__", str(config[0]))
        temp_str = temp_str.replace("__DIM2__", str(config[1]))
        temp_str = temp_str.replace("__DIM3__", str(config[2]))
        temp_str = temp_str.replace("__CONFIG_ID__", str(config[4]))
        new_content_parts.append(temp_str)

    new_content = "".join(new_content_parts)
    content = re.sub(
        r"__TEMPLATE_STR_START__(.*?)__TEMPLATE_STR_END__",
        str(new_content),
        content,
        flags=re.DOTALL,
    )

    output_path.write_text(content, encoding="utf-8")


def generate_header(template_path: Path, output_path: Path, replacements = None) -> None:
    content = template_path.read_text(encoding="utf-8")

    if replacements is not None:
        for key, value in replacements.items():
            content = content.replace(key, str(value))

    output_path.write_text(content, encoding="utf-8")

def generate_config(template_path: Path, output_path: Path, tb_data_path: Path) -> None:
    print("Generating config file")
    replacements = {
        "__OUTPUT_DIR__": output_path.parent.absolute(),
        "<tb_data_path>": tb_data_path.absolute(),
        "<result_file_path>": output_path.parent.absolute().joinpath("results.txt"),
    }
    generate_header(template_path, output_path, replacements)

def generate_macro_header(template_path: Path, output_path: Path, processor) -> None:
    print("Generating tensor macros")
    input_spinorial_mapping = processor.get_input_map_dim()
    last_layer_count = int(np.power(2, processor.get_tree_height() - 1))

    # bit_width = 16
    # bit_to_pack = processor.get_input_map_dim() * processor.get_num_features() * bit_width

    replacements = {
        "__FEATURES__": f"{processor.get_num_features()}",
        "__IN_DIM__": f"{input_spinorial_mapping}",
        "__MAX_BOND_DIM__": f"{processor.get_max_bond_dimension()}",
        "__NUM_NODE__": f"{processor.get_num_node()}",
        "__NUM_WEIGHTS__": f"{processor.get_total_weights()}",
        "__CLASSES__": f"{processor.get_num_classes()}",
        "__HEIGHT__": f"{processor.get_tree_height()}",
        "__LAST_LAYER_COUNT__": f"{last_layer_count}",
        "__BITS_PACKED__": f"{processor.calculate_bits_to_pack()}",
        "__WORD_DEPTH__": f"{processor.word_depth}",
        "__INT_BITS__": f"{processor.int_bits}",
    }
    generate_header(template_path, output_path, replacements)


def generate_tensor_data(template_path: Path, output_path: Path, processor) -> None:
    print("Generating tensor values")
    replacements = {
        "__NUM_WEIGHTS__": f"{processor.get_total_weights()}",
        "__NUM_NODES__": f"{processor.get_num_node()}",
        "__tensor_values__": f"{', '.join(str(x) for x in processor.get_tensor_values())}",
        "__tensor_metadata__": f"{', '.join(str(y) for y in processor.get_tensor_metadata())}",
    }
    generate_header(template_path, output_path, replacements)


def generate_immutables(template_path: Path, output_path: Path, processor) -> None:
    print(f"Generating {output_path.name} with immutable parameters")
    generate_header(template_path, output_path)
