import re
from pathlib import Path

import numpy as np

from hls_tenet.generators import generate_header
from hls_tenet.processor import MPSProcessor


def generate_top_module(template_path: Path, output_path: Path, processor) -> None:
    content = template_path.read_text(encoding="utf-8")
    str_literal_prefix = ""
    str_literal_suffix = ""
    for i in range(processor.num_nodes):
        if i == 0 and i != processor.label_site_id:
            str_literal_prefix = "__START_LEFTMOST_KERNELS__"
            str_literal_suffix = "__LEFTMOST_KERNELS_END__"
            matched_str = re.search(f"{str_literal_prefix}(.*?){str_literal_suffix}", content, re.DOTALL)
            if not matched_str:
                raise ValueError("Process failed: template string markers not found")

            template_str = matched_str.group(1)
            a, b, c = processor.tensor_metadata[i]
            # print("Generating kernel calls for leftmost kernel")
        elif i < processor.label_site_id:
            str_literal_prefix = "__START_CALL_LEFT_ENV_KERNELS__"
            str_literal_suffix = "__CALL_LEFT_ENV_KERNELS_END__"
            matched_str = re.search(f"{str_literal_prefix}(.*?){str_literal_suffix}", content, re.DOTALL)
            if not matched_str:
                raise ValueError("Process failed: template string markers not found")

            template_str = matched_str.group(1)
            a, b, c = processor.tensor_metadata[i]
            # print(f"Generating kernel calls for left kernel {i}")
            
        new_content_parts = []   
        temp_str = template_str
        temp_str = temp_str.replace("__LEFT_DIM__", str(a))
        temp_str = temp_str.replace("__RIGHT_DIM__", str(b))
        temp_str = temp_str.replace("__PHY_DIM__", str(c))
        temp_str = temp_str.replace("__ID__", str(i))
        new_content_parts.append(temp_str)
        
        new_content = "".join(new_content_parts)
        content = re.sub(
            rf"{str_literal_prefix}(.*?){str_literal_suffix}",
            str(new_content),
            content,
            flags=re.DOTALL,
        )
            
    
    
    for i in range(processor.num_nodes-1,0, -1):   
        # print(f"\nProcessing node {i} for right environment kernels")
        if i == processor.num_nodes - 1 and i != processor.label_site_id:
            str_literal_prefix = "__START_RIGHTMOST_KERNELS__"
            str_literal_suffix = "__RIGHTMOST_KERNELS_END__"
            matched_str = re.search(f"{str_literal_prefix}(.*?){str_literal_suffix}", content, re.DOTALL)
            if not matched_str:
                raise ValueError("Process failed: template string markers not found")

            template_str = matched_str.group(1)
            a, b, c = processor.tensor_metadata[i]
            # print(f"Generating kernel calls for rightmost kernel")
            
        elif i > processor.label_site_id:
            str_literal_prefix = "__START_CALL_RIGHT_ENV_KERNELS__"
            str_literal_suffix = "__CALL_RIGHT_ENV_KERNELS_END__"
            matched_str = re.search(f"{str_literal_prefix}(.*?){str_literal_suffix}", content, re.DOTALL)
            if not matched_str:
                raise ValueError("Process failed: template string markers not found")

            template_str = matched_str.group(1)
            a, b, c = processor.tensor_metadata[i]
            # print(f"Generating kernel calls for right kernel {i}")
        
        new_content_parts = []
        temp_str = template_str
        temp_str = temp_str.replace("__LEFT_DIM__", str(a))
        temp_str = temp_str.replace("__RIGHT_DIM__", str(b))
        temp_str = temp_str.replace("__PHY_DIM__", str(c))
        temp_str = temp_str.replace("__ID__", str(i))
        new_content_parts.append(temp_str)

        new_content = "".join(new_content_parts)
        content = re.sub(
            rf"{str_literal_prefix}(.*?){str_literal_suffix}",
            str(new_content),
            content,
            flags=re.DOTALL,
        )
        
    
    # Handle label site kernel separately
    str_literal_prefix = "__LABEL_KERNEL_START__"
    str_literal_suffix = "__LABEL_KERNEL_END__"
    matched_str = re.search(f"{str_literal_prefix}(.*?){str_literal_suffix}", content, re.DOTALL)
    if not matched_str:        raise ValueError("Process failed: template string markers not found")
    template_str = matched_str.group(1)
    # print(f"Generating kernel call for label site kernel")
    
    new_content = template_str
    new_content = new_content.replace("__LEFT_DIM__", str(processor.tensor_metadata[processor.label_site_id][0]))
    new_content = new_content.replace("__RIGHT_DIM__", str(processor.tensor_metadata[processor.label_site_id][1]))
    new_content = new_content.replace("__PHY_DIM__", str(processor.tensor_metadata[processor.label_site_id][2]))
    new_content = new_content.replace("__LABEL_DIM__", str(processor.tensor_metadata[processor.label_site_id][3]))
    new_content = new_content.replace("__LABEL_SITE_ID__", str(processor.label_site_id))
    content = re.sub(
        rf"{str_literal_prefix}(.*?){str_literal_suffix}",
        str(new_content),
        content,
        flags=re.DOTALL,
    )
    
    # If any template markers are left unmatched, 
    # it could mean that the mps structure does not require those kernels (ex: no left environment kernels if label site is at the leftmost position). 
    # In that case, we should remove the template markers and the content in between to avoid leaving template artifacts in the final generated code.
    matched_str = re.search(f"__START(.*?)END__", content, re.DOTALL)
    while matched_str:
        content = re.sub(
            f"__START(.*?)END__",
            "",
            content,
            flags=re.DOTALL,
        )
        matched_str = re.search(f"__START(.*?)END__", content, re.DOTALL)

    output_path.write_text(content, encoding="utf-8")


def generate_mpsmacro_header(template_path: Path, output_path: Path, processor: MPSProcessor) -> None:
    print("Generating tensor macros")

    bit_width = 18

    replacements = {
        "__FEATURES__": f"{processor.num_features}",
        "__NUM_NODES__": f"{processor.num_nodes}",
        "__PHY_BOND_DIM__": f"{processor.phy_bond_dim}",
        "__MAX_BOND_DIM__": f"{processor.max_bond_dim}",
        "__LABEL_DIM__": f"{processor.classes}",
        "__LABEL_SITE_ID__": f"{processor.label_site_id}",
        "__BITS_PACKED__": f"{processor.calculate_bits_to_pack(bit_width)}",
    }
    generate_header(template_path, output_path, replacements)



def generate_mpsweight_file(template_path: Path, output_path: Path, processor: MPSProcessor) -> None:
    content = template_path.read_text(encoding="utf-8")
    
    # Generate tensor definitions for non-label sites
    matched_str = re.search("__TENSOR_STR_START__(.*?)__TENSOR_STR_END__", content, re.DOTALL)
    if not matched_str:
        raise ValueError("Process failed: template string markers not found")

    template_str = matched_str.group(1)
    print("Generating tensor definitions for weights file")

    new_content_parts = []
    for i in range(processor.num_nodes):
        if i == processor.label_site_id:
            continue  # Skip the label site for now, require special handling
        # print(f"\nProcessing tensor {i} with metadata: {processor.tensor_metadata[i]}")
        a, b, c = processor.tensor_metadata[i]
        tensor_values = processor.tensor_values[i]
        temp_str = template_str
        temp_str = temp_str.replace("__TENSOR_ID__", f"{i}")
        temp_str = temp_str.replace("left_dim", str(a))
        temp_str = temp_str.replace("right_dim", str(b))
        temp_str = temp_str.replace("phy_dim", str(c))
        temp_str = temp_str.replace("__tensor_values__",str(tensor_values).replace("[", "{").replace("]", "}"))
        new_content_parts.append(temp_str)

    new_content = "".join(new_content_parts)
    content = re.sub(
        r"__TENSOR_STR_START__(.*?)__TENSOR_STR_END__",
        str(new_content),
        content,
        flags=re.DOTALL,
    )
    
    # Handle label site tensor separately   
    label_str = re.search("__LABEL_TENSOR_STR_START__(.*?)__LABEL_TENSOR_STR_END__", content, re.DOTALL)
    if not label_str:
        raise ValueError("Process failed: template string markers not found")

    template_str = label_str.group(1)
    a, b, c, d = processor.tensor_metadata[processor.label_site_id]
    tensor_values = processor.tensor_values[processor.label_site_id]
    temp_str = template_str
    temp_str = temp_str.replace("left_dim", str(a))
    temp_str = temp_str.replace("right_dim", str(b))
    temp_str = temp_str.replace("phy_dim", str(c))
    temp_str = temp_str.replace("label_dim", str(d))
    temp_str = temp_str.replace("__label_values__",str(tensor_values).replace("[", "{").replace("]", "}"))
    content = re.sub(
        r"__LABEL_TENSOR_STR_START__(.*?)__LABEL_TENSOR_STR_END__",
        str(temp_str),
        content,
        flags=re.DOTALL,
    )
    output_path.write_text(content, encoding="utf-8")
    
    pass