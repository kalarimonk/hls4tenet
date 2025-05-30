import os
import argparse

TEMPLATE_FILE = "./vivado_tb_template.txt"
OUTPUT_FILE = "./vivado_tb.txt"

def generate_header(template_path, output_path, replacements):
    with open(template_path, 'r') as file:
        content = file.read()

    for key, value in replacements.items():
        content = content.replace(key, str(value))

    with open(output_path, 'a') as file:
        file.write(content)


def main():
    parser = argparse.ArgumentParser(description = "Code generation for Vivado Simulation")
    parser.add_argument("control_reg", type = str, help = "AXI-lite control register")
    parser.add_argument("weight", type = str, help = "Weight to be stored in the register")
    args = parser.parse_args()
    
    control_reg = hex(int(args.control_reg) )
    data = hex(int(args.weight) & ((1 << 32) - 1) & (0x0000ffff)) # to get a 16 bit word 
    
    print("Control Reg:", str(control_reg).replace('0x',''))
    print("Weight:", str(data).replace('0x',''))

    replacements = {
    	"__ADDRESS__" : f"{str(control_reg).replace('0x','')}",
    	"__DATA__" : f"{str(data).replace('0x','')}"
    }
    
    generate_header(TEMPLATE_FILE, OUTPUT_FILE, replacements)

if __name__ == "__main__":
    main()	
