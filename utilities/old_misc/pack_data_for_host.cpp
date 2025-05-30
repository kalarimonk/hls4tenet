#include <iostream>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <string>
#include <iomanip>
#include <inttypes.h>

//#define OUTPUT_FILE "/home/pgupta/tenet_hls/datasets/set1/processed/data.bin"
//#define INPUT_FILE "/home/pgupta/tenet_hls/datasets/set1/processed/iris_embedded.bin"


#define OUTPUT_FILE "/home/pgupta/tenet_hls/datasets/set2/processed/data.bin"
#define INPUT_FILE "/home/pgupta/tenet_hls/datasets/set2/processed/x_train.bin"

#define SAMPLES 80
#define NUM_FEATURES 16//4
#define IN_MAP_DIM 2

using namespace std;

int16_t cast_to_fixed16_2(double value) {
    int16_t fixed_val = static_cast<int16_t>((value * (1 << 14))); // 2^14
    return fixed_val;
}

// Optional: Print uint128 in hex format
void print_uint128_hex(__uint128_t val) {
    uint64_t lower = static_cast<uint64_t>(val);
    uint64_t upper = static_cast<uint64_t>(val >> 64);
    std::cout << "Packed 128-bit value (hex): 0x"
              << std::hex << std::setw(16) << std::setfill('0') << upper
              << std::setw(16) << std::setfill('0') << lower << std::dec << std::endl;
}

std::string uint128_to_string(__uint128_t value) {
    if (value == 0) return "0";
    std::string result;
    while (value > 0) {
        result = char('0' + (value % 10)) + result;
        value /= 10;
    }
    return result;
}

int main() {
    ifstream ifile(INPUT_FILE, ios::binary);
    ofstream ofile(OUTPUT_FILE, ios::binary);

    if (!ifile || !ofile) {
        cerr << "Failed to open file.\n";
        return 1;
    }

    for (int i = 0; i < SAMPLES; i++) {
        __uint128_t packed128 = 0;
        int index = 0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            double temp;
            for (int k = 0; k < IN_MAP_DIM; k++) {
                ifile.read(reinterpret_cast<char*>(&temp), sizeof(double));
                int16_t fixed_temp = cast_to_fixed16_2(temp);
                printf("Hex Lowercase: 0x%" PRIx16 "\n", fixed_temp);
               // ofile.write(reinterpret_cast<char*>(&fixed_temp), 1);
                cout << "Original Value: " << setprecision(16) << temp << ", Fixed-Point: " << fixed_temp << endl;
                //packed128 |= static_cast<__uint128_t>(fixed_temp) << index;
                //cout << "Packed (decimal): " << uint128_to_string(packed128) << endl;
                index+=16;
                
                uint16_t byte = static_cast<uint16_t>(fixed_temp) & 0xffff;
                printf("Byte Written: 0x%" PRIx16 "\n", byte);
	        ofile.write(reinterpret_cast<char*>(&byte), 2);
            }
        }
        
        cout << "\nIndex:"  << index <<endl;      
        /* // Write packed128 to binary file as 16 bytes
        for (int b = 0; b < 16; b++) {
            uint8_t byte = static_cast<uint8_t>(packed128 >> (b * 8)) & 0xff;
	        ofile.write(reinterpret_cast<char*>(&byte), 1);
        }*/
        if  (index < 256) {
		for (int b = 15; b >= 0; --b) {
		    uint8_t byte = 0x0000;
		    ofile.write(reinterpret_cast<char*>(&byte), 1);
		}
        }
        
	//print_uint128_hex(packed128);
    }

    ifile.close();
    ofile.close();
    
    return 0;
}

