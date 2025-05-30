#include <iostream>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <string>
#include <iomanip>
#include <inttypes.h>

#define SAMPLES 80
#define NUM_FEATURES 16 //4
#define IN_MAP_DIM 2

using namespace std;

int16_t cast_to_fixed16_2(double value) {
    int16_t fixed_val = static_cast<int16_t>((value * (1 << 14))); // 2^14
    return fixed_val;
}

void print_uint128_hex(__uint128_t val) {
    uint64_t lower = static_cast<uint64_t>(val);
    uint64_t upper = static_cast<uint64_t>(val >> 64);
    cout << "Packed 128-bit value (hex): 0x"
         << hex << setw(16) << setfill('0') << upper
         << setw(16) << setfill('0') << lower << dec << endl;
}

string uint128_to_string(__uint128_t value) {
    if (value == 0) return "0";
    string result;
    while (value > 0) {
        result = char('0' + (value % 10)) + result;
        value /= 10;
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    ifstream ifile(input_file, ios::binary);
    ofstream ofile(output_file, ios::binary);

    if (!ifile || !ofile) {
        cerr << "Failed to open input or output file.\n";
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
                cout << "Original Value: " << setprecision(16) << temp << ", Fixed-Point: " << fixed_temp << endl;

                index += 16;

                uint16_t byte = static_cast<uint16_t>(fixed_temp) & 0xffff;
                printf("Byte Written: 0x%" PRIx16 "\n", byte);
                ofile.write(reinterpret_cast<char*>(&byte), 2);
            }
        }

        cout << "\nIndex: " << index << endl;

        if (index < 256) {
            for (int b = 15; b >= 0; --b) {
                uint8_t byte = 0x00;
                ofile.write(reinterpret_cast<char*>(&byte), 1);
            }
        }
    }

    ifile.close();
    ofile.close();

    return 0;
}

