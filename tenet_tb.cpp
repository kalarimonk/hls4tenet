#define AP_INT_MAX_W 2048
#include <ap_int.h>
#include <ap_fixed.h>

#include "./hls_tensor.h"
#include "./hls_tenet.h"

#include <iostream>
#include <fstream>
#include <hls_stream.h>
#include <math.h>
#include <inttypes.h>

#define NUM_SAMPLES 80
#define PRECISION 16

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <weights_file> <input_data_file> <output_results_file>\n";
        return 1;
    }

    const char* weights_file_path = argv[1];
    const char* input_file_path = argv[2];
    const char* result_file_path = argv[3];

    Tensor3D tensor_d[NUM_NODES];
    tensor_t weight1d[NUM_PARAM];

    hls::stream<tensor_t> test_data[NUM_SAMPLES][NUM_FEATURES][IN_MAP_DIM];
    hls::stream<tensor_t> output[ROOT_NODE_DIM];

    ofstream result_file(result_file_path, ios::out);
    if (!result_file) {
        cerr << "Error opening file for writing results.\n";
        return 1;
    }

    ifstream weights_file(weights_file_path, ios::binary);
    if (!weights_file) {
        cerr << "Error: Cannot open weights file!\n";
        return 1;
    }

    tensor_t weights[NUM_NODES][MAX_BOND_DIM][MAX_BOND_DIM][MAX_BOND_DIM];

    int num_tensors;
    weights_file.read(reinterpret_cast<char*>(&num_tensors), sizeof(int));

    int curr_wptr = 0;
    for (int t = 0; t < num_tensors; t++) {
        int a, b, c;
        weights_file.read(reinterpret_cast<char*>(&a), sizeof(int));
        weights_file.read(reinterpret_cast<char*>(&b), sizeof(int));
        weights_file.read(reinterpret_cast<char*>(&c), sizeof(int));

        double temp;
        for (int k = 0; k < a * b * c; k++) {
            weights_file.read(reinterpret_cast<char*>(&temp), sizeof(double));
            weight1d[curr_wptr + k] = tensor_t(temp);
        }
        curr_wptr += a * b * c;
    }

    ifstream ifile(input_file_path, ios::binary);
    if (!ifile) {
        cerr << "Error opening input data file\n";
        return 1;
    }

    tensor_t out[ROOT_NODE_DIM];

    for (int n = 0; n < NUM_SAMPLES; n++) {
        hls::stream<ap_uint<BITS_PACKED>> sample_new;
        ap_uint<BITS_PACKED> packed = 0;

        int index = 0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            double temp;
            for (int k = 0; k < IN_MAP_DIM; k++) {
                ifile.read(reinterpret_cast<char*>(&temp), sizeof(double));
                tensor_t raw_bits = tensor_t(temp);
                packed.range(index + PRECISION - 1, index) = raw_bits.range(PRECISION - 1, 0);
                index += PRECISION;
            }
        }

        sample_new.write(packed);
        run_tenet(sample_new, output);

        for (int k = 0; k < ROOT_NODE_DIM; ++k) {
            if (!output[k].empty()) {
                out[k] = output[k].read();
                result_file << setprecision(PRECISION - 2) << out[k] << endl;
            }
        }
    }

    ifile.close();
    weights_file.close();
    result_file.close();

    return 0;
}

