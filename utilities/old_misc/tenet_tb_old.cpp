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

#define NUM_SAMPLES 1
#define PRECISION 16

using namespace std;

int main(){

/*
	Double datatype works while reading the binaries without causing incorrect conversions
	Since the numpy function saves the values in float64 and the equivalent type in C++ is double.
	Next to load these such as to run the contraction tests
*/

	Tensor3D tensor_d[NUM_NODES];
	tensor_t weight1d[NUM_PARAM];


	hls::stream<tensor_t> test_data[NUM_SAMPLES][NUM_FEATURES][IN_MAP_DIM];
	hls::stream<tensor_t> output[ROOT_NODE_DIM];

	std::ofstream result_file("/home/pgupta/new_hls/hls_tenet/results/set2_results.txt", ios::out);
	if (!result_file) {
	    std::cerr << "Error opening file for writing results.\n";
	    return 1;
	}


	cout<< "-----Executing Test Bench-----" << endl;


	// ifstream file("/home/pgupta/new_hls/hls_tenet/datasets/set1/processed/weights/weights.bin", ios::binary);
	ifstream file("/home/pgupta/new_hls/hls_tenet/datasets/set2/processed/wl16_fl14-model/weights.bin", ios::binary);
	// ifstream file("/home/pgupta/new_hls/hls_tenet/datasets/set3/processed/bound_asym/weights.bin", ios::binary);
	if (!file) {
		cerr << "Error: Cannot open file!" << endl;
		return 1;
	}

	tensor_t weights[NUM_NODES][MAX_BOND_DIM][MAX_BOND_DIM][MAX_BOND_DIM];

	int num_tensors;
	file.read(reinterpret_cast<char*>(&num_tensors), sizeof(int));
	cout << "Number of tensors: " << num_tensors << endl;

	int curr_wptr = 0;

	for (int t = 0; t < num_tensors; t++) {
		int a, b, c;
		file.read(reinterpret_cast<char*>(&a), sizeof(int));
		file.read(reinterpret_cast<char*>(&b), sizeof(int));
		file.read(reinterpret_cast<char*>(&c), sizeof(int));


		cout << "Tensor " << t + 1 << " Shape: [" << a << "][" << b << "][" << c << "]" << endl;
		cout << "Current Weight Pointer:" << curr_wptr << endl;

//		tensor_d[t].dim1 = a;
//		tensor_d[t].dim2 = b;
//		tensor_d[t].dim3 = c;
//		tensor_d[t].start_idx = curr_wptr;

		double temp;
		for(int k = 0;k < a*b*c ; k++){
			file.read(reinterpret_cast<char*>(&temp), sizeof(double));
			weight1d[curr_wptr + k] = tensor_t(temp);
		}
		curr_wptr+= a*b*c;
	}

	tensor_t out[ROOT_NODE_DIM];


	// std::ifstream ifile("/home/pgupta/new_hls/hls_tenet/datasets/set1/processed/iris_embedded.bin", std::ios::binary);
	std::ifstream ifile("/home/pgupta/new_hls/hls_tenet/datasets/set2/processed/x_train.bin", std::ios::binary);
	// std::ifstream ifile("/home/pgupta/new_hls/hls_tenet/datasets/set3/processed/hls-training_x_emb.bin", std::ios::binary);

	if(!ifile){
		cerr << "Error opening input data file: 16_feat_model data file" << endl;
		return 1;
	}

	cout<< "-----File opened successfully-----" << endl;

	for(int n = 0; n < NUM_SAMPLES ; n++){

//		hls::stream<tensor_t> sample[NUM_FEATURES][IN_MAP_DIM];

		hls::stream<ap_uint<BITS_PACKED>> sample_new;
		ap_uint<BITS_PACKED> packed = 0;

		cout<< "-----Reading Sample-----" << endl;
		int index=0;
		for (int j = 0; j< NUM_FEATURES; j ++){
			double temp;
			for (int k =0; k < IN_MAP_DIM; k++){
				ifile.read(reinterpret_cast<char*>(&temp), sizeof(double));
				tensor_t raw_bits =  tensor_t(temp);
//				std::cout << "OG Val:" << std::setprecision(16) << raw_bits << std::endl;
				packed.range(index + PRECISION - 1, index) = raw_bits.range(PRECISION - 1,0);
				index+=PRECISION;
				//std::cout<< "Packed:" << packed << std::endl;
//				sample[j][k].write(tensor_t(temp));
			}
		}


//		tensor_t unpacked[4][2];
//
//		index = 0;
//		for (int i = 0; i < 4; ++i) {
//		    for (int j = 0; j < 2; ++j) {
//		        ap_uint<16> raw_bits = packed.range(index + 15, index);
//		        unpacked[i][j].range(15, 0) = raw_bits;  // assign bits directly
//		        std::cout <<"Unpacked:" << unpacked[i][j] << std::endl;
//		        index += 16;
//		    }
//		}



		sample_new.write(packed);

		cout<< "-----Sample Read-----" << endl;
//		tensor_d, weight1d,
//		run_tenet(weight1d, sample_new, output);
		run_tenet( sample_new, output);

//		result_file << "Sample" << n << endl;

		for(int k = 0; k < ROOT_NODE_DIM; ++k) { // 32+ 40 + 20 + 5

			if(!output[k].empty()){
				out[k] = output[k].read();
				std::cout << "result[" << k << "] = " << out[k] << std::endl;
				// printf("Hex Lowercase: 0x%" PRIx16 "\n", out[k]);
				result_file <<  setprecision(PRECISION - 2) << out[k] << endl; /**" Label:" << round(double(out[k]))**/
			}
		}

	}


	cout<< "-----Closing Data File-----" << endl;

	ifile.close();
	result_file.close();

	return 0;

}

