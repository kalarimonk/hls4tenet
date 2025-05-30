#define AP_INT_MAX_W 2048

#include <ap_int.h>
#include <ap_fixed.h>
#include "./hls_tensor.h"
#include "./hls_contraction.h"
#include "./hls_tenet.h"
#include "./hls_weights.h"
#include "./hls_dispatcher.h"
#include <inttypes.h>

using namespace std;

void run_tenet( hls::stream<ap_uint<BITS_PACKED>>& features, hls::stream<tensor_t> predictions[ROOT_NODE_DIM])
{

#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS PIPELINE II=1

#pragma HLS ARRAY_PARTITION variable=predictions complete dim=0
#pragma HLS ARRAY_PARTITION variable=w1d complete dim=0

#pragma HLS INTERFACE axis port = features
#pragma HLS INTERFACE axis port = predictions

//#pragma HLS INTERFACE s_axilite port = w1d

	tensor_t in_data[NUM_FEATURES][MAX_BOND_DIM];

#pragma HLS ARRAY_PARTITION variable=in_data complete dim=0

	int height =  HEIGHT;
	int curr_layer_nodes =  BOTTOM_NODES;

	tensor_t curr_layer_outputs[NUM_FEATURES][MAX_BOND_DIM];

#pragma HLS ARRAY_PARTITION variable=curr_layer_outputs complete dim=0

	bool not_empty_features = 1;
	ap_uint<BITS_PACKED> packed_feature;


	chek_empty: not_empty_features &= (!features.empty());

	if (not_empty_features){
		packed_feature = features.read();
		int index = 0;
		for (int i = 0; i < NUM_FEATURES; ++i) {
			for (int j = 0; j < IN_MAP_DIM; ++j) {
				ap_uint<PRECISION> raw_bits = packed_feature.range(index + PRECISION - 1, index);
				in_data[i][j].range(PRECISION - 1, 0) = raw_bits;  // assign bits directly
				index += PRECISION;
			}
		}
	}


	if(not_empty_features) {
	int p = 0;
	tree_loop: for (int level = height -1 ; level >= 0; level--){

		curr_layer_nodes = (1 << level);

		int start = (1 << level) - 1;
		int end = std::min(NUM_NODES, (1 << (level+1)) - 1);
		int t = start;
		int outt_dim = tensor_d[t].dim3;

		layer_loop: for(int j =  0; j < curr_layer_nodes; j++ ){
			if (t<end){
				dispatch_tensor_contraction(in_data[j*2], in_data[j*2 + 1], tensor_d[t], w1d, curr_layer_outputs[j]);
				t++;
			}
		}


		set_new_inputsl1: for (int k = 0; k < curr_layer_nodes; k++){
			set_new_inputsl2: for (int l = 0; l < outt_dim; l++){
								in_data[k][l] = curr_layer_outputs[k][l];
							}
		}

	}

	write_predictions: for (int p = 0; p < ROOT_NODE_DIM; p++){
		predictions[p].write(curr_layer_outputs[0][p]);
	}

	}

}



