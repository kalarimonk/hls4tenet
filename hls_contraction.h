#ifndef HLS_CONTRACTION_H
#define HLS_CONTRACTION_H

#include "./hls_tensor.h"

template <int diml, int dimr, int out_dim>
void tensor_contraction_new(tensor_t *inputl, tensor_t *inputr, const Tensor3D tensor_d,const tensor_t *weight_arr, tensor_t *output_t){

	int start_bias = tensor_d.start_idx;

	tensor_t accum[diml*dimr*out_dim];


	for(int k = 0; k < out_dim; k++){
		accum[k] = 0;
		for(int i = 0; i < diml; i++){
			for(int j = 0;j < dimr; j++){
				accum[k] += tensor_t(inputl[i] * inputr[j]) * weight_arr[start_bias + (i * dimr * out_dim + j * out_dim + k)];
			}
		}
	}



	for (int k = 0; k < out_dim; k++) {
		output_t[k] = accum[k];
	}

}


#endif
