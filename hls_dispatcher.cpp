
#include "./hls_dispatcher.h"


void dispatch_tensor_contraction(
    tensor_t *inputl,
    tensor_t *inputr,
    const Tensor3D &tensor_d1,
    const tensor_t *weight_arr,	
    tensor_t *output_t)
{

switch (tensor_d1.config_id) {


	case 0:
		tensor_contraction_new<10, 10, 5>(inputl, inputr, tensor_d1, weight_arr, output_t);
		break;

	case 1:
		tensor_contraction_new<10, 10, 10>(inputl, inputr, tensor_d1, weight_arr, output_t);
		break;

	case 2:
		tensor_contraction_new<4, 4, 10>(inputl, inputr, tensor_d1, weight_arr, output_t);
		break;

	case 3:
		tensor_contraction_new<2, 2, 4>(inputl, inputr, tensor_d1, weight_arr, output_t);
		break;

	default:
		// Optional: handle error
		break;

	}
}
