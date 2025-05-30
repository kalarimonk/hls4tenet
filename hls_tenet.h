#ifndef HLS_TENET_T
#define HLS_TENET_T

#include "./hls_tensor.h"

#include <iostream>
#include <hls_math.h>
#include <hls_stream.h>

#define clm(x) ( 1 << x )
#define PRECISION 16

void run_tenet( hls::stream<ap_uint<BITS_PACKED>>& features, hls::stream<tensor_t> predictions[ROOT_NODE_DIM]);

#endif
