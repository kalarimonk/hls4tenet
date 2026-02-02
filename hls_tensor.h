#ifndef HLS_TENSOR_H
#define HLS_TENSOR_H

#include <ap_fixed.h>

#define NUM_FEATURES 16
#define IN_MAP_DIM 2

#define MAX_BOND_DIM 10

#define NUM_NODES 15
#define NUM_PARAM 3268
#define ROOT_NODE_DIM 5

#define HEIGHT 4
#define BOTTOM_NODES 8

#define BITS_PACKED 512

typedef ap_fixed<16,2> tensor_t;
// typedef unsigned long size_t;

/*
 * Structure to be used when scaling and dynamically introducing weights etc
 * dim1: Left Left Dimension,
 * dim2: Right Link Dimension,
 * dim3: Top Link Dimension (Internal Bond Dimension)
 * start_idx: Index to access the current weight tensor from the flattened 1D Weights Array
 */
struct Tensor3D{
    int dim1;
    int dim2;
    int dim3;
    int start_idx;
    int config_id;
};

#endif
