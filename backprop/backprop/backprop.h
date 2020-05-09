#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../common/support.h"

// mindfuzz parameters - default values
#define num_windows 32
#define window_size 8
#define neurons_perwin 4
#define tsamps_perbatch 32
#define batches_perindata 8
#define epochs_perbatch 32
#define num_batches 512
#define learning_rate  0.01
#define do_relu 0

// derived parameters - NN structure from mindfuzz params
#define input_dimension window_size
#define output_dimension input_dimension
#define layer1_dimension neurons_perwin
#define training_sets num_batches

// some useful sizes
#define W2_size = num_windows*output_dimension*layer1_dimension
#define W1_size = num_windows*layer1_dimension*input_dimension
#define B2_size = num_windows*output_dimension
#define B1_size = num_windows*layer1_dimension

// used in generate and local_support for testing
#define max 1.0
#define offset 0.5

//Data Bounds
#define TYPE double

void backprop(TYPE weights1[W1_size], 
                TYPE weights2[W2_size],
                TYPE biases1[B1_size], 
                TYPE biases2[B2_size],
                TYPE training_data[tsamps_perbatch*num_windows*input_dimension],
                bool flag[num_windows]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t { 
    TYPE weights1[W1_size];
    TYPE weights2[W2_size];
    TYPE biases1[B1_size];
    TYPE biases2[B2_size];
    TYPE training_data[tsamps_perbatch*num_windows*input_dimension];
    bool flag[num_windows];
};
