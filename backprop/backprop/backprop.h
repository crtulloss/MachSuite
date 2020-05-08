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
#define input_dimension (window_size * num_tsamps)
#define output_dimension input_dimension
#define layer1_dimension neurons_perwin
#define training_sets num_batches

// used in generate and local_support for testing
#define max 1.0
#define offset 0.5

//Data Bounds
#define TYPE double

void backprop(
    TYPE weights1[input_dimension*layer1_dimension],
    TYPE weights2[layer1_dimension*output_dimension],
    TYPE biases1[layer1_dimension],
    TYPE biases2[output_dimension],
    TYPE training_data[num_iters_perin*input_dimension]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t { 
    TYPE weights1[input_dimension*layer1_dimension];
    TYPE weights2[layer1_dimension*output_dimension];
    TYPE biases1[layer1_dimension];
    TYPE biases2[output_dimension];
    TYPE training_data[input_dimension];
};
