#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../common/support.h"

// mindfuzz parameters - default values
#define num_electrodes 1024
#define num_tsamps 64
#define window_size 4
#define window_skip 2
#define num_neurons 8
#define total_layers 2
// FIX THIS
#define num_iters 512
#define num_iters_perin 1

// derived parameters - NN structure from mindfuzz params
#define input_dimension (window_size * num_tsamps)
#define output_dimension input_dimension
#define layer1_dimension num_neurons

// leftover MachSuite parameters - training hyperparams
#define training_sets   163
#define test_sets        15
#define learning_rate  0.01
#define epochs            1
#define norm_param    0.005

#define max 1.0
#define offset 0.5

//Data Bounds
#define TYPE double
#define MAX 1000
#define MIN 1

void backprop(
    TYPE weights1[input_dimension*layer1_dimension],
    TYPE weights2[layer1_dimension*layer1_dimension],
    TYPE weights3[layer1_dimension*output_dimension],
    TYPE biases1[layer1_dimension],
    TYPE biases2[layer1_dimension],
    TYPE biases3[output_dimension],
    TYPE training_data[training_sets*input_dimension],
    TYPE training_targets[training_sets*output_dimension]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
    TYPE weights1[input_dimension*layer1_dimension];
    TYPE weights2[layer1_dimension*layer1_dimension];
    TYPE weights3[layer1_dimension*output_dimension];
    TYPE biases1[layer1_dimension];
    TYPE biases2[layer1_dimension];
    TYPE biases3[output_dimension];
    TYPE training_data[training_sets*input_dimension];
    TYPE training_targets[training_sets*output_dimension];
};
