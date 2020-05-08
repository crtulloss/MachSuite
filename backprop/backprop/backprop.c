#include "backprop.h"

void RELU(TYPE activations[layer1_dimension], TYPE dactivations[layer1_dimension], int size) {
    int i;
    for( i = 0; i < size; i++) {
        dactivations[i] = activations[i]*(1.0-activations[i]);
        activations[i] = 1.0/(1.0+exp(-activations[i]));
    }
}

void matrix_vector_product_with_bias_input_layer(TYPE biases[layer1_dimension],
                                                 TYPE weights[input_dimension*layer1_dimension],
                                                 TYPE activations[layer1_dimension],
                                                 TYPE input_sample[input_dimension]){
    int i,j;
    for(j = 0; j < layer1_dimension; j++){
        activations[j] = (TYPE)0.0;
        for (i = 0; i < input_dimension; i++){
            activations[j] += weights[j*input_dimension + i] * input_sample[i];
        }
	activations[j] += biases[j];
    }
}

void matrix_vector_product_with_bias_output_layer(TYPE biases[output_dimension],
                                                 TYPE weights[layer1_dimension*output_dimension],
                                                 TYPE activations[output_dimension],
                                                 TYPE input_activations[layer1_dimension]){
    int i, j;
    for(j = 0; j < output_dimension; j++){
        activations[j] = (TYPE)0.0;
        for (i = 0; i < layer1_dimension; i++){
            activations[j] += weights[j*layer1_dimension + i] * input_activations[i];
        }
        activations[j] += biases[j];
    }
}

void take_difference(TYPE net_outputs[output_dimension], 
                     TYPE solutions[output_dimension], 
                     TYPE output_difference[output_dimension],
                     TYPE dactivations[output_dimension]) {
    int i;
    for( i = 0; i < output_dimension; i++){
        output_difference[i] = (((net_outputs[i]) - solutions[i]) * -1.0) * dactivations[i];
    }
}

void get_delta_matrix_weights3(TYPE delta_weights3[layer1_dimension*output_dimension],
                               TYPE output_difference[output_dimension],
                               TYPE last_activations[layer1_dimension]) {
    int i, j;
    for( i = 0; i < layer1_dimension; i++) {
        for( j = 0; j < output_dimension; j++) {
            delta_weights3[i*output_dimension + j] = last_activations[i] * output_difference[j];
        }
    }
}

void get_oracle_activations2(TYPE weights3[layer1_dimension*output_dimension], 
                             TYPE output_differences[output_dimension], 
                             TYPE oracle_activations[layer1_dimension],
                             TYPE dactivations[layer1_dimension]) {
    int i, j;
    for( i = 0; i < layer1_dimension; i++) {
        oracle_activations[i] = (TYPE)0.0;
        for( j = 0; j < output_dimension; j++) {
            oracle_activations[i] += output_differences[j] * weights3[i*output_dimension + j];
        }
        oracle_activations[i] = oracle_activations[i] * dactivations[i];
    }
}

void get_delta_matrix_weights2(TYPE delta_weights2[layer1_dimension*layer1_dimension],
                               TYPE output_difference[layer1_dimension],
                               TYPE last_activations[layer1_dimension]) {
    int i, j;
    for( i = 0; i < layer1_dimension; i++) {
        for( j = 0; j < layer1_dimension; j++) {
            delta_weights2[i*layer1_dimension + j] = last_activations[i] * output_difference[j];
        }
    }
}

void get_oracle_activations1(TYPE weights2[layer1_dimension*layer1_dimension], 
                             TYPE output_differences[layer1_dimension], 
                             TYPE oracle_activations[layer1_dimension],
                             TYPE dactivations[layer1_dimension]) {
    int i, j;
    for( i = 0; i < layer1_dimension; i++) {
        oracle_activations[i] = (TYPE)0.0;
        for( j = 0; j < layer1_dimension; j++) {
            oracle_activations[i] += output_differences[j] * weights2[i*layer1_dimension + j];
        }
        oracle_activations[i] = oracle_activations[i] * dactivations[i];
    }
}

void get_delta_matrix_weights1(TYPE delta_weights1[input_dimension*layer1_dimension],
                               TYPE output_difference[layer1_dimension],
                               TYPE last_activations[input_dimension]) {
    int i, j;
    for( i = 0; i < input_dimension; i++) {
        for( j = 0; j < layer1_dimension; j++) {
            delta_weights1[i*layer1_dimension + j] = last_activations[i] * output_difference[j];
        }
    }
}

void update_weights(TYPE weights1[input_dimension*layer1_dimension],
                    TYPE weights2[layer1_dimension*output_dimension],
                    TYPE d_weights1[input_dimension*layer1_dimension],
                    TYPE d_weights2[layer1_dimension*output_dimension],
                    TYPE biases1[layer1_dimension],
                    TYPE biases2[output_dimension],
                    TYPE d_biases1[layer1_dimension],
                    TYPE d_biases2[output_dimension]) {
    int i, j;
    TYPE norm, bias_norm;
    norm = 0.0;
    bias_norm = 0.0;

    for(i=0; i < input_dimension; i++){
        for(j = 0; j < layer1_dimension; j++){
            weights1[i*layer1_dimension + j] -= (d_weights1[i*layer1_dimension + j] * learning_rate);
            norm += weights1[i*layer1_dimension + j]*weights1[i*layer1_dimension + j];
        }
    }
    for(i=0; i < layer1_dimension; i++){
        biases1[i] -= (d_biases1[i]*learning_rate);
        bias_norm += biases1[i]*biases1[i];
    }
    
    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    for(i=0; i < input_dimension; i++){
        for(j = 0; j < layer1_dimension; j++){
            weights1[i*layer1_dimension + j] = (weights1[i*layer1_dimension + j]/norm);
        }
    }
    for(i=0; i < layer1_dimension; i++){
        biases1[i] = (biases1[i]/bias_norm);
    }

    norm = (TYPE)0.0;
    bias_norm = (TYPE)0.0;

    for(i=0; i < layer1_dimension; i++){
        for(j = 0; j < output_dimension; j++){
            weights2[i*output_dimension + j] -= (d_weights2[i*output_dimension + j] * learning_rate);
            norm += weights2[i*output_dimension + j]*weights2[i*output_dimension + j];
        }
    }
    for(i=0; i<output_dimension;i++){
        biases2[i] -= d_biases2[i]*learning_rate;
        bias_norm += biases2[i]*biases2[i];
    }

    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    for(i=0; i < layer1_dimension; i++){
        for(j = 0; j < output_dimension; j++){
            weights2[i*output_dimension + j] = (weights2[i*output_dimension + j]/norm);
        }
    }
    for(i=0; i < output_dimension; i++){
        biases2[i] = (biases2[i]/bias_norm);
    }
}

void backprop(TYPE weights1[num_windows*input_dimension*layer1_dimension], 
                TYPE weights2[layer1_dimension*output_dimension],
                TYPE biases1[layer1_dimension], 
                TYPE biases2[output_dimension],
                TYPE training_data[num_windows*tsamps_perbatch*input_dimension],
                bool flag[num_windows]) {
    int i,j;

    // forward and training structures
    TYPE activations1[layer1_dimension];
    TYPE activations2[output_dimension];
    TYPE dactivations1[layer1_dimension];
    TYPE dactivations2[output_dimension];
    TYPE net_outputs[output_dimension];

    // training structure
    TYPE output_difference[output_dimension];
    TYPE delta_weights1[input_dimension*layer1_dimension]; 
    TYPE delta_weights2[layer1_dimension*output_dimension];
    TYPE oracle_activations1[layer1_dimension];

    // single-tsamp data for all electrodes - size e.g. 4 * 32 = 256
    TYPE elecdata[input_dimension*num_windows];
    uint32_t num_electrodes = num_windows*input_dimension;
    uint32_t samp_offset;
    uint32_t window_offset;

    for (uint32_t epoch = 0; epoch < epochs_perbatch; epochs++) {
        
        // INSERT HERE set up accum variables for all windows
        
        for (uint32_t samp = 0; samp < tsamps_perbatch; samp++) {

            // offset to access input data for this time samp
            samp_offset = samp*num_electrodes;

            // access input data for all windows from PLM
            for (uint32_t elec = 0; elec < num_electrodes; elec++) {
                // this is a PLM access - can only UNROLL if has multiple ports
                elecdata[elec] = training_data[samp_offset+elec];
            }

            for (uint32_t window = 0; window < num_windows; window++) {
                //UNROLL?

                window_offset = window*input_dimension;

                if (flag[window]) {
                    // use activation variable that is the size of all neurons, all electrodes?
                    // so that it can be parallelized
                    // forward pass
                    // accum results
                }
            }
        }
        for (uint32_t window = 0; window < num_windows; window++) {
            //UNROLL?
            // post process accum
            // backprop
            // update weights
        }
    }

    // same training data is used for num_iters_perin
    for (i = 0; i < num_iters_perin; i++) {

        // reset activations at the beginning of each iteration
        for (j = 0; layer1_dimension; j++) {
            activations1[j] = (TYPE)0.0;
        }
        for (j = 0; j < output_dimension; j++) {
            activations2[j] = (TYPE)0.0;
        }

        matrix_vector_product_with_bias_input_layer(biases1, weights1, activations1, training_data);
	if (do_relu) {
            RELU(activations1, dactivations1, layer1_dimension);
	}

        matrix_vector_product_with_bias_output_layer(biases2, weights2, activations2, activations1);
	if (do_relu) {
            RELU(activations2, dactivations2, output_dimension);
	}

        take_difference(activations2, training_data, output_difference, dactivations2);

        get_delta_matrix_weights3(delta_weights2, output_difference, activations1);
        get_oracle_activations2(weights2, output_difference, oracle_activations1, dactivations1);
        
        get_delta_matrix_weights1(delta_weights1, oracle_activations1, training_data);

        update_weights(weights1, weights2, delta_weights1, delta_weights2, 
                       biases1, biases2, oracle_activations1, output_difference);
    }
}
