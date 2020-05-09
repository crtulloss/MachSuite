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

void update_weights(TYPE weights1[W1_size],
                    TYPE weights2[W2_size],
                    TYPE d_weights1[W1_size],
                    TYPE d_weights2[W2_size],
                    TYPE biases1[B1_size],
                    TYPE biases2[B2_size],
                    TYPE d_biases1[B1_size],
                    TYPE d_biases2[B2_size],
                    bool flag[num_windows]) {
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

void backprop(TYPE weights1[W1_size], 
                TYPE weights2[W2_size],
                TYPE biases1[B1_size], 
                TYPE biases2[B2_size],
                TYPE training_data[tsamps_perbatch*num_windows*input_dimension],
                bool flag[num_windows]) {

    // forward and training structures
    TYPE activations1[layer1_dimension];
    TYPE activations2[output_dimension];
    TYPE dactivations1[layer1_dimension];
    TYPE dactivations2[output_dimension];
    TYPE net_outputs[output_dimension];

    // single-tsamp data for all electrodes - size e.g. 4 * 32 = 256
    uint32_t num_electrodes = num_windows*input_dimension;
    
    // FLATTEN THIS?
    TYPE elecdata[num_electrodes];

    // some offsets useful for indexing
    uint32_t samp_offset;
    uint32_t window_offset_weights2;
    uint32_t window_offset_weights1;
    uint32_t window_offset_output;
    uint32_t window_offset_layer1;
    uint32_t window_offset_input;

    // epoch accumulation variables for batched backprop
    TYPE dW2[W2_size];
    TYPE dW1[W1_size];
    TYPE dB2[B2_size];
    TYPE dB1[B1_size];

    // temporary variables to store some results
    // forward pass: activation of layer 1 and difference between out and in
    TYPE act1[num_windows*layer1_dimension];
    TYPE diff[num_windows*output_dimension];
    // backward pass: sample accum variable W2(x2-x0) used for backprop
    TYPE W2xdiff[num_windows*layer1_dimension];

    for (uint32_t epoch = 0; epoch < epochs_perbatch; epochs++) {
        
        // reset weight and bias delta accumulation variables
        // assumes W2_size = W1_size
        for (uint32_t i = 0; i < W2_size; i++) {
            dW1[i] = 0;
            dW2[i] = 0;
            if (i < B2_size) {
                dB2[i] = 0;
            }
            if (i < B1_size) {
                dB1[i] 0;
            }
        }

        for (uint32_t samp = 0; samp < tsamps_perbatch; samp++) {

            // offset to access input data for this time samp
            samp_offset = samp*num_electrodes;

            // access input data for all windows from PLM
            for (uint32_t elec = 0; elec < num_electrodes; elec++) {
                // this is a PLM access - can only UNROLL if has multiple ports
                elecdata[elec] = training_data[samp_offset+elec];
            }

            for (uint32_t window = 0; window < num_windows; window++) {
                //UNROLL? - if so, need to fix the offsets

                // compute some offsets for loop indexing
                window_offset_weights2 = window*output_dimension*layer1_dimension;
                window_offset_weights1 = window*layer1_dimension*input_dimension;
                window_offset_output = window*output_dimension;
                window_offset_layer1 = window*layer1_dimension;
                window_offset_input = window*input_dimension;

                if (flag[window]) {
		    
                    // forward pass
		            // compute layer1 activations
		            for (uint32_t neuron = 0; neuron < layer1_dimension; neuron++) {

                        // reset activation for this sample
                        act1[window_offset_layer1 + neuron] = 0;

                        // mac
                        for (uint32_t in = 0; in < input_dimension; in++) {
                            act1[window_offset_layer1 + neuron] +=
                                weights1[window_offset_weights1 + neuron*input_dimension + in] *
                                elecdata[window_offset_input + in];
                        }

                        // add bias
                        act1[window_ofset_layer1 + neuron] += bias1[window_offset_layer1 + neuron];
                    }

                    // compute output activations
                    for (uint32_t out = 0; out < output_dimension; out++) {

                        // reset output difference for this sample
                        diff[window_offset_output + out] = 0;

                        // mac
                        for (uint32_t neuron = 0; neuron < layer1_dimension; neuron++) {
                            diff[window_offset_output + out] +=
                                weights2[window_offset_weights2 + out*layer1_dimension + neuron] *
                                act1[window_offset_layer1 + neuron];
                        }

                        // add bias
                        diff[window_offset_output + out] += bias[window_offset_output + out];

                        // subtract the ground truth difference
                        // we don't need the output, only the difference
                        diff[window_offset_output + out] -= elecdata[window_offset_input + out];

                        // beginning of backprop for this sample
                        // this part only requires a loop over output
                        // epoch-accum dB2 - simple because we just add diff
                        dB2[window_offset_output + out] += diff[window_offset_output + out];

                    }

                    // backprop for this sample (with no weight update yet)
                    for (uint32_t neuron = 0; neuron < layer1_dimension; neuron++) {

                        // reset W2xdiff sample accum variable
                        W2xdiff[window_offset_layer1 + neuron] = 0;

                        // dual-purpose loop; both computations here looped over neurons and outputs
                        for (uint32_t out = 0; out < output_dimension; out++) {
                            // mac W2xdiff
                            W2xdiff[window_offset_layer1 + neuron] +=
                                weights2[window_offset_weights2 + out*layer1_dimension + neuron] *
                                diff[window_offset_output + out];

                            // epoch-accum dW2
                            dW2[window_offset_weights2 + out*layer1_dimension + neuron] +=
                                diff[window_offset_output + out] *
                                act1[window_offset_layer1 + neuron];
                        }

                        // these must be done after because they depend on W2xdiff

                        // epoch-accum dB1
                        dB1[window_offset_layer1 + neuron] += W2xdiff[window_offset_layer1 + neuron];

                        // epoch-accum dW1
                        for (uint32_t in = 0; in < input_dimension; in++) {
                            dW1[window_offset_weights1 + neuron*input_dimension + in] +=
                                W2xdiff[window_offset_layer1 + neuron] *
                                elecdata[window_offset_input + in];
                        }
                    }
                }
            }
        }

        // all samples have now been processed,
        // and we are ready to perform a weight update for this epoch
        for (uint32_t window = 0; window < num_windows; window++) {
            //UNROLL?

            // compute some offsets for loop indexing
            window_offset_weights2 = window*output_dimension*layer1_dimension;
            window_offset_weights1 = window*layer1_dimension*input_dimension;
            window_offset_output = window*output_dimension;
            window_offset_layer1 = window*layer1_dimension;
            window_offset_input = window*input_dimension;

            if (flag[window]) {

            }
        }
        // this epoch is now complete
    }

    // same training data is used for num_iters_perin
    for (i = 0; i < num_iters_perin; i++) {

        update_weights(weights1, weights2, delta_weights1, delta_weights2, 
                       biases1, biases2, oracle_activations1, output_difference);
    }
}
