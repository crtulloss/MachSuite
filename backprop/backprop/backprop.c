#include "backprop.h"

// TODO figure out RELU implementation
/*
void RELU(TYPE activations[layer1_dimension], TYPE dactivations[layer1_dimension], int size) {
    int i;
    for( i = 0; i < size; i++) {
        dactivations[i] = activations[i]*(1.0-activations[i]);
        activations[i] = 1.0/(1.0+exp(-activations[i]));
    }
}
*/

void backprop(TYPE weights1[W1_size], 
                TYPE weights2[W2_size],
                TYPE biases1[B1_size], 
                TYPE biases2[B2_size],
                TYPE training_data[tsamps_perbatch*num_windows*input_dimension],
                bool flag[num_windows]) {

    // single-tsamp data for all electrodes - size e.g. 4 * 32 = 256
    uint32_t num_electrodes = num_windows*input_dimension;
    
    // TODO FLATTEN THIS?
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
            dW1[i] = 0.0.;
            dW2[i] = 0.0;
            if (i < B2_size) {
                dB2[i] = 0.0;
            }
            if (i < B1_size) {
                dB1[i] 0.0;
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
                // TODO UNROLL? - if so, need to fix the offsets

                if (flag[window]) {

                    // compute some offsets for loop indexing
                    window_offset_weights2 = window*output_dimension*layer1_dimension;
                    window_offset_weights1 = window*layer1_dimension*input_dimension;
                    window_offset_output = window*output_dimension;
                    window_offset_layer1 = window*layer1_dimension;
                    window_offset_input = window*input_dimension;

                    // forward pass
		            // compute layer1 activations
		            for (uint32_t neuron = 0; neuron < layer1_dimension; neuron++) {

                        // reset activation for this sample
                        act1[window_offset_layer1 + neuron] = 0.0;

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
                        diff[window_offset_output + out] = 0.0;

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
                        W2xdiff[window_offset_layer1 + neuron] = 0.0;

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
                // end of this window
            }
            // this sample is complete for this epoch
        }

        // all samples have now been processed,
        // and we are ready to perform a weight update for this epoch
        for (uint32_t window = 0; window < num_windows; window++) {
            // TODO UNROLL?

            if (flag[window]) {

                // compute some offsets for loop indexing
                window_offset_weights2 = window*output_dimension*layer1_dimension;
                window_offset_weights1 = window*layer1_dimension*input_dimension;
                window_offset_output = window*output_dimension;
                window_offset_layer1 = window*layer1_dimension;
                window_offset_input = window*input_dimension;

                // these normalizations only useful for this window
                TYPE norm, bias_norm;
                norm = 0.0;
                bias_norm = 0.0;

                for (uint32_t neuron = 0; neuron < layer1_dimension; neuron++) {
                    
                    // update B1
                    biases1[window_offset_layer1 + neuron] -=
                        (dB1[window_offset_layer1 + neuron] * learning_rate);

                    // add to bias normalization
                    bias_norm += biases1[window_offset_layer1 + neuron] *
                        biases1[window_offset_layer1 + neuron];

                    // update W1
                    for (uint32_t in = 0; in < input_dimension; in++) {

                        weights1[window_offset_weights1 + neuron*input_dimension + in] -=
                            (dW1[window_offset_weights1 + neuron*input_dimension + in] *
                             learning_rate);

                        // add to weight normalization
                        norm += weights1[window_offset_weights1 + neuron*input_dimension + in] *
                            weights1[window_offset_weights1 + neuron*input_dimension + in];
                    }
                }

                // TODO normalization is temporarily disallowed,
                // until we figure out how to do sqrt and
                // whether division is ok
                /*               
                norm = sqrt(norm);
                bias_norm = sqrt(bias_norm);

                // perform normalization
                for (uint32_t neuron = 0; neuron < layer1_dimension; neuron++) {

                    // bias normalization
                    biases1[window_offset_layer1 + neuron] =
                        (biases1[window_offset_layer1 + neuron] / bias_norm);
                    
                    // weight normalization
                    for (uint32_t in = 0; in < input_dimension; in++) {

                        weights1[window_offset_weights1 + neuron*input_dimension + in] =
                            (weights1[window_offset_weights1 + neuron*input_dimension + in] / norm);
                    }
                }

                norm = (TYPE)0.0;
                bias_norm = (TYPE)0.0;
                */

                for (uint32_t out = 0; out < output_dimension; out++) {
                    
                    // update B2
                    biases2[window_offset_output + out] -=
                        (dB2[window_offset_output + out] * learning_rate);

                    // add to bias normalization
                    bias_norm += biases2[window_offset_output + out] *
                        biases2[window_offset_output + out];

                    // update W2
                    for (uint32_t neuron = 0; neuron < layer1_dimension; neuron++) {

                        weights2[window_offset_weights2 + out*layer1_dimension + neuron] -=
                            (dW2[window_offset_weights2 + out*layer1_dimension + neuron] *
                             learning_rate);

                        // add to weight normalization
                        norm += weights2[window_offset_weights2 + out*layer1_dimension + neuron] *
                            weights2[window_offset_weights2 + out*layer1_dimension + neuron];
                    }
                }
                
                // TODO normalization is temporarily disallowed,
                // until we figure out how to do sqrt and
                // whether division is ok
                /*
                norm = sqrt(norm);
                bias_norm = sqrt(bias_norm);

                // perform normalization
                for (uint32_t out = 0; out < output_dimension; out++) {

                    // bias normalization
                    biases2[window_offset_output + out] =
                        (biases2[window_offset_output + out] / bias_norm);
                    
                    // weight normalization
                    for (uint32_t neuron = 0; neuron < layer1_dimension; neuron++) {

                        weights2[window_offset_weights2 + out*layer1_dimension + neuron] =
                            (weights2[window_offset_weights2 + out*layer1_dimension + neuron] / norm);
                    }
                }
                */
            }
            // this window is now complete
        }
        // this epoch is now complete
    }
    // all epochs complete
}
