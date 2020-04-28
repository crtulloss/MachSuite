#include "backprop.h"

void soft_max(TYPE net_outputs[output_dimension], TYPE activations[output_dimension]) {
    int i;
    TYPE sum;
    sum = (TYPE) 0.0;

    for(i=0; i < output_dimension; i++) {
        sum += exp(-activations[i]);
    }
    for(i=0; i < output_dimension; i++) {
        net_outputs[i] = exp(-activations[i])/sum;
    }
}

void RELU(TYPE activations[layer1_dimension], TYPE dactivations[layer1_dimension], int size) {
    int i;
    for( i = 0; i < size; i++) {
        dactivations[i] = activations[i]*(1.0-activations[i]);
        activations[i] = 1.0/(1.0+exp(-activations[i]));
    }
}

void add_bias_to_activations(TYPE biases[layer1_dimension], 
                               TYPE activations[layer1_dimension],
                               int size) {
    int i;
    for( i = 0; i < size; i++){
        activations[i] = activations[i] + biases[i];
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
    }
    add_bias_to_activations(biases, activations, layer1_dimension);
}

void matrix_vector_product_with_bias_second_layer(TYPE biases[layer1_dimension],
                                                 TYPE weights[layer1_dimension*layer1_dimension],
                                                 TYPE activations[layer1_dimension],
                                                 TYPE input_activations[layer1_dimension]){
    int i,j;
    for (i = 0; i < layer1_dimension; i++){
        activations[i] = (TYPE)0.0;
        for(j = 0; j < layer1_dimension; j++){
            activations[i] += weights[i*layer1_dimension + j] * input_activations[j];
        }
    }
    add_bias_to_activations(biases, activations, layer1_dimension);
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
    }
    add_bias_to_activations(biases, activations, output_dimension);
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
                    TYPE weights2[layer1_dimension*layer1_dimension],
                    TYPE weights3[layer1_dimension*output_dimension],
                    TYPE d_weights1[input_dimension*layer1_dimension],
                    TYPE d_weights2[layer1_dimension*layer1_dimension],
                    TYPE d_weights3[layer1_dimension*output_dimension],
                    TYPE biases1[layer1_dimension],
                    TYPE biases2[layer1_dimension],
                    TYPE biases3[output_dimension],
                    TYPE d_biases1[layer1_dimension],
                    TYPE d_biases2[layer1_dimension],
                    TYPE d_biases3[output_dimension]) {
    int i, j;
    double norm, bias_norm;
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

    norm = (double)0.0;
    bias_norm = (double)0.0;

    for(i=0; i < layer1_dimension; i++){
        for(j = 0; j < layer1_dimension; j++){
            weights2[i*layer1_dimension + j] -= (d_weights2[i*layer1_dimension + j] * learning_rate);
            norm += weights2[i*layer1_dimension + j]*weights2[i*layer1_dimension + j];
        }
    }
    for(i=0; i < layer1_dimension; i++){
        biases2[i] -= (d_biases2[i]*learning_rate);
        bias_norm += biases2[i]*biases2[i];
    }

    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    for(i=0; i < layer1_dimension; i++){
        for(j = 0; j < layer1_dimension; j++){
            weights2[i*layer1_dimension + j] = (weights2[i*layer1_dimension + j]/norm);
        }
    }
    for(i=0; i < layer1_dimension; i++){
        biases2[i] = (biases2[i]/bias_norm);
    }

    norm = (double)0.0;
    bias_norm = (double)0.0;

    for(i=0; i < layer1_dimension; i++){
        for(j = 0; j < output_dimension; j++){
            weights3[i*output_dimension + j] -= (d_weights3[i*output_dimension + j] * learning_rate);
            norm += weights3[i*output_dimension + j]*weights3[i*output_dimension + j];
        }
    }
    for(i=0; i<output_dimension;i++){
        biases3[i] -= d_biases3[i]*learning_rate;
        bias_norm += biases3[i]*biases3[i];
    }

    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    for(i=0; i < layer1_dimension; i++){
        for(j = 0; j < output_dimension; j++){
            weights3[i*output_dimension + j] = (weights3[i*output_dimension + j]/norm);
        }
    }
    for(i=0; i < output_dimension; i++){
        biases3[i] = (biases3[i]/bias_norm);
    }
}

void backprop(TYPE weights1[input_dimension*layer1_dimension], 
                TYPE weights2[layer1_dimension*layer1_dimension],
                TYPE weights3[layer1_dimension*output_dimension],
                TYPE biases1[layer1_dimension], 
                TYPE biases2[layer1_dimension],
                TYPE biases3[output_dimension],
                TYPE training_data[training_sets*input_dimension],
                TYPE training_targets[training_sets*output_dimension]) {
    int i,j;
    //Forward and training structures
    TYPE activations1[layer1_dimension];
    TYPE activations2[layer1_dimension];
    TYPE activations3[output_dimension];
    TYPE dactivations1[layer1_dimension];
    TYPE dactivations2[layer1_dimension];
    TYPE dactivations3[output_dimension];
    TYPE net_outputs[output_dimension];
    //Training structure
    TYPE output_difference[output_dimension];
    TYPE delta_weights1[input_dimension*layer1_dimension]; 
    TYPE delta_weights2[layer1_dimension*layer1_dimension];
    TYPE delta_weights3[layer1_dimension*output_dimension];
    TYPE oracle_activations1[layer1_dimension];
    TYPE oracle_activations2[layer1_dimension];

    for(i=0; i<training_sets; i++){
        for(j=0;j<layer1_dimension;j++){
            activations1[j] = (TYPE)0.0;
            activations2[j] = (TYPE)0.0;
            if(j<output_dimension){
                activations3[j] = (TYPE)0.0;
            }
        }
        matrix_vector_product_with_bias_input_layer(biases1, weights1, activations1, &training_data[i*input_dimension]);
        RELU(activations1, dactivations1, layer1_dimension);
        matrix_vector_product_with_bias_second_layer(biases2, weights2, activations2, activations1);
        RELU(activations2, dactivations2, layer1_dimension);
        matrix_vector_product_with_bias_output_layer(biases3, weights3, activations3, activations2);
        RELU(activations3, dactivations3, output_dimension);
        soft_max(net_outputs, activations3);
        take_difference(net_outputs, &training_targets[i*output_dimension], output_difference, dactivations3);
        get_delta_matrix_weights3(delta_weights3, output_difference, activations2);
        get_oracle_activations2(weights3, output_difference, oracle_activations2, dactivations2);
        get_delta_matrix_weights2(delta_weights2, oracle_activations2, activations1);
        get_oracle_activations1(weights2, oracle_activations2, oracle_activations1, dactivations1);
        get_delta_matrix_weights1(delta_weights1, oracle_activations1, &training_data[i*input_dimension]);
        update_weights(weights1, weights2, weights3, delta_weights1, delta_weights2, delta_weights3, 
                       biases1, biases2, biases3, oracle_activations1, oracle_activations2, output_difference);
    }
}
