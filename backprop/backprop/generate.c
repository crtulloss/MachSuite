#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include "../../common/support.h"

//#include "sol.h"
//#include "train.h"
#include "backprop.h"

int main( int argc, const char* argv[] ){
    int i, j, fd;
    
    struct bench_args_t data;
    struct prng_rand_t state;

    prng_srand(1, &state);
    for( i = 0; i < input_dimension; i++){
        for( j = 0; j < layer1_dimension; j++){
            data.weights1[i*layer1_dimension + j] = (((TYPE)prng_rand(&state)/((TYPE)(PRNG_RAND_MAX))) * max) - offset;
        }
    }
    for( i = 0; i < layer1_dimension; i++){
        data.biases1[i] = (((TYPE)prng_rand(&state)/((TYPE)(PRNG_RAND_MAX))) * max) - offset;
        data.biases2[i] = (((TYPE)prng_rand(&state)/((TYPE)(PRNG_RAND_MAX))) * max) - offset;
        for( j = 0; j < layer1_dimension; j++){
            data.weights2[i*layer1_dimension + j] = (((TYPE)prng_rand(&state)/((TYPE)(PRNG_RAND_MAX))) * max) - offset;
        }
    }
    
    for( i = 0; i < training_sets; i++){
        for( j = 0; j < input_dimension; j++)
            data.training_data[i*input_dimension + j] = (TYPE)training_data[i][j];
        for( j = 0; j < output_dimension; j++)
            data.training_targets[i*output_dimension + j] = (TYPE)0;
        data.training_targets[i*output_dimension + (training_targets[i] - 1)] = 1.0;
    }

    fd = open("input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    assert( fd>0 && "Couldn't open input data file");

    data_to_input(fd, (void *)(&data));

    return 0;
}
