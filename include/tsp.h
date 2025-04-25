#pragma once
#include <cuda_runtime.h>

struct TspInput {
    unsigned int dimension;
    float *distances;
};

struct TspResult {
    unsigned int dimension;
    float cost;
    unsigned int *tour;
};

#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                               \
    if(e!=cudaSuccess) {                                            \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(e));                             \
        exit(1);                                                    \
    }                                                               \
}
