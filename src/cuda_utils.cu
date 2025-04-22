// cuda_utils.cu
#include "cuda_utils.h"

// Helper function to perform atomic min on float and track the index
__device__ void atomicMinf(float* address, float val, int* index_addr, int index) {
    if (*address > val) {
        atomicExch(address, val);
        atomicExch(index_addr, index);
    }
}