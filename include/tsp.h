#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

inline void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

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


extern __device__ int get_idx(int num_ants);
extern __global__ void initialize_rand_states(curandState* rand_states, int num_ants, unsigned long seed);
extern __global__ void depositPheromoneKernel(
    float* d_pheromone,
    const int* d_ant_tours,
    const float* d_tour_lengths,
    int num_ants,
    int num_cities,
    float Q
);

extern __global__ void evaporatePheromoneKernel(float* d_pheromone, float evaporation_rate, int num_cities);
void verifyToursHost(const int* h_ant_tours, int num_ants, int num_cities);
extern __global__ void computeTourLengthsKernel(
    const int* d_ant_tours,
    const float* d_distances,
    float* d_tour_lengths,
    int num_ants,
    int num_cities
);

extern __global__ void computeChoiceInfoKernel(
    float* d_choice_info,
    const float* d_pheromone,
    const float* d_distances,
    int num_cities,
    float alpha,
    float beta
);