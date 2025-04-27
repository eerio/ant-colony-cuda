#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define MAX_BLOCKS 1024
#define MAX_TPB 1024

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


void initializeRandStates(curandState* rand_states, int num_ants, unsigned long seed);

void depositPheromone(
    float* d_pheromone,
    const int* d_ant_tours,
    const float* d_tour_lengths,
    int num_ants,
    int num_cities
);

void evaporatePheromone(float* d_pheromone, float evaporation_rate, int num_cities);

void verifyToursHost(const int* h_ant_tours, int num_ants, int num_cities);

void computeTourLengths(
    const int* d_ant_tours,
    const float* d_distances,
    float* d_tour_lengths,
    int num_ants,
    int num_cities
);

void initializePheromones(float* d_pheromone, int num_cities);

void computeChoiceInfo(
    float* d_choice_info,
    const float* d_pheromone,
    const float* d_distances,
    int num_cities,
    float alpha,
    float beta
);