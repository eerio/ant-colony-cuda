#include <cuda_runtime.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "tsp.h"

__global__ void computeChoiceInfoKernel(
    float* d_choice_info,
    const float* d_pheromone,
    const float* d_distances,
    int num_cities,
    float alpha,
    float beta
) {
    int total = num_cities * num_cities;
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / num_cities;
        int col = idx % num_cities;

        if (row == col) {
            d_choice_info[idx] = 0.0f;
        } else {
            float tau = d_pheromone[idx];
            float dist = d_distances[idx];
            float eta = (dist > 0.0f) ? 1.0f / dist : 0.0f;

            d_choice_info[idx] = __powf(tau, alpha) * __powf(eta, beta);
        }
    }
}

__global__ void printChoiceInfoKernel(float* d_choice_info, int num_cities) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("choice_info matrix:\n");
        for (int i = 0; i < num_cities; ++i) {
            for (int j = 0; j < num_cities; ++j) {
                printf("%6.3f ", d_choice_info[i * num_cities + j]);
            }
            printf("\n");
        }
    }
}

// Dummy kernels
__global__ void dummyTourConstructionKernel() {}
__global__ void dummyPheromoneUpdateKernel() {}

// Main ACO driver
TspResult solveTSPWorker(
    const TspInput& tsp_input,
    unsigned int num_iter,
    float alpha,
    float beta,
    float evaporate,
    unsigned int seed
) {
    int num_cities = tsp_input.dimension;
    size_t matrix_size = sizeof(float) * num_cities * num_cities;

    // Allocate memory
    float *d_choice_info, *d_pheromone, *d_distances;
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_distances, matrix_size);

    // Initialize distances from input
    cudaMemcpy(d_distances, tsp_input.distances, matrix_size, cudaMemcpyHostToDevice);

    // Initialize pheromones to 1.0
    float* h_initial_pheromone = new float[num_cities * num_cities];
    for (int i = 0; i < num_cities * num_cities; ++i) {
        h_initial_pheromone[i] = 1.0f;
    }
    cudaMemcpy(d_pheromone, h_initial_pheromone, matrix_size, cudaMemcpyHostToDevice);
    delete[] h_initial_pheromone;

#ifdef DEBUG
    // Use cudaMemset to zero out the choice_info array
    cudaMemset(d_choice_info, 0, matrix_size);
#endif

    // Loop
    // int threads_per_block = 256;
    int threads_per_block = 4;
    int total = num_cities * num_cities;
    // int blocks = (total + threads_per_block - 1) / threads_per_block;
    int blocks = 3;

    for (unsigned int iter = 0; iter < num_iter; ++iter) {
        computeChoiceInfoKernel<<<blocks, threads_per_block>>>(
            d_choice_info, d_pheromone, d_distances, num_cities, alpha, beta
        );
        cudaDeviceSynchronize();
#ifdef DEBUG
        // Copy the result back to host for validation
        float* h_choice_info = new float[num_cities * num_cities];
        cudaMemcpy(h_choice_info, d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
    
        // Check if any value in the array is zero
        bool found_zero = false;
        for (int i = 0; i < num_cities * num_cities; ++i) {
            if (i % num_cities != i / num_cities && h_choice_info[i] == 0.0f) {
                found_zero = true;
                printf("Error: d_choice_info[%d] is zero\n", i);
            }
        }
    
        if (!found_zero) {
            printf("All values in d_choice_info are non-zero.\n");
        }
    
        delete[] h_choice_info;
#endif
    

        dummyTourConstructionKernel<<<1, 1>>>();
        dummyPheromoneUpdateKernel<<<1, 1>>>();
        cudaDeviceSynchronize();

        if (iter == num_iter - 1) {
            printChoiceInfoKernel<<<1, 1>>>(d_choice_info, num_cities);
            cudaDeviceSynchronize();
        }        
    }

    // Return dummy result
    TspResult result;
    result.dimension = num_cities;
    result.cost = 0.0f;
    result.tour = new unsigned int[num_cities];
    for (int i = 0; i < num_cities; ++i) result.tour[i] = i;

    // Free GPU memory
    cudaFree(d_choice_info);
    cudaFree(d_pheromone);
    cudaFree(d_distances);

    return result;
}
