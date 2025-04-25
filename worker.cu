#include "tsp.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h> // For printf (debugging only!)

// CUDA kernel to perform tour construction for a single ant
__global__ void tspWorkerKernel(
    const TspInput &tsp_input,
    unsigned int num_cities,
    float alpha,
    float beta,
    curandState_t *rand_states,
    float *d_choice_info,
    int *d_tabu_list,
    unsigned int *d_tour,
    unsigned int current_step
) {
    unsigned int ant_id = blockIdx.x;
    unsigned int city_id = threadIdx.x;

    extern __shared__ float s_choice_info[];
    extern __shared__ int   s_tabu_list[];
    extern __shared__ float s_selection_probs[];

    // Calculate offset for this ant's data
    unsigned int ant_offset = ant_id * num_cities;
    unsigned int current_city = (current_step > 0) ? d_tour[ant_offset + current_step - 1] : ant_id;

    // Load data to shared memory
    s_choice_info[city_id] = d_choice_info[current_city * num_cities + city_id];
    s_tabu_list[city_id]   = d_tabu_list[ant_offset + city_id];

    __syncthreads();

    // Calculate selection probabilities
    float sum_probs = 0.0f;
    if (s_tabu_list[city_id] == 0) {
        s_selection_probs[city_id] = 0.0f;
    } else {
        s_selection_probs[city_id] = s_choice_info[city_id];
        sum_probs += s_selection_probs[city_id];
    }

    __syncthreads();

    // Perform Roulette Wheel Selection
    if (city_id == 0) {
        // float random_val = curand_uniform(&rand_states[blockIdx.x * blockDim.x + threadIdx.x]);
        float random_val = curand_uniform(&rand_states[ant_id]);
        // __global__ void generate_random(float* result, curandState* states) {
        //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        //     result[idx] = curand_uniform(&states[idx]);  // generate float in (0, 1]
        // }
        // generate_random(&random_val, rand_states);
        float roulette_spin = random_val * sum_probs;

        float current_prob_sum = 0.0f;
        unsigned int next_city = (unsigned int)-1;
        for (int i = 0; i < num_cities; ++i) {
            if (s_selection_probs[i] != 0.0f) {
                current_prob_sum += s_selection_probs[i];
                if (current_prob_sum >= roulette_spin) {
                    next_city = i;
                    break;
                }
            }
        }
        d_tour[ant_offset + current_step] = next_city;

        // Debugging: Print tour (SLOW! REMOVE LATER)
        //printf("Ant %d, Step %d, Next City: %d\n", ant_id, current_step, next_city);
    }
    __syncthreads();
}

// CUDA kernel to precompute choice info
__global__ void precomputeChoiceInfoKernel(const float *d_distances, float *d_choice_info, int num_cities, float alpha, float beta) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < num_cities && col < num_cities) {
        if (row != col) {
            float pheromone = 1.0f;
            float heuristic = 1.0f / d_distances[row * num_cities + col];
            d_choice_info[row * num_cities + col] = powf(pheromone, alpha) * powf(heuristic, beta);
        } else {
            d_choice_info[row * num_cities + col] = 0.0f;
        }
    }
}

// CUDA kernel to print a matrix (debugging)
__global__ void printMatrixKernel(const float *matrix, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        printf("Matrix[%d][%d] = %f\n", row, col, matrix[row * cols + col]);
    }
}

__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Seed is the random seed, idx is like "subseed"
    // Each thread is thus initialized with different value and has its own currandState variable.
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void generate_random(float* result, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    result[idx] = curand_uniform(&states[idx]);  // generate float in (0, 1]
}

// Wrapper function
TspResult solveTSPWorker(
    const TspInput &tsp_input,
    unsigned int num_iter,
    float alpha,
    float beta,
    float evaporate,
    unsigned int seed
) {
    int num_cities = tsp_input.dimension;
    int num_ants = num_cities;

    // Allocate memory on the GPU
    float *d_choice_info;
    int *d_tabu_list;
    unsigned int *d_tour;
    curandState_t *d_rand_states;

    cudaMalloc(&d_choice_info, sizeof(float) * num_cities * num_cities);
    cudaMalloc(&d_tabu_list, sizeof(int) * num_ants * num_cities);
    cudaMalloc(&d_tour, sizeof(unsigned int) * num_ants * num_cities);
    cudaMalloc(&d_rand_states, sizeof(curandState_t) * num_ants);

    // Copy input data to GPU
    float *d_distances;
    cudaMalloc(&d_distances, sizeof(float) * num_cities * num_cities);
    cudaMemcpy(d_distances, tsp_input.distances, sizeof(float) * num_cities * num_cities, cudaMemcpyHostToDevice);

    // Debugging: Print distance matrix
    //printMatrixKernel<<<num_cities, num_cities>>>(d_distances, num_cities, num_cities);
    //cudaDeviceSynchronize();

    // Precompute choice_info on the GPU
    precomputeChoiceInfoKernel<<<num_cities, num_cities>>>(d_distances, d_choice_info, num_cities, alpha, beta);

    // Debugging: Print choice info
    //printMatrixKernel<<<num_cities, num_cities>>>(d_choice_info, num_cities, num_cities);
    //cudaDeviceSynchronize();

    // Initialize tabu list
    int *h_tabu_list = new int[num_ants * num_cities];
    for (int i = 0; i < num_ants * num_cities; ++i) {
        h_tabu_list[i] = 1;
    }
    cudaMemcpy(d_tabu_list, h_tabu_list, sizeof(int) * num_ants * num_cities, cudaMemcpyHostToDevice);
    delete[] h_tabu_list;

    // Initialize the first city for each ant's tour
    unsigned int *h_tour_starts = new unsigned int[num_ants];
    for (int i = 0; i < num_ants; ++i) {
        h_tour_starts[i] = i % num_cities;
    }
    cudaMemcpy(d_tour, h_tour_starts, sizeof(unsigned int) * num_ants, cudaMemcpyHostToDevice);
    delete[] h_tour_starts;

    // Initialize random number states
    init_rng<<<num_ants, 1>>>(d_rand_states, seed);
    cudaDeviceSynchronize();

    TspResult result;
    result.dimension = num_cities;
    result.tour = new unsigned int[num_ants * num_cities];
    result.cost = 0.0f;

    for (unsigned int i = 1; i < num_iter + 1; ++i) {
        // Launch the kernel
        tspWorkerKernel<<<num_ants, num_cities, 3 * num_cities * sizeof(float) + num_cities * sizeof(int)>>>(
            tsp_input,
            num_cities,
            alpha,
            beta,
            d_rand_states,
            d_choice_info,
            d_tabu_list,
            d_tour,
            i
        );

        cudaDeviceSynchronize();

        // Update tabu lists on the host and copy back to device
        int *h_tabu_list_update = new int[num_ants * num_cities];
        cudaMemcpy(h_tabu_list_update, d_tabu_list, sizeof(int) * num_ants * num_cities, cudaMemcpyDeviceToHost);
        unsigned int *h_tour = new unsigned int[num_ants * num_cities];
        cudaMemcpy(h_tour, d_tour, sizeof(unsigned int) * num_ants * num_cities, cudaMemcpyDeviceToHost);

        for (int ant = 0; ant < num_ants; ant++) {
            h_tabu_list_update[ant * num_cities + h_tour[ant * num_cities + i]] = 0;
        }
        cudaMemcpy(d_tabu_list, h_tabu_list_update, sizeof(int) * num_ants * num_cities, cudaMemcpyHostToDevice);
        delete[] h_tabu_list_update;
        delete[] h_tour;
    }

    cudaMemcpy(result.tour, d_tour, sizeof(unsigned int) * num_ants * num_cities, cudaMemcpyDeviceToHost);

    // Calculate the cost of the best tour (simplified - needs optimization)
    float min_cost = FLT_MAX;
    unsigned int min_cost_ant = 0;
    for (int ant = 0; ant < num_ants; ant++) {
        float current_cost = 0.0f;
        for (int i = 0; i < num_cities - 1; ++i) {
            current_cost += tsp_input.distances[result.tour[ant * num_cities + i] * num_cities + result.tour[ant * num_cities + i + 1]];
        }
        current_cost += tsp_input.distances[result.tour[ant * num_cities + num_cities - 1] * num_cities + result.tour[ant * num_cities]];

        // Debugging: Print current cost
        //printf("Ant %d, Cost: %f\n", ant, current_cost);

        if (current_cost < min_cost) {
            min_cost = current_cost;
            min_cost_ant = ant;
        }
    }
    result.cost = min_cost;
    result.tour = new unsigned int[num_cities];
    cudaMemcpy(result.tour, d_tour + min_cost_ant * num_cities, sizeof(unsigned int) * num_cities, cudaMemcpyDeviceToHost);

    // Debugging: Print final tour
    //printf("Best Ant: %d, Best Cost: %f, Tour: ", min_cost_ant, min_cost);
    //for (int i = 0; i < num_cities; ++i) {
    //    printf("%d ", result.tour[i]);
    //}
    //printf("\n");

    // Free GPU memory
    cudaFree(d_distances);
    cudaFree(d_choice_info);
    cudaFree(d_tabu_list);
    cudaFree(d_tour);
    cudaFree(d_rand_states);

    return result;
}