#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <vector>
#include <numeric>
#include <float.h>
#include <algorithm>
#include <cmath>
#include "tsp.h"
#include <cub/cub.cuh>

#define MAX_CITIES 1024
#define N_THREADS 1024

using BlockLoadT = cub::BlockLoad<float, N_THREADS, (N_THREADS + MAX_CITIES - 1) / N_THREADS>;
 
using BlockScanT = cub::BlockScan<float, N_THREADS>;
 

using BlockReduceT = cub::BlockReduce<float, N_THREADS>;
 
typedef union {
 
    typename BlockLoadT::TempStorage load;
 
    typename BlockScanT::TempStorage scan;
 

    typename BlockReduceT::TempStorage reduce;
 
} myTempStorageT;

struct Shared {
    int tabu_list[MAX_CITIES];
    float row_choice_info[MAX_CITIES];
    myTempStorageT cubStorage; 
    float selection_probs[MAX_CITIES + 1]; // + 1 for exclusive scan
    float total_sum;
    int selected_city;
};

__global__ void tourConstructionKernelQueen(
    int *tours,
    const float *choice_info,
    int num_ants,
    int num_cities,
    curandState *states
) {
    extern __shared__ Shared shared_data[];
    float *selection_probs = shared_data->selection_probs;
    int *tabu_list = shared_data->tabu_list;
    float *total_sum = &shared_data->total_sum;
    int *selected_city = &shared_data->selected_city;
    float *row_choice_info = shared_data->row_choice_info;
    myTempStorageT *cubStorage = &shared_data->cubStorage;

    int queen_id = blockIdx.x; // Each block handles one queen ant
    int worker_id = threadIdx.x; // Each thread in block is a worker (city checker)

    if (queen_id >= num_ants) return;

    curandState local_state = states[queen_id]; // Each queen uses its own RNG state

    // Initialize tabu list
    for (int city = worker_id; city < num_cities; city += blockDim.x) {
        tabu_list[city] = false;
    }
    __syncthreads();

    // Randomly select initial city
    int start_city = curand(&local_state) % num_cities;
    if (worker_id == 0) {
        tours[queen_id * num_cities + 0] = start_city;
        tabu_list[start_city] = true;
    }
    __syncthreads();

    for (int step = 1; step < num_cities; step++) {
        int current_city = tours[queen_id * num_cities + step - 1];

        for (int j=worker_id; j < num_cities; j += blockDim.x) {
            row_choice_info[j] = choice_info[current_city * num_cities + j];
        }
        __syncthreads();

        float threadData[1];
        if (worker_id < num_cities && !tabu_list[worker_id]) {
            threadData[0] = row_choice_info[worker_id];
        } else {
            threadData[0] = 0;
        }
        BlockScanT(cubStorage->scan).InclusiveSum(threadData, threadData);
        if (worker_id < num_cities) {
            selection_probs[worker_id] = threadData[0];
        }
        __syncthreads();
        if (worker_id == 0 ){
            *total_sum = selection_probs[num_cities - 1];
        }
        __syncthreads();

        // Draw random value for roulette selection
        float rand_val = curand_uniform(&local_state) * *total_sum - FLT_MIN;
        assert(selection_probs[num_cities - 1] == *total_sum);

        // Perform Roulette Wheel-style selection
        if (worker_id == 0) {
            *selected_city = MAX_CITIES + 1;
        }
        __syncthreads();
        float tol = 1e-8;
        if (
            (worker_id == 0 && selection_probs[0] >= rand_val)
            || (worker_id < num_cities && selection_probs[worker_id] + tol >= rand_val && selection_probs[worker_id - 1] < rand_val)
         ) {
            // here weird stuff can happen; because of floats, node can be tabued
            // or it can have chance 0 etc.
            if (!tabu_list[worker_id]) {
                // *selected_city = worker_id;
                // this should be accessed by at most a few threads (depending on tol)
                atomicMin(selected_city, worker_id);
            }
        }
        __syncthreads();
        if (worker_id == 0 && *selected_city == MAX_CITIES + 1) {
            // shouldn't happen often
            for (int city=0; city < num_cities; ++city) {
                if (!tabu_list[city]) {
                    *selected_city = city;
                    break;
                }
            }
        }
        __syncthreads();

        assert(*selected_city >= 0);
        assert(*selected_city < num_cities);
        assert(!tabu_list[*selected_city]);

        if (worker_id == 0) {
            tours[queen_id * num_cities + step] = *selected_city;
            tabu_list[*selected_city] = true;
        }
        __syncthreads();
    }

    // Save RNG state
    if (worker_id == 0) {
        states[queen_id] = local_state;
    }
}

TspResult solveTSPQueen(
    const TspInput& tsp_input,
    unsigned int num_iter,
    float alpha,
    float beta,
    float evaporate,
    unsigned int seed
) {
    int num_cities = tsp_input.dimension;
    int num_queens = num_cities;
    int num_workers = MAX_CITIES;
    
    int num_blocks = num_queens; // number of Queens
    int threads_per_block = num_workers; // number of worker ants
    int shared_memory_size = sizeof(Shared);
    printf("Config: num_blocks: %d, tpb: %d, shmem: %d\n", num_blocks, threads_per_block, shared_memory_size);

    size_t matrix_size = sizeof(float) * num_cities * num_cities;
    int* d_ant_tours;
    curandState* d_rand_states;
    float* d_tour_lengths;
    float* d_choice_info;
    float* d_distances;
    float* d_pheromone;
    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_ant_tours, sizeof(int) * num_queens * num_cities);
    cudaMalloc(&d_rand_states, sizeof(curandState) * num_queens);
    cudaMalloc(&d_tour_lengths, sizeof(float) * num_queens);
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    cudaMemcpy(d_distances, tsp_input.distances, matrix_size, cudaMemcpyHostToDevice);

    initializeRandStates(d_rand_states, num_queens, seed);
    initializePheromones(d_pheromone, num_cities);
    
    std::vector<float> iteration_times_ms;
    cudaEvent_t iter_start, iter_end;
    HANDLE_ERROR(cudaEventCreate(&iter_start));
    HANDLE_ERROR(cudaEventCreate(&iter_end));
    for (unsigned int iter = 0; iter < num_iter; ++iter) {
        HANDLE_ERROR(cudaEventRecord(iter_start));
        computeChoiceInfo(d_choice_info, d_pheromone, d_distances, num_cities, alpha, beta);

        tourConstructionKernelQueen<<<num_blocks, threads_per_block, shared_memory_size>>>(
            d_ant_tours, d_choice_info, num_queens, num_cities, d_rand_states
        );
        cudaDeviceSynchronize();

        evaporatePheromone(d_pheromone, evaporate, num_cities);

        computeTourLengths(
            d_ant_tours, d_distances, d_tour_lengths, num_queens, num_cities
        );

        depositPheromone(
            d_pheromone, d_ant_tours, d_tour_lengths, num_queens, num_cities
        );

        HANDLE_ERROR(cudaEventRecord(iter_end));
        HANDLE_ERROR(cudaEventSynchronize(iter_end));
        float elapsed_ms = 0.0f;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsed_ms, iter_start, iter_end));
        iteration_times_ms.push_back(elapsed_ms);
    }
    HANDLE_ERROR(cudaEventDestroy(iter_start));
    HANDLE_ERROR(cudaEventDestroy(iter_end));

    float* h_tour_lengths = new float[num_queens];
    cudaMemcpy(h_tour_lengths, d_tour_lengths, sizeof(float) * num_queens, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float best_length = FLT_MAX;
    int best_idx = -1;
    for (int i = 0; i < num_queens; ++i) {
        if (h_tour_lengths[i] < best_length) {
            best_length = h_tour_lengths[i];
            best_idx = i;
        }
    }
    delete[] h_tour_lengths;

    if (best_idx == -1) {
        printf("Error: No valid ant found!\n");
        return {};
    }

    unsigned int* h_best_tour = new unsigned int[num_cities];
    cudaMemcpy(h_best_tour, &d_ant_tours[best_idx * num_cities], sizeof(unsigned int) * num_cities, cudaMemcpyDeviceToHost);

    TspResult result;
    result.dimension = num_cities;
    result.cost = best_length;
    result.tour = h_best_tour;

    cudaFree(d_ant_tours);
    cudaFree(d_pheromone);
    cudaFree(d_rand_states);
    cudaFree(d_tour_lengths);
    cudaFree(d_choice_info);
    cudaFree(d_distances);

    float sum = std::accumulate(iteration_times_ms.begin(), iteration_times_ms.end(), 0.0f);
    float mean = sum / iteration_times_ms.size();

    float sq_sum = std::inner_product(
        iteration_times_ms.begin(), iteration_times_ms.end(),
        iteration_times_ms.begin(), 0.0f
    );
    float stddev = std::sqrt(sq_sum / iteration_times_ms.size() - mean * mean);

    auto [min_it, max_it] = std::minmax_element(iteration_times_ms.begin(), iteration_times_ms.end());
    float min_time = *min_it;
    float max_time = *max_it;

    printf("\n=== Iteration Timing Statistics ===\n");
    printf("Number of iterations: %lu\n", iteration_times_ms.size());
    printf("Min time (ms): %.3f\n", min_time);
    printf("Max time (ms): %.3f\n", max_time);
    printf("Mean time (ms): %.3f\n", mean);
    printf("Stddev time (ms): %.3f\n", stddev);
    printf("===================================\n\n");

    return result;
}