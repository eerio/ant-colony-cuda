#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <vector>
#include <numeric>
#include <float.h> // For FLT_EPSILON
#include <algorithm>
#include <cmath>
#include "tsp.h"


#define MAX_CITIES 1024

// this stores 0 in temp[0], stores sum(temp[0..n-2]) in temp[n-1], destroys temp[n-1]
// it assumes that n is a power of two!!!!!!
// https://users.umiacs.umd.edu/~ramanid/cmsc828e_gpusci/ScanTalk.pdf
__device__ void prescan(float *temp, int n)
{
    int thid = threadIdx.x;
    int offset = 1;

    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();

        if (thid < d && (offset * (2 * thid + 2) - 1) < n)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    if (thid == 0) { temp[n - 1] = 0; } // clear the last element
    
    for (int d = 1; d < n && offset > 1; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        
        if (thid < d && (offset * (2 * thid + 2) - 1) < n)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }    
    // __syncthreads();
}

__global__ void tourConstructionKernelQueen(
    int *tours,
    const float *choice_info,
    int num_ants,
    int num_cities,
    curandState *states
) {
    extern __shared__ float shared_data[]; 
    float *selection_probs = shared_data;      // Shared memory for selection probs
    int *tabu_list = (int*)&selection_probs[MAX_CITIES + 1]; // Shared memory for tabu list
    float *total_sum = (float *)&tabu_list[num_cities];

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

        // Compute selection probabilities
        for (int city = worker_id; city < MAX_CITIES; city += blockDim.x) {
            if (city < num_cities) {
                float prob = tabu_list[city] ? 0.0f : choice_info[current_city * num_cities + city];
                selection_probs[city] = prob;
            } else {
                selection_probs[city] = 0;
            }
        }
        __syncthreads();
        if (worker_id == 0) { selection_probs[MAX_CITIES] = selection_probs[MAX_CITIES - 1]; }
        __syncthreads();
        prescan(selection_probs, MAX_CITIES); // destroys last city, but it's saved already
        __syncthreads();
        if (worker_id == 0) {
            selection_probs[MAX_CITIES] += selection_probs[MAX_CITIES - 1];
            *total_sum = selection_probs[MAX_CITIES];
        }
        __syncthreads();

        // Draw random value for roulette selection
        float rand_val = curand_uniform(&local_state) * *total_sum;

        // Perform Roulette Wheel-style selection
        int selected_city = -1;
        if (
            (worker_id == 0 && selection_probs[1] >= rand_val)
            || (selection_probs[worker_id + 1] >= rand_val && selection_probs[worker_id] < rand_val)
         ) {
            // hack; change to another shared variable
            if (tabu_list[worker_id]) {
                printf("This is weird! %d %f %f %f %f\n", worker_id, selection_probs[worker_id], selection_probs[worker_id + 1], rand_val, *total_sum);
                assert(false);
            }
            if (!tabu_list[worker_id]) {*total_sum = worker_id;}
        }
        __syncthreads();
        if (worker_id == 0 && (int)*total_sum == -1) {
            // shouldn't happen
            for (int city=0; city < num_cities; ++city) { if (!tabu_list[city]) { selected_city = city; break; }}
        }
        __syncthreads();
        selected_city = (int)*total_sum;
        assert(selected_city >= 0);
        assert(selected_city < num_cities);
        assert(!tabu_list[selected_city]);

        if (worker_id == 0) {
            tours[queen_id * num_cities + step] = selected_city;
            tabu_list[selected_city] = true;
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
    int num_queens = 68;
    int num_workers = num_cities;
    assert(num_cities >= num_workers);
    assert(num_cities / num_workers <= 32);
    assert(num_cities <= 1024);

    int value;
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    const int max_shmem_per_block = value;

    // optimization: assign threads uniformly to blocks
    // int num_blocks = (MAX_TPB + num_workers - 1) / MAX_TPB;
    // int threads_per_block = MAX_TPB;
    int num_blocks = num_queens; // 68; // rtx 2080ti has 68 SMs
    // int threads_per_block = (68 + num_workers - 1) / 68;
    
    int float_section_size = (MAX_CITIES + 1) * sizeof(float);  // selection_probs
    // ant_visited; round up to multiple of 4
    int int_section_size = num_cities * sizeof(int);
    int thread_memory_size = float_section_size + int_section_size + sizeof(float);
    
    // 4: number of units on a single SM of RTX 2080 Ti
    assert(thread_memory_size < max_shmem_per_block / 4);
    int threads_per_block = num_workers; // max_shmem_per_block / thread_memory_size;
    int shared_memory_size = thread_memory_size; // per block!

    cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, 0);
    assert(threads_per_block > 0);
    assert(threads_per_block <= value);
    assert(shared_memory_size > 0);
    assert(shared_memory_size <= max_shmem_per_block);
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    assert(num_blocks / 68 <= 32); // rtx 2080ti: max 32 blocks per SM
    assert(num_blocks <= MAX_BLOCKS);

    printf("Config: num_blocks: %d, tpb: %d, shmem: %d\n", num_blocks, threads_per_block, shared_memory_size);

    size_t matrix_size = sizeof(float) * num_cities * num_cities;
    int* d_ant_tours;
    // bool* d_ant_visited;
    curandState* d_rand_states;
    float* d_tour_lengths;
    float* d_choice_info;
    float* d_distances;
    // float* d_selection_probs;
    float* d_pheromone;
    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_ant_tours, sizeof(int) * num_queens * num_cities);
    // cudaMalloc(&d_ant_visited, sizeof(bool) * num_queens * num_cities);
    cudaMalloc(&d_rand_states, sizeof(curandState) * num_queens);
    cudaMalloc(&d_tour_lengths, sizeof(float) * num_queens);
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    // cudaMalloc(&d_selection_probs, sizeof(float) * num_queens * num_cities);
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

#ifdef DEBUG
        int* h_ant_tours = new int[num_queens * num_cities];
        cudaMemcpy(h_ant_tours, d_ant_tours, sizeof(int) * num_queens * num_cities, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        verifyToursHost(h_ant_tours, num_queens, num_cities);
        delete[] h_ant_tours;
#endif

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
    // cudaFree(d_ant_visited);
    cudaFree(d_rand_states);
    cudaFree(d_tour_lengths);
    cudaFree(d_choice_info);
    cudaFree(d_distances);
    // cudaFree(d_selection_probs);

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