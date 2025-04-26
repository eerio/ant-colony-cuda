#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "tsp.h"

#define PI 3.14159265358979323846
#define TPB 1024
#define FLT_MAX (3.4e+38F)

__global__ void tourConstructionKernel(
    int* d_ant_tours,
    bool* d_ant_visited,
    float* d_choice_info,
    float* d_selection_probs,
    int num_ants,
    int num_cities,
    curandState* d_rand_state
) {
    int ant_idx = get_idx(num_ants);
    if (ant_idx >= num_ants) return;
    int i = ant_idx;

    int* ant_tour = &d_ant_tours[i * num_cities];
    bool* ant_visited = &d_ant_visited[i * num_cities];

    for (int j = 0; j < num_cities; ++j) ant_visited[j] = false;

    int current_city = i % num_cities;
    ant_tour[0] = current_city;
    ant_visited[current_city] = true;

    curandState local_state = d_rand_state[i];

    for (int step = 1; step < num_cities; ++step) {
        float sum_probs = 0.0f;
        float* selection_probs = &d_selection_probs[i * num_cities];

        for (int j = 0; j < num_cities; ++j) {
            if (!ant_visited[j]) {
                int idx = current_city * num_cities + j;
                selection_probs[j] = d_choice_info[idx];
                sum_probs += selection_probs[j];
            } else {
                selection_probs[j] = 0.0f;
            }
        }

        float r = curand_uniform(&local_state) * sum_probs;
        float accumulated_prob = 0.0f;
        int next_city = -1;

        for (int j = 0; j < num_cities; ++j) {
            if (selection_probs[j] > 0.0f && !ant_visited[j]) {
                accumulated_prob += selection_probs[j];
                if (accumulated_prob >= r) {
                    next_city = j;
                    break;
                }
            }
        }

        // if (next_city == -1) {
        //     printf("Error: No valid next city found for ant %d\n", i);
        // }

        // if (ant_visited[next_city]) {
        //     printf("Error! Choosing city which is already visited!\n");
        // }

        // if (next_city == current_city) {
        //     printf("Error! Choosing the same city again!\n");
        // }

        ant_tour[step] = next_city;
        ant_visited[next_city] = true;
        current_city = next_city;
    }

    d_rand_state[i] = local_state;
}

TspResult solveTSPWorker(
    const TspInput& tsp_input,
    unsigned int num_iter,
    float alpha,
    float beta,
    float evaporate,
    unsigned int seed
) {
    int num_cities = tsp_input.dimension;
    int num_ants = 128;
    size_t matrix_size = sizeof(float) * num_cities * num_cities;
    int* d_ant_tours;
    bool* d_ant_visited;
    curandState* d_rand_states;
    float* d_tour_lengths;
    float* d_choice_info;
    float* d_distances;
    float* d_selection_probs;
    float* d_pheromone;
    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_ant_tours, sizeof(int) * num_ants * num_cities);
    cudaMalloc(&d_ant_visited, sizeof(bool) * num_ants * num_cities);
    cudaMalloc(&d_rand_states, sizeof(curandState) * num_ants);
    cudaMalloc(&d_tour_lengths, sizeof(float) * num_ants);
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    cudaMalloc(&d_selection_probs, sizeof(float) * num_ants * num_cities);
    cudaMemcpy(d_distances, tsp_input.distances, matrix_size, cudaMemcpyHostToDevice);

    int num_blocks = (num_ants + TPB - 1) / TPB;
    assert(num_blocks * TPB >= num_ants);

    initialize_rand_states<<<num_blocks, TPB>>>(d_rand_states, num_ants, seed);
    cudaDeviceSynchronize();

    // Initialize pheromones to 1.0
    float* h_initial_pheromone = new float[num_cities * num_cities];
    for (int i = 0; i < num_cities * num_cities; ++i) {
        h_initial_pheromone[i] = 1.0f;
    }
    cudaMemcpy(d_pheromone, h_initial_pheromone, matrix_size, cudaMemcpyHostToDevice);
    delete[] h_initial_pheromone;

#ifdef DEBUG
    float* h_choice_info = new float[num_cities * num_cities];
    for (int i=0; i < num_cities * num_cities; ++i) {
        h_choice_info[i] = PI;
    }
    cudaMemcpy(d_choice_info, h_choice_info, matrix_size, cudaMemcpyHostToDevice);
    printf("Before: d_choice_info[-1][-2] = %f\n", h_choice_info[num_cities * num_cities - 2]);
#endif
    computeChoiceInfoKernel<<<num_blocks, TPB>>>(d_choice_info, d_pheromone, d_distances, num_cities, alpha, beta);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudaMemcpy(h_choice_info, d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
    for (int i=0; i < num_cities * num_cities; ++i) {
        assert(h_choice_info[i] != PI);
    }
    printf("After: d_choice_info[-1][-2] = %f\n", h_choice_info[num_cities * num_cities - 2]);
#endif

    for (unsigned int iter = 0; iter < num_iter; ++iter) {
        computeChoiceInfoKernel<<<num_blocks, TPB>>>(d_choice_info, d_pheromone, d_distances, num_cities, alpha, beta);
        cudaDeviceSynchronize();

        tourConstructionKernel<<<num_blocks, TPB>>>(
            d_ant_tours, d_ant_visited, d_choice_info, d_selection_probs, num_ants, num_cities, d_rand_states
        );
        cudaDeviceSynchronize();

#ifdef DEBUG
        int* h_ant_tours = new int[num_ants * num_cities];
        cudaMemcpy(h_ant_tours, d_ant_tours, sizeof(int) * num_ants * num_cities, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        verifyToursHost(h_ant_tours, num_ants, num_cities);
        delete[] h_ant_tours;
#endif

        evaporatePheromoneKernel<<<num_blocks, TPB>>>(d_pheromone, evaporate, num_cities);
        cudaDeviceSynchronize();

        computeTourLengthsKernel<<<num_blocks, TPB>>>(
            d_ant_tours, d_distances, d_tour_lengths, num_ants, num_cities
        );
        cudaDeviceSynchronize();

        depositPheromoneKernel<<<num_blocks, TPB>>>(
            d_pheromone, d_ant_tours, d_tour_lengths, num_ants, num_cities, 1.0 // TODO: what value Q here?
        );
        cudaDeviceSynchronize();
    }

    float* h_tour_lengths = new float[num_ants];
    cudaMemcpy(h_tour_lengths, d_tour_lengths, sizeof(float) * num_ants, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float best_length = FLT_MAX;
    int best_idx = -1;
    for (int i = 0; i < num_ants; ++i) {
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
    cudaFree(d_ant_visited);
    cudaFree(d_rand_states);
    cudaFree(d_tour_lengths);
    cudaFree(d_choice_info);
    cudaFree(d_distances);
    cudaFree(d_selection_probs);

    return result;
}