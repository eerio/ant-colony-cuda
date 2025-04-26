#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "tsp.h"

#define PI 3.14159265358979323846
#define TPB 7
#define FLT_MAX (3.4e+38F)

__device__ int get_idx(int num_ants) {
#ifdef DEBUG
    // our biggest test: pr2392, <= 3 * 1024
    assert(gridDim.x == 1 + (num_ants - 1) / TPB);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert(blockDim.x == TPB);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(threadIdx.x < TPB);
#else
    (void)num_ants;
#endif

    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void initialize_rand_states(curandState* rand_states, int num_ants, unsigned long seed) {
    int idx = get_idx(num_ants);
    if (idx >= num_ants) { return; }
    curand_init(seed, idx, 0, &rand_states[idx]);
}

__global__ void computeChoiceInfoKernel(
    float* d_choice_info,
    const float* d_distances,
    int num_cities,
    float beta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_cities * num_cities;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / num_cities;
        int col = idx % num_cities;

        if (row == col) {
            d_choice_info[idx] = 0.0f;
        } else {
            float dist = d_distances[idx];
            float eta = (dist > 0.0f) ? 1.0f / dist : 0.0f;
            d_choice_info[idx] = __powf(eta, beta);  // No pheromone
        }
    }
}

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

__global__ void computeTourLengthsKernel(
    const int* d_ant_tours,
    const float* d_distances,
    float* d_tour_lengths,
    int num_ants,
    int num_cities
) {
    int ant_idx = get_idx(num_ants);
    if (ant_idx >= num_ants) return;

    const int* tour = &d_ant_tours[ant_idx * num_cities];
    float length = 0.0f;

    for (int i = 0; i < num_cities - 1; ++i) {
        int from = tour[i];
        int to = tour[i + 1];
        length += d_distances[from * num_cities + to];
    }
    length += d_distances[tour[num_cities - 1] * num_cities + tour[0]];

    d_tour_lengths[ant_idx] = length;
}

void verifyToursHost(const int* h_ant_tours, int num_ants, int num_cities) {
    for (int ant = 0; ant < num_ants; ++ant) {
        const int* tour = &h_ant_tours[ant * num_cities];
        bool* visited = new bool[num_cities];
        for (int i = 0; i < num_cities; ++i) {
            visited[i] = false;
        }

        for (int i = 0; i < num_cities; ++i) {
            int city = tour[i];
            if (city < 0 || city >= num_cities) {
                printf("Host Verify Error: Ant %d has invalid city index %d at position %d\n", ant, city, i);
                delete[] visited;
                return;
            }
            if (visited[city]) {
                printf("Host Verify Error: Ant %d visits city %d more than once!\n", ant, city);
                delete[] visited;
                return;
            }
            visited[city] = true;
        }

        for (int i = 0; i < num_cities; ++i) {
            if (!visited[i]) {
                printf("Host Verify Error: Ant %d did not visit city %d\n", ant, i);
                delete[] visited;
                return;
            }
        }

        delete[] visited;
    }
}

TspResult solveTSPWorker(
    const TspInput& tsp_input,
    unsigned int num_iter,
    float alpha,  // NOT used anymore
    float beta,
    float evaporate,  // NOT used anymore
    unsigned int seed
) {
    int num_cities = tsp_input.dimension;
    int num_ants = 1971;
    size_t matrix_size = sizeof(float) * num_cities * num_cities;
    int* d_ant_tours;
    bool* d_ant_visited;
    curandState* d_rand_states;
    float* d_tour_lengths;
    float* d_choice_info;
    float* d_distances;
    float* d_selection_probs;
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

#ifdef DEBUG
    float* h_choice_info = new float[num_cities * num_cities];
    for (int i=0; i < num_cities * num_cities; ++i) {
        h_choice_info[i] = PI;
    }
    cudaMemcpy(d_choice_info, h_choice_info, matrix_size, cudaMemcpyHostToDevice);
    printf("Before: d_choice_info[-1][-2] = %f\n", h_choice_info[num_cities * num_cities - 2]);
#endif
    computeChoiceInfoKernel<<<num_blocks, TPB>>>(d_choice_info, d_distances, num_cities, beta);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudaMemcpy(h_choice_info, d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
    for (int i=0; i < num_cities * num_cities; ++i) {
        assert(h_choice_info[i] != PI);
    }
    printf("After: d_choice_info[-1][-2] = %f\n", h_choice_info[num_cities * num_cities - 2]);
#endif

    assert(num_iter == 1);
    for (unsigned int iter = 0; iter < num_iter; ++iter) {
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

        computeTourLengthsKernel<<<num_blocks, TPB>>>(
            d_ant_tours, d_distances, d_tour_lengths, num_ants, num_cities
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
    cudaFree(d_ant_visited);
    cudaFree(d_rand_states);
    cudaFree(d_tour_lengths);
    cudaFree(d_choice_info);
    cudaFree(d_distances);
    cudaFree(d_selection_probs);

    return result;
}