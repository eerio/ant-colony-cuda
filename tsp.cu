#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "tsp.h"

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
    const float* d_pheromone,
    const float* d_distances,
    int num_cities,
    float alpha,
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
            float tau = d_pheromone[idx];
            float dist = d_distances[idx];
            float eta = (dist > 0.0f) ? 1.0f / dist : 0.0f;
            d_choice_info[idx] = __powf(tau, alpha) * __powf(eta, beta);
        }
    }
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


__global__ void evaporatePheromoneKernel(float* d_pheromone, float evaporation_rate, int num_cities) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_cities * num_cities;

    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        d_pheromone[idx] *= (1.0f - evaporation_rate);
    }
}

__global__ void depositPheromoneKernel(
    float* d_pheromone,
    const int* d_ant_tours,
    const float* d_tour_lengths,
    int num_ants,
    int num_cities,
    float Q
) {
    int ant_idx = get_idx(num_ants);
    if (ant_idx >= num_ants) { return; }

    const int* tour = &d_ant_tours[ant_idx * num_cities];
    float contribution = Q / d_tour_lengths[ant_idx];

    for (int i = 0; i < num_cities - 1; ++i) {
        int from = tour[i];
        int to = tour[i + 1];
        atomicAdd(&d_pheromone[from * num_cities + to], contribution);
        atomicAdd(&d_pheromone[to * num_cities + from], contribution); // symmetric TSP
    }

    // Add pheromone for the return to the start city
    int last = tour[num_cities - 1];
    int first = tour[0];
    atomicAdd(&d_pheromone[last * num_cities + first], contribution);
    atomicAdd(&d_pheromone[first * num_cities + last], contribution);
}