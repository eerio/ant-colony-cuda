#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <float.h>
#include "tsp.h"

__global__ void initializeRandStatesKernel(curandState* rand_states, int num_ants, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = tid; idx < num_ants; idx += stride) {
        curand_init(seed, idx, 0, &rand_states[idx]);
    }
}

void initializeRandStates(curandState* d_rand_states, int num_ants, unsigned long seed) {
    initializeRandStatesKernel<<<(MAX_TPB + num_ants - 1) / MAX_TPB, MAX_TPB>>>(d_rand_states, num_ants, seed);
    cudaDeviceSynchronize();
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
        float dist = d_distances[idx];
        float eps = 1e-9;
        if (dist < eps) {
            d_choice_info[idx] = 0.0f;
        } else {
            float tau = d_pheromone[idx];
            float eta = 1.0f / dist;
            d_choice_info[idx] = fmaxf(FLT_MIN, __powf(tau, alpha) * __powf(eta, beta));
        }
    }
}

void computeChoiceInfo(
    float* d_choice_info,
    const float* d_pheromone,
    const float* d_distances,
    int num_cities,
    float alpha,
    float beta
) {
#ifdef DEBUG
    const float pi = 3.14159265358979323846;
    float* h_choice_info = new float[num_cities * num_cities];
    for (int i=0; i < num_cities * num_cities; ++i) {
        h_choice_info[i] = pi;
    }
    cudaMemcpy(d_choice_info, h_choice_info, matrix_size, cudaMemcpyHostToDevice);
    printf("Before: d_choice_info[-1][-2] = %f\n", h_choice_info[num_cities * num_cities - 2]);
#endif

    computeChoiceInfoKernel<<<(MAX_TPB + num_cities * num_cities - 1) / MAX_TPB, MAX_TPB>>>(d_choice_info, d_pheromone, d_distances, num_cities, alpha, beta);
    cudaDeviceSynchronize();

#ifdef DEBUG
    cudaMemcpy(h_choice_info, d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
    for (int i=0; i < num_cities * num_cities; ++i) {
        assert(h_choice_info[i] != pi);
    }
    printf("After: d_choice_info[-1][-2] = %f\n", h_choice_info[num_cities * num_cities - 2]);
#endif
}

__global__ void initializePheromonesKernel(float* d_pheromone, int num_cities) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_cities * num_cities;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += stride) {
        d_pheromone[idx] = 1.0f;
    }
}

void initializePheromones(float* d_pheromone, int num_cities) {
    initializePheromonesKernel<<<(MAX_TPB + num_cities * num_cities - 1) / MAX_TPB, MAX_TPB>>>(d_pheromone, num_cities);
    cudaDeviceSynchronize();
}

__global__ void computeTourLengthsKernel(
    const int* d_ant_tours,
    const float* d_distances,
    float* d_tour_lengths,
    int num_ants,
    int num_cities
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ant_idx=tid; ant_idx < num_ants; ant_idx += blockDim.x * gridDim.x) {
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
}

void computeTourLengths(
    const int* d_ant_tours,
    const float* d_distances,
    float* d_tour_lengths,
    int num_ants,
    int num_cities
) {
    computeTourLengthsKernel<<<(MAX_TPB + num_ants - 1) / MAX_TPB, MAX_TPB>>>(
        d_ant_tours, d_distances, d_tour_lengths, num_ants, num_cities
    );
    cudaDeviceSynchronize();
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

void evaporatePheromone(float* d_pheromone, float evaporation_rate, int num_cities) {
    evaporatePheromoneKernel<<<(MAX_TPB + num_cities * num_cities - 1) / MAX_TPB, MAX_TPB>>>(d_pheromone, evaporation_rate, num_cities);
    cudaDeviceSynchronize();
}

__global__ void depositPheromoneKernel(
    float* d_pheromone,
    const int* d_ant_tours,
    const float* d_tour_lengths,
    int num_ants,
    int num_cities
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ant_idx=tid; ant_idx < num_ants; ant_idx += blockDim.x * gridDim.x) {
        const int* tour = &d_ant_tours[ant_idx * num_cities];
        float contribution = 1.0f / d_tour_lengths[ant_idx];

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
}

void depositPheromone(
    float* d_pheromone,
    const int* d_ant_tours,
    const float* d_tour_lengths,
    int num_ants,
    int num_cities
) {
    depositPheromoneKernel<<<(MAX_TPB + num_ants - 1) / MAX_TPB, MAX_TPB>>>(
        d_pheromone, d_ant_tours, d_tour_lengths, num_ants, num_cities
    );
    cudaDeviceSynchronize();
}