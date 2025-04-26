#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "tsp.h"

__global__ void initialize_rand_states(curandState* rand_states, int num_ants, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_ants) {
        curand_init(seed, idx, 0, &rand_states[idx]);
    }
}

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

__global__ void tourConstructionKernel(
    int* d_ant_tours,               // Stores the tours of each ant
    bool* d_ant_visited,            // Keeps track of visited cities for each ant
    float* d_choice_info,           // Choice info matrix (pheromone * heuristic)
    float* d_selection_probs,
    int num_ants,                   // Number of ants
    int num_cities,                 // Number of cities in the TSP instance
    curandState* d_rand_state       // Random state for each thread (ant)
) {
    int ant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_idx >= num_ants) { return; }
    // printf("Dupa from: %d\n", ant_idx);

    // If there are more ants than threads, we need to loop over them in batches
    for (int i = ant_idx; i < num_ants; i += blockDim.x * gridDim.x) {
        int* ant_tour = &d_ant_tours[i * num_cities];    // Tour for this ant
        bool* ant_visited = &d_ant_visited[i * num_cities]; // Visited cities for this ant

        // Initialize the tabu list: all cities unvisited
        for (int j = 0; j < num_cities; ++j) {
            ant_visited[j] = false;
        }

        // Randomly pick the initial city for the ant
        int current_city = i % num_cities; // Start at a random city (for simplicity)
        ant_tour[0] = current_city;
        ant_visited[current_city] = true;

        curandState local_state = d_rand_state[i];

        // Build the tour step-by-step
        for (int step = 1; step < num_cities; ++step) {
            float sum_probs = 0.0f;
            float* selection_probs = &d_selection_probs[i * num_cities];

            // Calculate the selection probabilities for all cities
            for (int j = 0; j < num_cities; ++j) {
                if (!ant_visited[j]) {
                    int idx = current_city * num_cities + j;
                    selection_probs[j] = d_choice_info[idx];
                    sum_probs += selection_probs[j];
                } else {
                    selection_probs[j] = 0.0f;
                }
            }

            // Roulette Wheel Selection
            // float r = curand_uniform(&local_state) * sum_probs;
            float r = 0.79 * sum_probs;
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

            // Mark the chosen city as visited
            if (next_city == -1) {
                printf("Error: No valid next city found for ant %d\n", i);
            }

            if (ant_visited[next_city]) {
                printf("Error! Choosing city which is already visited!\n");
            }
            // printf("Ant %d choosing city %d at step %d\n", ant_idx, next_city, step);
            ant_tour[step] = next_city;
            ant_visited[next_city] = true;
            current_city = next_city;
        }

        // Store the final state of the random state for the ant
        d_rand_state[i] = local_state;
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
    int ant_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ant_idx < num_ants) {
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
}

__global__ void computeTourLengthsKernel(
    const int* d_ant_tours,
    const float* d_distances,
    float* d_tour_lengths,
    int num_ants,
    int num_cities
) {
    int ant_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ant_idx < num_ants) {
        const int* tour = &d_ant_tours[ant_idx * num_cities];
        double length = 0.0f;

        for (int i = 0; i < num_cities - 1; ++i) {
            int from = tour[i];
            int to = tour[i + 1];
            // Check for valid indices
            if (from < 0 || from >= num_cities || to < 0 || to >= num_cities) {
                printf("Error: Invalid city index in tour! ant_idx=%d, i=%d, from=%d, to=%d\n", ant_idx, i, from, to);
                return; // Exit the kernel if an error is detected
            }
            length += d_distances[from * num_cities + to];
            if (length < 0) {
                printf("Error: Negative length encountered! ant_idx=%d, i=%d, length=%f\n", ant_idx, i, length);
                return; // Exit the kernel if an error is detected
            }
            if (length > 1e6) {
                printf("Warning: Length too large! ant_idx=%d, i=%d, length=%f\n", ant_idx, i, length);
            }
        }

        // Add distance from last to first to complete the tour
        length += d_distances[tour[num_cities - 1] * num_cities + tour[0]];
        if (length < 0) {
            printf("Error: Negative length encountered! ant_idx=%d, i=last, length=%f\n", ant_idx, length);
            length = 42.42;
        }
        if (length > 1e6) {
            printf("Warning: Length too large! ant_idx=%d, i=last, length=%f\n", ant_idx, length);
            length = 42.42;
        }

        d_tour_lengths[ant_idx] = (float)length;
    }
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
    int num_ants = 8;

    // Allocate memory for ant tours and visited matrix
    int* d_ant_tours;
    bool* d_ant_visited;
    curandState* d_rand_states;

    cudaMalloc(&d_ant_tours, sizeof(int) * num_ants * num_cities);
    cudaMalloc(&d_ant_visited, sizeof(bool) * num_ants * num_cities);
    float* d_tour_lengths;
    cudaMalloc(&d_tour_lengths, sizeof(float) * num_ants);

    // float* h_initial_tour_lengths = new float[num_ants];
    // for (int i=0; i < num_ants; ++i) {
    //     h_initial_tour_lengths[i] = 1e12f;
    // }
    // cudaMemcpy(d_tour_lengths, h_initial_tour_lengths, sizeof(float) * num_ants, cudaMemcpyHostToDevice);
    // delete[] h_initial_tour_lengths;

    cudaMalloc(&d_rand_states, sizeof(curandState) * num_ants);

    // Initialize visited array to 0
    cudaMemset(d_ant_visited, 0, sizeof(bool) * num_ants * num_cities);

    // Allocate memory
    float *d_choice_info, *d_pheromone, *d_distances, *d_selection_probs;
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    cudaMalloc(&d_selection_probs, sizeof(float) * num_cities * num_cities);

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
    int threads_per_block = 256;
    int num_blocks = (num_ants + threads_per_block - 1) / threads_per_block;

    assert(num_blocks * threads_per_block >= num_ants);

    initialize_rand_states<<<num_blocks, threads_per_block>>>(d_rand_states, num_ants, seed);
    cudaDeviceSynchronize();

    for (unsigned int iter = 0; iter < num_iter; ++iter) {
        computeChoiceInfoKernel<<<num_blocks, threads_per_block>>>(
            d_choice_info, d_pheromone, d_distances, num_cities, alpha, beta
        );
        cudaDeviceSynchronize();

        tourConstructionKernel<<<num_blocks, threads_per_block>>>(
            d_ant_tours,
            d_ant_visited,
            d_choice_info,
            d_selection_probs,
            num_ants,
            num_cities,
            d_rand_states
        );
        cudaDeviceSynchronize();
#ifdef DEBUG
        // Allocate host memory
int* h_ant_tours = new int[num_ants * num_cities];

// Copy tours from device to host
cudaMemcpy(h_ant_tours, d_ant_tours, sizeof(int) * num_ants * num_cities, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

// Verify tours
verifyToursHost(h_ant_tours, num_ants, num_cities);

// Free temporary host memory
delete[] h_ant_tours;
#endif

        evaporatePheromoneKernel<<<num_blocks, threads_per_block>>>(d_pheromone, evaporate, num_cities);
        cudaDeviceSynchronize();
        computeTourLengthsKernel<<<num_blocks, threads_per_block>>>(
            d_ant_tours, d_distances, d_tour_lengths, num_ants, num_cities
        );
        cudaDeviceSynchronize();
        float Q = 1.0f;  // You can tune this
        depositPheromoneKernel<<<num_blocks, threads_per_block>>>(
            d_pheromone, d_ant_tours, d_tour_lengths, num_ants, num_cities, Q
        );
        cudaDeviceSynchronize();
    }

    // Return dummy result
    // Find best tour
    float best_tour_length = 1e7f;
    int best_ant_index = -1;
    float* h_tour_lengths = new float[num_ants];
    cudaDeviceSynchronize();
    cudaMemcpy(h_tour_lengths, d_tour_lengths, sizeof(float) * num_ants, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < num_ants; ++i) {
        if (h_tour_lengths[i] < best_tour_length) {
            best_tour_length = h_tour_lengths[i];
            best_ant_index = i;
        }
    }
    delete[] h_tour_lengths;
    if (best_ant_index == -1) {
        printf("Error: No valid ant found!\n");
        return {};
    }

    // Copy best tour to host
    unsigned int* h_best_tour = new unsigned int[num_cities];
    cudaMemcpy(h_best_tour, &d_ant_tours[best_ant_index * num_cities], sizeof(unsigned int) * num_cities, cudaMemcpyDeviceToHost);

    // Return result
    TspResult result;
    result.dimension = num_cities;
    result.cost = best_tour_length;
    result.tour = h_best_tour;

    if (result.cost < 0) {
        printf("Error: Negative length encountered! ant_idx=last, i=last, length=%f\n", result.cost);
    }
    if (result.cost > 1e6) {
        printf("Warning: Length too large! ant_idx=last, i=last, length=%f\n",  result.cost);
    }

    // Free GPU memory
    cudaFree(d_choice_info);
    cudaFree(d_pheromone);
    cudaFree(d_distances);
    cudaFree(d_tour_lengths);
    cudaFree(d_ant_tours);
    cudaFree(d_ant_visited);
    cudaFree(d_rand_states);

    return result;
}
