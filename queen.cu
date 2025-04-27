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

// #include <cuda_runtime.h>
// #include <curand_kernel.h>
// #include <float.h> // For FLT_EPSILON
// #include <stdio.h> // Uncomment for printf debugging

// Enable printf from device: Compile with e.g. -arch=sm_70 -rdc=true, link with -lcudadevrt
// #define KERNEL_DEBUG

// Inclusive prefix sum (scan) - Assumed correct from previous step
__device__ void prefixSumInclusiveArbitraryN(volatile float* sdata, unsigned int tid, unsigned int N) {
    if (tid >= N) { return; }
     for (unsigned int stride = 1; stride < N; stride *= 2) {
        float value_from_behind = 0;
        // Read from valid index within bounds [0, N-1]
        if (tid >= stride && tid < N) {
             value_from_behind = sdata[tid - stride];
        }
        __syncthreads(); // Ensure all reads complete before writes
        // Write to valid index within bounds [0, N-1]
        if (tid >= stride && tid < N) {
             sdata[tid] += value_from_behind;
        }
        __syncthreads(); // Ensure all writes complete before next stride
    }
}


__global__ void tourConstructionKernelQueen(
    int* d_ant_tours,      // Output: Tours for each ant [num_ants * num_cities]
    bool* d_ant_visited,   // Global visited flags [num_ants * num_cities]
    float* d_choice_info,  // Precomputed pheromone^alpha * heuristic^beta [num_cities * num_cities]
    float* d_selection_probs, // Workspace - unused
    int num_ants,
    int num_cities,
    curandState* d_rand_states // Per-ant random states
) {
    // --- Basic Setup ---
    int ant_id = blockIdx.x;
    int worker_tid = threadIdx.x; // Represents potential next city index

    // Boundary checks
    if (ant_id >= num_ants) return;
    // Handle trivial case where no tour needs construction
    if (num_cities <= 1) {
         if (num_cities == 1 && worker_tid == 0) {
             d_ant_tours[ant_id * num_cities + 0] = 0; // Tour is just city 0
         }
         return;
    }

    // --- Shared Memory ---
    extern __shared__ char shared_mem[];
    int* s_visited_flags = (int*)shared_mem; // 1 if unvisited, 0 if visited
    float* s_step_probs = (float*)(shared_mem + num_cities * sizeof(int)); // Raw probabilities for this step
    // Place broadcast variable carefully at the end
    volatile int* s_next_city_ptr = (volatile int*)(&s_step_probs[num_cities]); // Requires +sizeof(int) shared mem

    // --- Initialization ---
    curandState local_rand_state = d_rand_states[ant_id];

    // Initialize shared visited flags & clear global flags for this ant
    if (worker_tid < num_cities) {
        s_visited_flags[worker_tid] = 1; // Mark all as unvisited locally
        d_ant_visited[ant_id * num_cities + worker_tid] = false; // Clear global flag
    }
    // Initialize padding area in shared memory if blockDim.x > num_cities
    // else if (worker_tid < blockDim.x) {
    //     s_visited_flags[worker_tid] = 0; // Ensure padding doesn't interfere
    // }
    __syncthreads();

    int start_city = 0; // Fixed start city
    int current_city = start_city;

    // Set start city state
    if (worker_tid == 0) { // Only thread 0 writes global tour/visited start state
        d_ant_tours[ant_id * num_cities + 0] = start_city;
        d_ant_visited[ant_id * num_cities + start_city] = true;
    }
    // The specific thread corresponding to start_city marks it visited in shared memory
    if (worker_tid == start_city) {
         s_visited_flags[start_city] = 0;
    }
    __syncthreads(); // Ensure shared flag is set before first step


    // --- Tour Construction Loop ---
    int remaining_cities = num_cities - 1;
    for (int step = 1; step < num_cities; ++step) {

        // 1. Calculate Step Probabilities based on current_city
        float prob = 0.0f;
        if (worker_tid < num_cities) {
            // Probability is only non-zero if the city 'worker_tid' has NOT been visited
            if (s_visited_flags[worker_tid] == 1) {
                // Ensure we don't use d_choice_info diagonal or invalid values
                float choice_val = d_choice_info[current_city * num_cities + worker_tid];
                // Ensure probability is non-negative
                prob = fmaxf(0.0f, choice_val);
            } // else prob remains 0.0f
            s_step_probs[worker_tid] = prob;
        }
        // Ensure padding area (if any) has zero probability
        // else if (worker_tid < blockDim.x) {
        //    s_step_probs[worker_tid] = 0.0f;
        // }
        __syncthreads(); // Ensure all probabilities are written to shared memory

        // 2. Calculate Cumulative Probabilities (Scan)
        // Pass only the valid range [0..num_cities-1] to the scan function
        prefixSumInclusiveArbitraryN(s_step_probs, worker_tid, num_cities);
        // s_step_probs[num_cities - 1] now holds the total sum of probabilities

        // 3. Select Next City (Thread 0 coordinates)
        int next_city = -1; // Initialize as "not found"
        if (worker_tid == 0) {
            float total_prob = 0.0f;
            // Read total probability safely
            if (num_cities > 0) { // Should always be true here due to earlier check
                total_prob = s_step_probs[num_cities - 1];
            }

            if (total_prob <= FLT_EPSILON || remaining_cities <= 0) {
                // --- Fallback: No valid probabilities or no cities left ---
                next_city = -1;
                // Find the *first* available city index marked as unvisited in shared memory
                for(int c = 0; c < num_cities; ++c) {
                    if (s_visited_flags[c] == 1) {
                        next_city = c;
                        break;
                    }
                }
                // If next_city is STILL -1 here, it means all flags are 0, but step < num_cities. This is an error state.
                if (next_city == -1) {
                     *s_next_city_ptr = -1; // Signal failure clearly
                }
            } else {
                // --- Standard Roulette Wheel ---
                float rand_val = curand_uniform(&local_rand_state) * total_prob;
                rand_val = fminf(rand_val, total_prob - 1e-6f);
                // Handle rand_val being exactly 0 slightly differently if needed,
                // but searching from index 0 should cover it.

                // Search for the first unvisited city whose cumulative probability range covers rand_val
                next_city = -1; // Reset before search
                for (int city_idx = 0; city_idx < num_cities; ++city_idx) {
                    // Check: Is this city unvisited? AND Does its cumulative probability cover rand_val?
                    if (s_visited_flags[city_idx] == 1 && rand_val <= s_step_probs[city_idx]) {
                        // Check if this is the first city in the cumulative list that satisfies this.
                        // Because we iterate 0 to N-1, the first hit IS the correct one.
                        // Need to handle potential case where rand_val exactly equals a previous cumulative sum
                        // for a VISITED city.
                        float prev_cum_prob = (city_idx == 0) ? 0.0f : s_step_probs[city_idx - 1];
                        if (rand_val > prev_cum_prob || city_idx == 0) { // Ensure rand_val falls *within* this city's specific probability mass
                           next_city = city_idx;
                           break; // Found the city
                        }
                        // If rand_val == prev_cum_prob, it belongs to the previous city's range end.
                        // The loop will continue and find the correct bin.
                    }
                }

                 // If loop finished but no city found (should only happen if total_prob > 0 but all contributing cities are somehow flagged visited - logic error)
                 if (next_city == -1 || s_visited_flags[next_city] == 0) {
                     // --- Fallback (Selection algorithm failed) ---
                     next_city = -1;
                     for(int c = 0; c < num_cities; ++c) {
                         if (s_visited_flags[c] == 1) {
                             next_city = c; // Pick first available as last resort
                             break;
                         }
                     }
                     if (next_city == -1) { // Should be impossible if total_prob > 0
                         *s_next_city_ptr = -1; // Signal failure
                     }
                 }
            }

            // 4. Broadcast chosen city (or -1 for failure)
            // This assignment happens ONLY IF thread 0 successfully found a next_city (either via roulette or fallback)
             if (next_city != -1) {
                 *s_next_city_ptr = next_city;
                 // Update global state (only thread 0 does this for the step)
                 d_ant_tours[ant_id * num_cities + step] = next_city;
                 d_ant_visited[ant_id * num_cities + next_city] = true;
             } else {
                 // If next_city remained -1 after all checks, signal failure
                 *s_next_city_ptr = -1;
             }
        } // End if worker_tid == 0

        // 5. Synchronize and Prepare for Next Step
        __syncthreads(); // Ensure s_next_city_ptr written by thread 0 is visible to all

        int chosen_city_for_this_step = *s_next_city_ptr; // All threads read broadcast value

        if (chosen_city_for_this_step == -1) {
             // Ant failed to find a next city, stop its tour construction
#ifdef KERNEL_DEBUG
            // if(ant_id == 0 && worker_tid == 0) printf("Ant 0, Step %d: Halting due to selection failure.\n", step);
#endif
             break;
        }

        // Update local 'current_city' for the *next* iteration's probability calculation
        current_city = chosen_city_for_this_step;
        remaining_cities--; // Decrement count of cities left to visit

        // Mark the chosen city as visited IN SHARED memory for the next iteration
        // Done by the thread whose ID matches the chosen city
        if (worker_tid == current_city) { // 'current_city' now holds the city chosen in this step
             s_visited_flags[current_city] = 0;
        }

        // Synchronize to ensure shared visited flag update is visible before the next loop iteration starts
        __syncthreads();

    } // End tour construction loop (step)

    // --- Finalization ---
    // Save the final random state back to global memory
    if (worker_tid == 0) {
        d_rand_states[ant_id] = local_rand_state;
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
    int num_ants = num_cities;
    // One queen = one block; one city per thread
    int num_blocks = num_ants; 
    int threads_per_block = num_cities;

    assert(num_cities <= MAX_CITIES);
    assert(num_blocks <= MAX_BLOCKS);
    assert(threads_per_block <= MAX_TPB);
    assert(num_blocks * threads_per_block >= num_ants);
    
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

    initializeRandStates(d_rand_states, num_ants, seed);
    initializePheromones(d_pheromone, num_cities);
    
    std::vector<float> iteration_times_ms;
    cudaEvent_t iter_start, iter_end;
    HANDLE_ERROR(cudaEventCreate(&iter_start));
    HANDLE_ERROR(cudaEventCreate(&iter_end));
    for (unsigned int iter = 0; iter < num_iter; ++iter) {
        HANDLE_ERROR(cudaEventRecord(iter_start));
        computeChoiceInfo(d_choice_info, d_pheromone, d_distances, num_cities, alpha, beta);

        tourConstructionKernelQueen<<<num_blocks, threads_per_block>>>(
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

        evaporatePheromone(d_pheromone, evaporate, num_cities);

        computeTourLengths(
            d_ant_tours, d_distances, d_tour_lengths, num_ants, num_cities
        );

        depositPheromone(
            d_pheromone, d_ant_tours, d_tour_lengths, num_ants, num_cities
        );

        HANDLE_ERROR(cudaEventRecord(iter_end));
        HANDLE_ERROR(cudaEventSynchronize(iter_end));
        float elapsed_ms = 0.0f;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsed_ms, iter_start, iter_end));
        iteration_times_ms.push_back(elapsed_ms);
    }
    HANDLE_ERROR(cudaEventDestroy(iter_start));
    HANDLE_ERROR(cudaEventDestroy(iter_end));

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
