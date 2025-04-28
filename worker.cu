#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <vector>
#include <numeric>
#include <float.h> // For FLT_EPSILON
#include <algorithm>
#include <cmath>
#include <chrono>
#include "tsp.h"

#define MAX_CITIES 1024

__global__ void tourConstructionKernelWorker(
    int* d_ant_tours, // [A * N global]
    const float* d_choice_info, // [N * N global]
    int num_ants,
    int num_cities,
    curandState* d_rand_state
) {
    int ant_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ensure that memory is float-aligned
    extern __shared__ float shared[];
    float *selection_probs = (float*)shared;
    int *ant_visited = (int*)&selection_probs[MAX_CITIES];
    
    assert(ant_idx + gridDim.x * blockDim.x >= num_ants);
    int *ant_tour = &d_ant_tours[ant_idx * num_cities];

    for (int j = 0; j < num_cities; ++j) ant_visited[j] = false;

    int current_city = ant_idx % num_cities;
    ant_tour[0] = current_city;
    ant_visited[current_city] = true;

    curandState local_state = d_rand_state[ant_idx];

    for (int step = 1; step < num_cities; ++step) {
        float sum_probs = 0.0f;

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

        if (next_city == -1) {
            for (int j=0; j < num_cities; ++j) {
                if (!ant_visited[j]) {
                    next_city = j;
                    break;
                }
            }
        }

        ant_tour[step] = next_city;
        ant_visited[next_city] = true;
        current_city = next_city;
    }

    d_rand_state[ant_idx] = local_state;
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
    int num_ants = num_cities;
    int num_blocks = num_ants;

    int float_section_size = MAX_CITIES * sizeof(float); 
    int bool_section_size_aligned = MAX_CITIES * sizeof(int);
    int thread_memory_size = float_section_size + bool_section_size_aligned;
    
    int threads_per_block = 1;
    int shared_memory_size = thread_memory_size;

    printf("Config: num_blocks: %d, tpb: %d, shmem: %d\n", num_blocks, threads_per_block, shared_memory_size);

    size_t matrix_size = sizeof(float) * num_cities * num_cities;
    int* d_ant_tours;
    curandState* d_rand_states;
    float* d_tour_lengths;
    float* d_choice_info;
    float* d_distances;
    float* d_pheromone;
    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_ant_tours, sizeof(int) * num_ants * num_cities);
    cudaMalloc(&d_rand_states, sizeof(curandState) * num_ants);
    cudaMalloc(&d_tour_lengths, sizeof(float) * num_ants);
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    cudaMemcpy(d_distances, tsp_input.distances, matrix_size, cudaMemcpyHostToDevice);

    initializeRandStates(d_rand_states, num_ants, seed);
    initializePheromones(d_pheromone, num_cities);
    
    std::vector<float> iteration_times_ms;
    cudaEvent_t iter_start, iter_end;
    HANDLE_ERROR(cudaEventCreate(&iter_start));
    HANDLE_ERROR(cudaEventCreate(&iter_end));

    cudaStream_t stream;
    // std::vector<cudaGraphNode_t> _node_list;
    // cudaGraphExec_t _graph_exec;
    // cudaGraphNode_t new_node;
    // cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // cudaGraph_t _capturing_graph;
    // cudaStreamCaptureStatus _capture_status;
    // const cudaGraphNode_t *_deps;
    // size_t _dep_count;
    // cudaStreamGetCaptureInfo_v2(stream, &_capture_status, nullptr &_capturing_graph, &_deps, &_dep_count);

    // cudakernelNodeParams _dynamic_params_cuda;
    // cudaGraphAddKernelNode(&new_node, _capturing_graph, _deps, _dep_count, &_dynamic_params_cuda);
    // _node_list.push_back(new_node);
    // cudaStreamUpdateCaptureDependencies(stream, &new_node, 1, 1); 
    // cudaGraph_t _captured_graph;
    // cudaStreamEndCapture(stream, &_captured_graph);

    // cudaGraphInstantiate(&_graph_exec, _captured_graph, nullptr, nullptr, 0);

    // // updating
    // cudakernelNodeParams _dynamic_params_updated_cuda;
    // cudaGraphExecKernelNodeSetParams(_graph_exec, _node_list[0], &_dynamic_params_updated_cuda);

    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int iter = 0; iter < num_iter; ++iter) {
        // cudaDeviceSynchronize();
        HANDLE_ERROR(cudaEventRecord(iter_start));
        
        if (!graphCreated) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            computeChoiceInfo(d_choice_info, d_pheromone, d_distances, num_cities, alpha, beta);
            
            tourConstructionKernelWorker<<<num_blocks, threads_per_block, shared_memory_size>>>(
                d_ant_tours, d_choice_info, num_ants, num_cities, d_rand_states
            );
            cudaDeviceSynchronize();
            
            evaporatePheromone(d_pheromone, evaporate, num_cities);
            
            computeTourLengths(
                d_ant_tours, d_distances, d_tour_lengths, num_ants, num_cities
            );
            
            depositPheromone(
                d_pheromone, d_ant_tours, d_tour_lengths, num_ants, num_cities
            );
            
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }
        
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();

        HANDLE_ERROR(cudaEventRecord(iter_end));
        HANDLE_ERROR(cudaEventSynchronize(iter_end));
        float elapsed_ms = 0.0f;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsed_ms, iter_start, iter_end));
        iteration_times_ms.push_back(elapsed_ms);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end - start;
    iteration_times_ms.clear(); 
    iteration_times_ms.push_back(duration_ms.count());
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