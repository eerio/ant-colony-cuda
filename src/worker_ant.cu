// worker_ant.cu
#include "worker_ant.h"
#include "cuda_utils.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>

// Device constants for ACO parameters
__constant__ float d_alpha;
__constant__ float d_beta;
__constant__ float d_evaporation_rate;
__constant__ int d_num_cities;

// Structure to represent an ant's tour
struct AntTour {
    int* visited_cities;  // Tour path (city indices)
    int* visited;         // Boolean array to track visited cities
    float tour_length;    // Total tour length
};

// Initialize RNG states
__global__ void initRNG(curandState* states, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

// Kernel for each ant to build its tour
__global__ void buildAntTours(curandState* states, 
                             float* distances,
                             float* pheromones,
                             AntTour* ant_tours) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = states[ant_id];
    
    AntTour* my_tour = &ant_tours[ant_id];
    
    // Start at a random city
    int current_city = curand(&local_state) % d_num_cities;
    my_tour->visited_cities[0] = current_city;
    my_tour->visited[current_city] = 1;
    my_tour->tour_length = 0.0f;
    
    // Build tour one city at a time
    for (int step = 1; step < d_num_cities; step++) {
        float total_prob = 0.0f;
        float probs[1024];  // Assuming max cities is 1024
        
        // Calculate probabilities for unvisited cities
        for (int next_city = 0; next_city < d_num_cities; next_city++) {
            if (!my_tour->visited[next_city]) {
                float distance = distances[current_city * d_num_cities + next_city];
                float pheromone = pheromones[current_city * d_num_cities + next_city];
                
                // Equation (1) from the paper
                if (distance > 0) {
                    probs[next_city] = powf(pheromone, d_alpha) * powf(1.0f / distance, d_beta);
                } else {
                    probs[next_city] = 0.0f;
                }
                
                total_prob += probs[next_city];
            } else {
                probs[next_city] = 0.0f;
            }
        }
        
        // Use roulette wheel selection
        float random = curand_uniform(&local_state) * total_prob;
        float prob_sum = 0.0f;
        int next_city = -1;
        
        for (int i = 0; i < d_num_cities; i++) {
            if (!my_tour->visited[i]) {
                prob_sum += probs[i];
                if (prob_sum >= random) {
                    next_city = i;
                    break;
                }
            }
        }
        
        // In case of numerical issues, just pick the first unvisited city
        if (next_city == -1) {
            for (int i = 0; i < d_num_cities; i++) {
                if (!my_tour->visited[i]) {
                    next_city = i;
                    break;
                }
            }
        }
        
        // Add city to tour
        my_tour->visited_cities[step] = next_city;
        my_tour->visited[next_city] = 1;
        my_tour->tour_length += distances[current_city * d_num_cities + next_city];
        current_city = next_city;
    }
    
    // Complete the tour by returning to the starting city
    int first_city = my_tour->visited_cities[0];
    my_tour->tour_length += distances[current_city * d_num_cities + first_city];
    
    // Save RNG state back
    states[ant_id] = local_state;
}

// Kernel to update pheromones based on ant tours
__global__ void updatePheromones(float* pheromones,
                                AntTour* ant_tours,
                                int num_ants) {
    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_id >= d_num_cities * d_num_cities) return;
    
    int i = edge_id / d_num_cities;
    int j = edge_id % d_num_cities;
    
    if (i == j) {
        pheromones[edge_id] = 0.0f;
        return;
    }
    
    // Evaporation (equation 2)
    pheromones[edge_id] *= (1.0f - d_evaporation_rate);
    
    // Add new pheromones from ants (equation 3)
    for (int ant = 0; ant < num_ants; ant++) {
        AntTour* tour = &ant_tours[ant];
        
        // Check if edge (i,j) is in this ant's tour
        for (int step = 0; step < d_num_cities; step++) {
            int city1 = tour->visited_cities[step];
            int city2 = tour->visited_cities[(step + 1) % d_num_cities];
            
            if ((city1 == i && city2 == j) || (city1 == j && city2 == i)) {
                pheromones[edge_id] += 1.0f / tour->tour_length;
                break;
            }
        }
    }
}

// Function to find the best tour among all ants
__global__ void findBestTour(AntTour* ant_tours, 
                           int num_ants,
                           int* best_tour,
                           float* best_length) {
    __shared__ float shared_best_length;
    __shared__ int shared_best_ant;
    
    if (threadIdx.x == 0) {
        shared_best_length = FLT_MAX;
        shared_best_ant = -1;
    }
    __syncthreads();
    
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id < num_ants) {
        float my_length = ant_tours[ant_id].tour_length;
        
        // Atomic min to find best tour
        atomicMinf(&shared_best_length, my_length, &shared_best_ant, ant_id);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        if (shared_best_length < *best_length) {
            *best_length = shared_best_length;
            
            // Copy the best tour
            for (int i = 0; i < d_num_cities; i++) {
                best_tour[i] = ant_tours[shared_best_ant].visited_cities[i] + 1; // Convert to 1-indexed
            }
        }
    }
}

void runWorkerAntACO(const TSPInstance& tsp, 
                   int num_iterations,
                   float alpha,
                   float beta,
                   float evaporation_rate,
                   unsigned int seed,
                   std::vector<int>& best_tour,
                   float& best_tour_length) {
    int num_cities = tsp.num_cities;
    int num_ants = num_cities; // Usually equal to the number of cities
    
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_beta, &beta, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_evaporation_rate, &evaporation_rate, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_cities, &num_cities, sizeof(int)));
    
    // Allocate device memory
    float *d_distances, *d_pheromones;
    int *d_best_tour;
    float *d_best_length;
    curandState *d_rng_states;
    
    CUDA_CHECK(cudaMalloc(&d_distances, num_cities * num_cities * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pheromones, num_cities * num_cities * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_tour, num_cities * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_length, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng_states, num_ants * sizeof(curandState)));
    
    // Copy distances to device
    CUDA_CHECK(cudaMemcpy(d_distances, tsp.distance_matrix, num_cities * num_cities * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize pheromones with a small constant
    float initial_pheromone = 1.0f / (num_cities * num_cities);
    float *h_initial_pheromones = new float[num_cities * num_cities];
    for (int i = 0; i < num_cities * num_cities; i++) {
        h_initial_pheromones[i] = initial_pheromone;
    }
    CUDA_CHECK(cudaMemcpy(d_pheromones, h_initial_pheromones, num_cities * num_cities * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_initial_pheromones;
    
    // Set initial best length
    float initial_best = FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_best_length, &initial_best, sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate memory for ant tours
    AntTour *h_ant_tours = new AntTour[num_ants];
    AntTour *d_ant_tours;
    CUDA_CHECK(cudaMalloc(&d_ant_tours, num_ants * sizeof(AntTour)));
    
    for (int a = 0; a < num_ants; a++) {
        CUDA_CHECK(cudaMalloc(&h_ant_tours[a].visited_cities, num_cities * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_ant_tours[a].visited, num_cities * sizeof(int)));
    }
    
    CUDA_CHECK(cudaMemcpy(d_ant_tours, h_ant_tours, num_ants * sizeof(AntTour), cudaMemcpyHostToDevice));
    
    // Initialize RNG states
    initRNG<<<(num_ants + 255) / 256, 256>>>(d_rng_states, seed);
    
    // Create CUDA Graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    
    // Capture graph
    cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
    
    // Zero out visited array
    for (int a = 0; a < num_ants; a++) {
        CUDA_CHECK(cudaMemset(h_ant_tours[a].visited, 0, num_cities * sizeof(int)));
    }
    
    // Build tours
    buildAntTours<<<(num_ants + 255) / 256, 256>>>(d_rng_states, d_distances, d_pheromones, d_ant_tours);
    
    // Update pheromones
    updatePheromones<<<(num_cities * num_cities + 255) / 256, 256>>>(d_pheromones, d_ant_tours, num_ants);
    
    // Find best tour
    findBestTour<<<1, num_ants>>>(d_ant_tours, num_ants, d_best_tour, d_best_length);
    
    // End graph capture
    cudaStreamEndCapture(0, &graph);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    
    // Execute graph for specified iterations
    for (int iter = 0; iter < num_iterations; iter++) {
        cudaGraphLaunch(graph_exec, 0);
        cudaStreamSynchronize(0);
    }
    
    // Copy results back
    best_tour.resize(num_cities);
    int *h_best_tour = new int[num_cities];
    CUDA_CHECK(cudaMemcpy(h_best_tour, d_best_tour, num_cities * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&best_tour_length, d_best_length, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Convert to 1-indexed tour
    for (int i = 0; i < num_cities; i++) {
        best_tour[i] = h_best_tour[i];
    }
    
    // Add the return to starting city (for output format)
    best_tour.push_back(best_tour[0]);
    
    // Cleanup
    delete[] h_best_tour;
    
    for (int a = 0; a < num_ants; a++) {
        CUDA_CHECK(cudaFree(h_ant_tours[a].visited_cities));
        CUDA_CHECK(cudaFree(h_ant_tours[a].visited));
    }
    delete[] h_ant_tours;
    
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_pheromones));
    CUDA_CHECK(cudaFree(d_best_tour));
    CUDA_CHECK(cudaFree(d_best_length));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_ant_tours));
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);
}