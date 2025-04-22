// queen_ant.cu
#include "queen_ant.h"
#include "cuda_utils.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>

// Device constants for ACO parameters (same as in worker_ant.cu)
__constant__ float d_alpha;
__constant__ float d_beta;
__constant__ float d_evaporation_rate;
__constant__ int d_num_cities;

// Structure to represent an ant colony
struct AntColony {
    int* best_tour;          // Best tour found
    float* best_length;      // Length of best tour
    float* distances;        // Distance matrix
    float* pheromones;       // Pheromone matrix
    curandState* rng_states; // Random states
};

// Initialize RNG states for queen ant model
__global__ void initQueenRNG(curandState* states, unsigned int seed, int num_cities) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_cities) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Each block cooperatively builds one tour (One block = one ant)
__global__ void buildQueenAntTour(AntColony colony, 
                                 int* tour_cities,
                                 int* visited,
                                 float* tour_length,
                                 int start_city) {
    __shared__ int current_city;
    __shared__ float probabilities[1024]; // Max supported cities = 1024
    
    int tid = threadIdx.x; // Thread ID = city ID
    
    // Initialize for first city
    if (tid == 0) {
        current_city = start_city;
        tour_cities[0] = current_city;
        visited[current_city] = 1;
        *tour_length = 0.0f;
    }
    __syncthreads();
    
    // Build tour step by step
    for (int step = 1; step < d_num_cities; step++) {
        // Each thread handles probability calculation for one potential next city
        float prob = 0.0f;
        
        if (tid < d_num_cities && tid != current_city && visited[tid] == 0) {
            float distance = colony.distances[current_city * d_num_cities + tid];
            float pheromone = colony.pheromones[current_city * d_num_cities + tid];
            
            // Equation (1) from the paper
            if (distance > 0) {
                prob = powf(pheromone, d_alpha) * powf(1.0f / distance, d_beta);
            }
        }
        
        probabilities[tid] = prob;
        __syncthreads();
        
        // Parallel reduction to compute total probability
        __shared__ float total_prob;
        
        if (tid == 0) {
            total_prob = 0.0f;
            for (int i = 0; i < d_num_cities; i++) {
                total_prob += probabilities[i];
            }
        }
        __syncthreads();
        
        // Roulette wheel selection (thread 0 manages the selection)
        if (tid == 0) {
            float random = curand_uniform(&colony.rng_states[blockIdx.x]) * total_prob;
            float running_sum = 0.0f;
            int next_city = -1;
            
            for (int i = 0; i < d_num_cities; i++) {
                if (!visited[i]) {
                    running_sum += probabilities[i];
                    if (running_sum >= random) {
                        next_city = i;
                        break;
                    }
                }
            }
            
            // Fallback in case of numerical issues
            if (next_city == -1) {
                for (int i = 0; i < d_num_cities; i++) {
                    if (!visited[i]) {
                        next_city = i;
                        break;
                    }
                }
            }
            
            // Add city to tour
            tour_cities[step] = next_city;
            visited[next_city] = 1;
            *tour_length += colony.distances[current_city * d_num_cities + next_city];
            current_city = next_city;
        }
        __syncthreads();
    }
    
    // Complete the tour by returning to start
    if (tid == 0) {
        int first_city = tour_cities[0];
        *tour_length += colony.distances[current_city * d_num_cities + first_city];
        
        // Update best tour if this is better
        if (*tour_length < *colony.best_length) {
            atomicExch(colony.best_length, *tour_length);
            for (int i = 0; i < d_num_cities; i++) {
                colony.best_tour[i] = tour_cities[i] + 1;  // Convert to 1-indexed
            }
        }
    }
}

// Kernel to update pheromones based on best ant tour
__global__ void updateQueenPheromones(AntColony colony, int* tour_cities, float tour_length) {
    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_id >= d_num_cities * d_num_cities) return;
    
    int i = edge_id / d_num_cities;
    int j = edge_id % d_num_cities;
    
    if (i == j) {
        colony.pheromones[edge_id] = 0.0f;
        return;
    }
    
    // Evaporation
    colony.pheromones[edge_id] *= (1.0f - d_evaporation_rate);
    
    // Check if edge (i,j) is in this ant's tour
    for (int step = 0; step < d_num_cities; step++) {
        int city1 = tour_cities[step];
        int city2 = tour_cities[(step + 1) % d_num_cities];
        
        if ((city1 == i && city2 == j) || (city1 == j && city2 == i)) {
            // Add new pheromone
            colony.pheromones[edge_id] += 1.0f / tour_length;
            break;
        }
    }
}

void runQueenAntACO(const TSPInstance& tsp, 
                  int num_iterations,
                  float alpha,
                  float beta,
                  float evaporation_rate,
                  unsigned int seed,
                  std::vector<int>& best_tour,
                  float& best_tour_length) {
    int num_cities = tsp.num_cities;
    int num_ants = num_cities; // Number of ants = number of blocks
    
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
    // queen_ant.cu continued...

    // Set initial best length
    float initial_best = FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_best_length, &initial_best, sizeof(float), cudaMemcpyHostToDevice));
    
    // Create AntColony structure
    AntColony h_colony;
    h_colony.best_tour = d_best_tour;
    h_colony.best_length = d_best_length;
    h_colony.distances = d_distances;
    h_colony.pheromones = d_pheromones;
    h_colony.rng_states = d_rng_states;
    
    // Allocate temporary tour memory for each ant
    int **d_tours = new int*[num_ants];
    int **d_visited = new int*[num_ants];
    float **d_tour_lengths = new float*[num_ants];
    
    for (int a = 0; a < num_ants; a++) {
        CUDA_CHECK(cudaMalloc(&d_tours[a], num_cities * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_visited[a], num_cities * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tour_lengths[a], sizeof(float)));
    }
    
    // Initialize RNG states
    initQueenRNG<<<(num_cities + 255) / 256, 256>>>(d_rng_states, seed, num_cities);
    
    // Create CUDA Graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    
    // Capture graph
    cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
    
    for (int ant = 0; ant < num_ants; ant++) {
        // Zero out visited array
        CUDA_CHECK(cudaMemset(d_visited[ant], 0, num_cities * sizeof(int)));
        
        // Each block = one ant, each thread = one city
        buildQueenAntTour<<<1, num_cities>>>(h_colony, d_tours[ant], d_visited[ant], d_tour_lengths[ant], ant % num_cities);
        
        // Update pheromones based on tour
        updateQueenPheromones<<<(num_cities * num_cities + 255) / 256, 256>>>(h_colony, d_tours[ant], *d_tour_lengths[ant]);
    }
    
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
        CUDA_CHECK(cudaFree(d_tours[a]));
        CUDA_CHECK(cudaFree(d_visited[a]));
        CUDA_CHECK(cudaFree(d_tour_lengths[a]));
    }
    
    delete[] d_tours;
    delete[] d_visited;
    delete[] d_tour_lengths;
    
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_pheromones));
    CUDA_CHECK(cudaFree(d_best_tour));
    CUDA_CHECK(cudaFree(d_best_length));
    CUDA_CHECK(cudaFree(d_rng_states));
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);
}