// common.h
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cfloat>
#include <iostream>

// Problem instance data structure 
struct TSPInstance {
    int num_cities;                // Number of cities in the problem
    std::vector<float> x_coords;   // X coordinates of cities
    std::vector<float> y_coords;   // Y coordinates of cities
    float* distance_matrix;        // Flattened distance matrix for GPU
};

// ACO parameters structure to keep them organized
struct ACOParams {
    float alpha;              // Pheromone influence factor
    float beta;               // Distance influence factor
    float evaporation_rate;   // Pheromone evaporation rate
    int num_iterations;       // Number of iterations to run
    unsigned int seed;        // Random seed for reproducibility
};

// Implementation types
enum ACOImplementation {
    WORKER_ANT,  // 1 thread = 1 ant
    QUEEN_ANT    // 1 block = 1 ant, threads collaborate
};

// Constants for ACO algorithm
const float INITIAL_PHEROMONE = 0.1f;  // Initial pheromone value
const int MAX_CITIES = 1024;           // Maximum supported cities (limited by block size)

// Common utility functions
inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << ": " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Macro for error checking
#define CHECK_CUDA_ERROR(call) checkCudaError(call, __FILE__, __LINE__)

// Helper for logging debug information
inline void debugLog(const std::string& message, bool verbose = false) {
    if (verbose) {
        std::cout << "[DEBUG] " << message << std::endl;
    }
}

// Structure to hold performance metrics
struct ACOPerformanceMetrics {
    float execution_time_ms;      // Total execution time in milliseconds
    float best_tour_length;       // Length of best tour found
    int iterations_to_best;       // Iterations needed to find best tour
    float avg_kernel_time_ms;     // Average kernel execution time
    
    void print() const {
        std::cout << "----- Performance Metrics -----" << std::endl;
        std::cout << "Execution time: " << execution_time_ms << " ms" << std::endl;
        std::cout << "Best tour length: " << best_tour_length << std::endl;
        std::cout << "Iterations to best: " << iterations_to_best << std::endl;
        std::cout << "Avg kernel time: " << avg_kernel_time_ms << " ms" << std::endl;
        std::cout << "-----------------------------" << std::endl;
    }
};

// Device constant memory declarations (to be defined in each implementation file)
extern __constant__ int d_num_cities;
extern __constant__ float d_alpha;
extern __constant__ float d_beta;
extern __constant__ float d_evaporation_rate;

// Function to convert string implementation type to enum
inline ACOImplementation getImplementationType(const std::string& type_str) {
    if (type_str == "WORKER") {
        return WORKER_ANT;
    } else if (type_str == "QUEEN") {
        return QUEEN_ANT;
    } else {
        std::cerr << "Invalid implementation type: " << type_str << std::endl;
        std::cerr << "Valid types are WORKER or QUEEN" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Helper function to check valid parameter ranges
inline void validateACOParams(const ACOParams& params) {
    if (params.alpha < 0.0f || params.beta < 0.0f || 
        params.evaporation_rate < 0.0f || params.evaporation_rate > 1.0f ||
        params.num_iterations <= 0) {
        std::cerr << "Invalid ACO parameters. Please ensure:" << std::endl;
        std::cerr << "- alpha and beta are non-negative" << std::endl;
        std::cerr << "- evaporation_rate is between 0 and 1" << std::endl;
        std::cerr << "- num_iterations is positive" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Helper function to calculate Euclidean distance between two points
inline float calculateDistance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx*dx + dy*dy);
}

// Forward declarations of ACO implementation functions
void runWorkerAntACO(const TSPInstance& tsp, 
                    int num_iterations,
                    float alpha,
                    float beta,
                    float evaporation_rate,
                    unsigned int seed,
                    std::vector<int>& best_tour,
                    float& best_tour_length);

void runQueenAntACO(const TSPInstance& tsp, 
                   int num_iterations,
                   float alpha,
                   float beta,
                   float evaporation_rate,
                   unsigned int seed,
                   std::vector<int>& best_tour,
                   float& best_tour_length);