// main.cu
#include <iostream>
#include <string>
#include "common.h"
#include "tsp_loader.h"
#include "worker_ant.h"
#include "queen_ant.h"

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <TYPE> <NUM_ITER> <ALPHA> <BETA> <EVAPORATE> <SEED>" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string type = argv[3];
    int num_iterations = std::stoi(argv[4]);
    float alpha = std::stof(argv[5]);
    float beta = std::stof(argv[6]);
    float evaporation_rate = std::stof(argv[7]);
    unsigned int seed = std::stoul(argv[8]);
    
    // Load TSP problem
    TSPInstance tsp;
    if (!loadTSPInstance(input_file, &tsp)) {
        std::cerr << "Failed to load TSP instance from " << input_file << std::endl;
        return 1;
    }
    
    // Initialize CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);
    
    // Run the appropriate ACO implementation
    std::vector<int> best_tour;
    float best_tour_length = 0.0f;
    
    if (type == "WORKER") {
        runWorkerAntACO(tsp, num_iterations, alpha, beta, evaporation_rate, seed, best_tour, best_tour_length);
    } else if (type == "QUEEN") {
        runQueenAntACO(tsp, num_iterations, alpha, beta, evaporation_rate, seed, best_tour, best_tour_length);
    } else {
        std::cerr << "Invalid TYPE: " << type << ". Must be WORKER or QUEEN." << std::endl;
        return 1;
    }
    
    // Output results
    std::ofstream out(output_file);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return 1;
    }
    
    out << best_tour_length << std::endl;
    for (size_t i = 0; i < best_tour.size(); i++) {
        out << best_tour[i];
        if (i < best_tour.size() - 1) out << " ";
    }
    out << std::endl;
    out.close();
    
    return 0;
}