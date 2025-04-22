// tsp_loader.cu
#include "tsp_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

bool loadTSPInstance(const std::string& filename, TSPInstance* tsp) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int dimension = 0;
    bool reading_coords = false;
    
    // Parse metadata
    while (getline(file, line)) {
        if (line.find("DIMENSION") != std::string::npos) {
            std::istringstream iss(line.substr(line.find(":") + 1));
            iss >> dimension;
        } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            reading_coords = true;
            break;
        }
    }
    
    if (dimension <= 0) {
        std::cerr << "Invalid or missing DIMENSION" << std::endl;
        return false;
    }
    
    tsp->num_cities = dimension;
    tsp->x_coords.resize(dimension);
    tsp->y_coords.resize(dimension);
    
    // Read city coordinates
    for (int i = 0; i < dimension && reading_coords; i++) {
        if (!getline(file, line) || line.find("EOF") != std::string::npos) {
            break;
        }
        
        int id;
        float x, y;
        std::istringstream iss(line);
        iss >> id >> x >> y;
        
        if (id < 1 || id > dimension) {
            std::cerr << "Invalid city ID: " << id << std::endl;
            return false;
        }
        
        tsp->x_coords[id-1] = x;  // Convert 1-indexed to 0-indexed
        tsp->y_coords[id-1] = y;
    }
    
    // Calculate distance matrix on CPU
    cudaMallocHost(&tsp->distance_matrix, dimension * dimension * sizeof(float));
    
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            if (i == j) {
                tsp->distance_matrix[i * dimension + j] = 0.0f;
            } else {
                float dx = tsp->x_coords[i] - tsp->x_coords[j];
                float dy = tsp->y_coords[i] - tsp->y_coords[j];
                tsp->distance_matrix[i * dimension + j] = sqrtf(dx*dx + dy*dy);
            }
        }
    }
    
    return true;
}