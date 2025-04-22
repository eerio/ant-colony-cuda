// tsp_loader.h
#pragma once
#include <vector>
#include <string>

struct TSPInstance {
    int num_cities;
    std::vector<float> x_coords;
    std::vector<float> y_coords;
    float* distance_matrix;  // Flattened distance matrix (for GPU)
};

bool loadTSPInstance(const std::string& filename, TSPInstance* tsp);