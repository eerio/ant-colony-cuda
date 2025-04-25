#include "tsp.h"

float dist(const TspInput &tsp_input, unsigned int i, unsigned int j) {
    return tsp_input.distances[i * tsp_input.dimension + j];
}

TspResult solveTSPBaseline(const TspInput &tsp_input, unsigned int num_iter, float alpha, float beta, float evaporate, unsigned int seed) {
    TspResult result;
    result.dimension = tsp_input.dimension;
    result.cost = 0.0f;
    result.tour = new unsigned int[result.dimension];

    // baseline
    result.tour[0] = 0;
    for (unsigned int i = 1; i < result.dimension; ++i) {
        result.tour[i] = i;
        result.cost += dist(tsp_input, i - 1, i);
    }
    result.cost += dist(tsp_input, result.dimension - 1, 0); // return to start

    return result;
}