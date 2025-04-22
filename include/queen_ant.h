// queen_ant.h
#pragma once
#include "common.h"
#include <vector>

void runQueenAntACO(const TSPInstance& tsp, 
                    int num_iterations,
                    float alpha,
                    float beta,
                    float evaporation_rate,
                    unsigned int seed,
                    std::vector<int>& best_tour,
                    float& best_tour_length);