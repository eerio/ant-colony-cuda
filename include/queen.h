#include "tsp.h"

TspResult solveTSPQueen(
    const TspInput &tsp_input,
    unsigned int num_iter,
    float alpha,
    float beta,
    float evaporate,
    unsigned int seed
);