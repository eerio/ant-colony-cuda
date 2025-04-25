#include "tsp.h"

TspResult solveTSPWorker(
    const TspInput &tsp_input,
    unsigned int num_iter,
    float alpha,
    float beta,
    float evaporate,
    unsigned int seed
);