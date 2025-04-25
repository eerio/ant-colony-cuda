#pragma once
struct TspInput {
    unsigned int dimension;
    float *distances;
};

struct TspResult {
    unsigned int dimension;
    float cost;
    unsigned int *tour;
};