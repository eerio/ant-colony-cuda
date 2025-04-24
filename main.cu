#include <iostream>
#include <string>
#include <fstream>
#include <optional>
#include <curand_kernel.h>

enum Implementation {
    WORKER,
    QUEEN
};

// http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
enum ProblemType {
    TSP
};

enum EdgeWeightType {
    EUC_2D,
    CEIL_2D,
    GEO
};

struct Point2D {
    float x;
    float y;
};

struct ProblemData {
    ProblemType type;
    unsigned int dimension;
    EdgeWeightType edge_weight_type;
    Point2D *points;
};

std::optional<int> tryParseInt(const std::string &str, const std::string &name) {
    try {
        return std::stoi(str);
    } catch (const std::invalid_argument&) {
        std::cerr << "Invalid " << name << ": " << str << std::endl;
    } catch (const std::out_of_range&) {
        std::cerr << name << " out of range: " << str << std::endl;
    }
    return {};
}

std::optional<float> tryParseFloat(const std::string &str, const std::string &name) {
    try {
        return std::stof(str);
    } catch (const std::invalid_argument&) {
        std::cerr << "Invalid " << name << ": " << str << std::endl;
    } catch (const std::out_of_range&) {
        std::cerr << name << " out of range: " << str << std::endl;
    }
    return {};
}

int main(int argc, char *argv[]) {
    // Expected arguments:
    // ./acotsp <input_file> <output_file> <TYPE> <NUM_ITER> <ALPHA> <BETA> <EVAPORATE> <SEED>
    
    if (argc < 9) {
        std::cerr << "Usage: ./acotsp <input_file> <output_file> <TYPE> <NUM_ITER> <ALPHA> <BETA> <EVAPORATE> <SEED>\n";
        std::cerr << "  TYPE: WORKER or QUEEN\n";
        std::cerr << "  NUM_ITER: Number of iterations (default: number of cities)\n";
        std::cerr << "  ALPHA: relative influence of the pheromone trail (default: 1)\n";
        std::cerr << "  BETA: relative influence of the heuristic information (default: 2)\n";
        std::cerr << "  EVAPORATE: Pheromone evaporation rate; 0 < EVAPORATE <= 1 (default: 0.5)\n";
        std::cerr << "  SEED: Random seed (default: 42)\n";
        return 1;
    }

    // Parse required arguments
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    // try writing to output file to check if it's valid
    std::ofstream test_output(output_file);
    if (!test_output) {
        std::cerr << "Error: Unable to open output file for writing." << std::endl;
        return 1;
    }
    test_output.close();
    
    // Default values for optional arguments
    Implementation impl = WORKER;
    unsigned int num_iter = 0;
    float alpha = 1.0;
    float beta = 2.0;
    float evaporate = 0.5;
    unsigned int seed = 42;
    
    // Parse optional arguments if provided
    std::string type_str = argv[3];
    if (type_str == "WORKER") {
        impl = WORKER;
    } else if (type_str == "QUEEN") {
        impl = QUEEN;
    } else {
        std::cerr << "Invalid TYPE. Must be WORKER or QUEEN." << std::endl;
        return 1;
    }

    auto num_iter_opt = tryParseInt(argv[4], "NUM_ITER");
    if (!num_iter_opt) { return 1; }
    if (num_iter_opt.value() > 0) {
        num_iter = num_iter_opt.value();
    } else {
        std::cerr << "NUM_ITER must be greater than 0." << std::endl;
        return 1;
    }
    auto alpha_opt = tryParseFloat(argv[5], "ALPHA");
    if (!alpha_opt) { return 1; }
    alpha = alpha_opt.value();
    auto beta_opt = tryParseFloat(argv[6], "BETA");
    if (!beta_opt) { return 1; }
    beta = beta_opt.value();
    auto evaporate_opt = tryParseFloat(argv[7], "EVAPORATE");
    if (!evaporate_opt) { return 1; }
    evaporate = evaporate_opt.value();
    if (evaporate <= 0 || evaporate > 1) {
        std::cerr << "EVAPORATE must be in the range (0, 1]." << std::endl;
        return 1;
    }
    auto seed_opt = tryParseInt(argv[8], "SEED");
    if (!seed_opt) { return 1; }
    if (seed_opt.value() <= 0) {
        std::cerr << "SEED must be greater than 0." << std::endl;
        return 1;
    }
    seed = seed_opt.value();
    
    // Print parsed arguments to stdout
    std::cout << "Input file: " << input_file << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Problem type: " << (impl == WORKER ? "WORKER" : "QUEEN") << std::endl;
    std::cout << "Number of iterations: " << num_iter << std::endl;
    std::cout << "ALPHA: " << alpha << std::endl;
    std::cout << "BETA: " << beta << std::endl;
    std::cout << "EVAPORATE: " << evaporate << std::endl;
    std::cout << "SEED: " << seed << std::endl;
    
    // // Initialize CUDA random number generator
    // curandState* d_states;
    // cudaMalloc(&d_states, sizeof(curandState) * 1); // Assuming 1 thread for simplicity
    // curandGenerator_t gen;
    // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // curandSetPseudoRandomGeneratorSeed(gen, seed);
    
    // std::cout << "CURAND initialized with seed: " << seed << std::endl;
    
    // curandDestroyGenerator(gen);
    
    return 0;
}