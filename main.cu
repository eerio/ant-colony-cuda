#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
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

std::optional<ProblemData> readProblemData(const std::string &filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return {};
    }

    ProblemData data;
    std::string line;
    bool in_node_coord_section = false;

    // Read the problem type
    while (std::getline(file, line)) {
        if (line.find("DISPLAY_DATA_TYPE") != std::string::npos) {
            continue;
        }
        else if (line.find("NAME") != std::string::npos) {
            continue;
        }
        else if (line.find("COMMENT") != std::string::npos) {
            continue;
        } 
        else if (line.find("EDGE_WEIGHT_TYPE") != std::string::npos) {
            if (line.find("EUC_2D") != std::string::npos) {
                data.edge_weight_type = EUC_2D;
            } else if (line.find("CEIL_2D") != std::string::npos) {
                data.edge_weight_type = CEIL_2D;
            } else if (line.find("GEO") != std::string::npos) {
                data.edge_weight_type = GEO;
            } else {
                std::cerr << "Unsupported edge weight type: " << line << std::endl;
                return {};
            }
        }
        else if (line.find("TYPE") != std::string::npos) {
            if (line.find("TSP") != std::string::npos) {
                data.type = TSP;
            } else {
                std::cerr << "Unsupported problem type: " << line << std::endl;
                return {};
            }
        }
        // Read the dimension
        else if (line.find("DIMENSION") != std::string::npos) {
            data.dimension = std::stoi(line.substr(line.find(":") + 1));
        }
        else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            in_node_coord_section = true;
            break; // Start reading points
        }
    }

    if (!in_node_coord_section) {
        std::cerr << "NODE_COORD_SECTION not found in file." << std::endl;
        return {};
    }

    // Allocate memory for points
    data.points = new Point2D[data.dimension];

    // Read the points
    for (unsigned int i = 0; i < data.dimension; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Error reading points from file." << std::endl;
            delete[] data.points;
            return {};
        }
        std::istringstream iss(line);
        unsigned int index;
        iss >> index >> data.points[i].x >> data.points[i].y;
        if (iss.fail()) {
            std::cerr << "Error parsing point data: " << line << std::endl;
            delete[] data.points;
            return {};
        }
        if (index != i + 1) {
            std::cerr << "Point index mismatch: expected " << (i + 1) << ", got " << index << std::endl;
            delete[] data.points;
            return {};
        }
    }
    // Check for end of section
    if (!std::getline(file, line) || line != "EOF") {
        std::cerr << "EOF not found after points." << std::endl;
        delete[] data.points;
        return {};
    }

    return data;
}

struct TspInput {
    unsigned int dimension;
    float *distances;
};

TspInput convertToTspInput(const ProblemData &problem_data) {
    TspInput tsp_input;
    tsp_input.dimension = problem_data.dimension;
    tsp_input.distances = new float[tsp_input.dimension * tsp_input.dimension];
    
    for (unsigned int i = 0; i < tsp_input.dimension; ++i) {
        for (unsigned int j = 0; j < tsp_input.dimension; ++j) {
            if (i == j) {
                tsp_input.distances[i * tsp_input.dimension + j] = 0;
            } else {
                float dx = problem_data.points[i].x - problem_data.points[j].x;
                float dy = problem_data.points[i].y - problem_data.points[j].y;
                tsp_input.distances[i * tsp_input.dimension + j] = sqrt(dx * dx + dy * dy);
            }
        }
    }
    return tsp_input;
}

struct TspResult {
    unsigned int dimension;
    float cost;
    unsigned int *tour;
};

float dist(const TspInput &tsp_input, unsigned int i, unsigned int j) {
    return tsp_input.distances[i * tsp_input.dimension + j];
}

TspResult solveTSP(const TspInput &tsp_input, unsigned int num_iter, float alpha, float beta, float evaporate) {
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
    // std::cout << "Input file: " << input_file << std::endl;
    // std::cout << "Output file: " << output_file << std::endl;
    // std::cout << "Problem type: " << (impl == WORKER ? "WORKER" : "QUEEN") << std::endl;
    // std::cout << "Number of iterations: " << num_iter << std::endl;
    // std::cout << "ALPHA: " << alpha << std::endl;
    // std::cout << "BETA: " << beta << std::endl;
    // std::cout << "EVAPORATE: " << evaporate << std::endl;
    // std::cout << "SEED: " << seed << std::endl;

    // Read problem data from file
    auto problem_data_opt = readProblemData(input_file);
    if (!problem_data_opt) { return 1; }
    ProblemData problem_data = problem_data_opt.value();

    TspInput tsp_input = convertToTspInput(problem_data);

    // Print problem data
    // std::cout << "Problem dimension: " << problem_data.dimension << std::endl;
    // std::cout << "Edge weight type: " << (problem_data.edge_weight_type == EUC_2D ? "EUC_2D" :
    //                                       problem_data.edge_weight_type == CEIL_2D ? "CEIL_2D" : "GEO") << std::endl;
    // std::cout << "Points:" << std::endl;
    // for (unsigned int i = 0; i < problem_data.dimension; ++i) {
    //     std::cout << "Point " << (i + 1) << ": (" << problem_data.points[i].x << ", " << problem_data.points[i].y << ")" << std::endl;
    // }

    // Solve TSP
    TspResult result = solveTSP(tsp_input, num_iter, alpha, beta, evaporate);
    if (result.dimension == 0) {
        std::cerr << "Error: TSP solver failed: dimension == 0" << std::endl;
        delete[] tsp_input.distances;
        delete[] result.tour;
        delete[] problem_data.points;
        return 1;
    }
    // std::cout << "TSP Result:" << std::endl;
    // std::cout << "Tour: ";
    // for (unsigned int i = 0; i < result.dimension; ++i) {
    //     std::cout << result.tour[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Cost: " << result.cost << std::endl;
    
    // Write result to output file
    std::ofstream output(output_file);
    if (!output) {
        std::cerr << "Error: Unable to open output file for writing." << std::endl;
        delete[] tsp_input.distances;
        delete[] result.tour;
        delete[] problem_data.points;
        return 1;
    }
    
    output << result.cost << std::endl;
    if (result.dimension == 1) {
        output << result.tour[0] + 1 << std::endl;
        output.close();
        delete[] tsp_input.distances;
        delete[] result.tour;
        delete[] problem_data.points;
        return 0;
    }

    for (unsigned int i = 0; i < result.dimension - 1; ++i) {
        output << result.tour[i] + 1 << " ";
    }
    output << result.tour[result.dimension - 1] + 1 << std::endl;
    output.close();

    // // Initialize CUDA random number generator
    // curandState* d_states;
    // cudaMalloc(&d_states, sizeof(curandState) * 1); // Assuming 1 thread for simplicity
    // curandGenerator_t gen;
    // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // curandSetPseudoRandomGeneratorSeed(gen, seed);
    
    // std::cout << "CURAND initialized with seed: " << seed << std::endl;
    
    // curandDestroyGenerator(gen);

    // Clean up allocated memory
    delete[] problem_data.points;
    delete[] tsp_input.distances;
    delete[] result.tour;
    problem_data.points = nullptr;
    
    return 0;
}