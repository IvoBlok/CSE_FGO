#include <iostream>
#include <fstream>

#include "FGO.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_atoms>\n";
        return 1;
    }

    FGOParameters params;
    params.numberOfAtoms = std::stoi(argv[1]);

    FuzzyGlobalOptimizer optimizer(params);

    for (size_t i = 0; i < 5; i++)
    {
        std::mt19937 rng(i);
        optimizer.runSingle(rng);
    }
    
    return 0;
}