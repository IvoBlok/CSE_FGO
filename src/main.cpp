#include <iostream>
#include <fstream>

#include "FGO.hpp"

int main(int argc, char** argv) {

    FGOParameters params;
    params.numberOfAtoms = 6;
    
    FuzzyGlobalOptimizer optimizer(params);
    std::mt19937 rng(42);
    optimizer.runSingle(rng);
    
    return 0;
}