#include <iostream>
#include <fstream>

#include "FGO.hpp"

int main(int argc, char** argv) {

    FGOParameters params;
    params.numberOfAtoms = 4;
    
    FuzzyGlobalOptimizer optimizer(params);

    for (size_t i = 0; i < 5; i++)
    {
        std::cout << "\nAttempt: " << i << "\n";
        std::mt19937 rng(i);
        optimizer.runSingle(rng);
    }
    

    
    return 0;
}