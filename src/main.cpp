#include <iostream>
#include <fstream>

#include "FGO.hpp"

int main(int argc, char** argv) {

    FGOParameters params;
    params.numberOfAtoms = 10;
    
    FuzzyGlobalOptimizer optimizer(params);

    for (size_t i = 0; i < 5; i++)
    {
        std::cout << "Attempt: " << i << "\n";
        std::mt19937 rng(i);
        optimizer.runSingle(rng);
    }
    

    
    return 0;
}