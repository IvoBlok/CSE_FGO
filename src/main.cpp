#include <iostream>
#include <fstream>

#include "FGO.hpp"

int main(int argc, char** argv) {

    FGOParameters params;
    params.numberOfAtoms = 20;
    
    FuzzyGlobalOptimizer optimizer(params);

    auto cluster = DiscreteCluster(params.numberOfAtoms, params.cutoffDistance);
    for (size_t i = 0; i < 100000; i++)
    {
        std::mt19937 rng(std::random_device{}());
        optimizer.initializeCluster(cluster, rng);
        optimizer.localDiscreteOptimization(cluster);

        std::cout << ".";
    }
    
    return 0;
}