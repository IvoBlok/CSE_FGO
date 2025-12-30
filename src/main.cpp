#include <iostream>
#include <fstream>

#include "FGO.hpp"

int main(int argc, char** argv) {
    FGOParameters params;
    params.numberOfAtoms = std::stoi(argv[1]);
    
    FuzzyGlobalOptimizer optimizer(params);
    
    for (size_t i = 0; i < 50; i++)
    {
        std::mt19937 rng(std::random_device{}());
        DiscreteCluster cluster(params.numberOfAtoms, params.cutoffDistance);
        optimizer.initializeCluster(cluster, rng);
        optimizer.localDiscreteOptimization(cluster);
        std::cout << cluster.getClusterEnergy(params.gridSpacingSquared) << "\n";
    }
    
    return 0;
}