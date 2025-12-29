#include <iostream>
#include <fstream>

#include "FGO.hpp"

int main(int argc, char** argv) {

    FGOParameters params;
    params.numberOfAtoms = 20;
    
    FuzzyGlobalOptimizer optimizer(params);

    auto cluster = DiscreteCluster(params.numberOfAtoms, params.cutoffDistance);
    for (size_t i = 0; i < 1; i++)
    {
        std::mt19937 rng(std::random_device{}());
        optimizer.initializeCluster(cluster, rng);
        std::cout << "start E:" << cluster.getClusterEnergy(params.gridSpacing * params.gridSpacing) << "\n";
        optimizer.localDiscreteOptimization(cluster);
        std::cout << "end E:" << cluster.getClusterEnergy(params.gridSpacing * params.gridSpacing) << "\n\n";
    }
    
    return 0;
}