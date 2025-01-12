#include <iostream>
#include "FGO.hpp"

int main() {
    float clusterBestEnergies[10] = {-1.f, -6.f, -16.505f, -44.326f, -72.659f, -86.809f, -108.315f, -190.533f, -543.66f};
    int clusterSizes[10] = {2, 4, 7, 13, 19, 22, 26, 41, 98};

    int i = 8;

    ContinuousCluster bestCluster(clusterSizes[i]);

    FuzzyGlobalOptimizer FGO(clusterSizes[i]);
    FGO.runFGO();

    std::cout << "N = " << clusterSizes[i] << " FEnergy: " << FGO.lowestEnergyFound << " DeltaEnergy: " << FGO.lowestEnergyFound - clusterBestEnergies[i] << "\n";
}