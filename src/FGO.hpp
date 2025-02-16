#ifndef FUZZY_GLOBAL_OPTIMIZER_H
#define FUZZY_GLOBAL_OPTIMIZER_H

#include "dataStructures.hpp"

#include <iostream>
#include <cmath>
#include <random>
#include <set>
#include <vector>
#include <string>

class FuzzyGlobalOptimizer {
public:
    int numberOfAtoms;
    float discreteGridSteps;
    float discreteCutoffDistance;

    float lowestEnergyFound;

    DiscreteCluster currentCluster;
    std::vector<DiscreteCluster> candidateClusters;

private:
    float spawningRadius;
    int discreteGridPointCount;
    int maxGridSquaredDistance;
    float realLOGradientStep;

public:
    FuzzyGlobalOptimizer(int numberOfAtoms, float discreteGridSteps = 0.02f, float discreteCutoffDistance = 2.1f, float realLOGradientStep = 0.001f);

    void runFGO();

private: 
    void localDiscreteOptimization(DiscreteCluster& cluster);

    int localDiscreteFrozenOptimization(DiscreteCluster& cluster, int nonFrozenAtom);
    int localDiscreteFrozenOptimization(DiscreteCluster& cluster, int nonFrozenAtom, std::vector<int>& neighbours);

    void localRealOptimization(ContinuousCluster& cluster);

    void discreteMonteCarlo(float activeEnergy, float targetEnergy, float targetSigma, float acceptanceEnergy, float convergenceFactor);

    DiscretePoint generateUniformRandomPointInSphere(float radius, DiscretePoint center, bool allowZero);

    void setAtomInRandomSphere(DiscreteCluster& cluster, int atomIndex, float radius, DiscretePoint center, bool allowZero);

    void generateInitialCluster(DiscreteCluster& cluster, float radius);

    int getRandomAtomByWeights(std::vector<float>& atomWeights);
};

#endif // FUZZY_GLOBAL_OPTIMIZER_H