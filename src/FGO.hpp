#ifndef FUZZY_GLOBAL_OPTIMIZER_H
#define FUZZY_GLOBAL_OPTIMIZER_H

#include "dataStructures.hpp"
#include "discreteDistribution.hpp"

#include <iostream>
#include <cmath>
#include <random>
#include <set>
#include <vector>
#include <string>
#include <vector>

# define M_PI           3.14159265358979323846

struct FGOParameters {
    size_t numberOfAtoms;

    float discreteGridSteps = 0.02f;
    float discreteCutoffDistance = 2.1f;

    float gradientStepSize = 0.001f;

    struct DMCParameters {
        float activeEnergy;
        float targetEnergy;
        float targetSigma;
        float acceptanceEnergy;
        float convergenceFactor;
    };

    DMCParameters dmcLayer1{1.0f, -4.1f, 1.25f, 0.4f, 2.5f};
    DMCParameters dmcLayer2{1.0f, -11.0f, 1.3f, 0.3f, 1.5f};

    float spawningRadiusFactor = 0.4f;

    size_t maxRealOptimizationIterations = 1000;
    float realOptimizationTolerance = 1e-6f;
};

struct SingleRunResult {
    ContinuousCluster bestCluster;
    float bestEnergy = std::numeric_limits<float>::max();

    std::vector<DiscreteCluster> discreteCandidates;
    std::vector<ContinuousCluster> continuousCandidates;

    // optionally more, to be used for debugging / performance analysis
};

struct MultiRunResult {
    ContinuousCluster globalBestCluster;
    float globalBestEnergy = std::numeric_limits<float>::max();

    std::vector<SingleRunResult> allRuns;

    // optionally more, for debug / performance analysis or just statistics across the runs
};


class FuzzyGlobalOptimizer {
private: 
    FGOParameters params;
    DiscreteDistribution atomSelector;
    std::mt19937 rng;

    std::uniform_real_distribution<float> uniformDist{0.0f, 1.0f};
    std::uniform_real_distribution<float> thetaDist{0.0f, 2.0f * M_PI};
    std::uniform_real_distribution<float> phiDist{0.0f, M_PI};

    struct RunState {
        DiscreteCluster currentDiscrete;
        std::vector<DiscreteCluster> discreteCandidates;
        std::vector<ContinuousCluster> continuousCandidates;

        float bestEnergy = std::numeric_limits<float>::max();
        size_t bestDiscreteIndex = static_cast<size_t>(-1);
    };

public:
    explicit FuzzyGlobalOptimizer(const FGOParameters& params);
    explicit FuzzyGlobalOptimizer(FGOParameters&& params);

    SingleRunResult runSingle();
    SingleRunResult runSingle(std::mt19937& rng);
    SingleRunResult runSingleWithSeed(uint32_t seed);

    MultiRunResult runMultiple(size_t numRuns = 0);

private: 
    void initializeCluster(DiscreteCluster& cluster, std::mt19937& rng);

    void localDiscreteOptimization(DiscreteCluster& cluster);

    void runDMCLayer(RunState& state, 
                    const FGOParameters::DMCParameters& dmcParams,
                    std::mt19937& rng);
    
    void localRealOptimization(ContinuousCluster& cluster);

    // helper functions for the main algorithm steps above
    DiscretePoint getPointInSphere(std::mt19937& rng, const float radius, const DiscretePoint& center, const bool allowZero = true);

    size_t localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t frozenIndex);
};

/*

    void localRealOptimization(ContinuousCluster& cluster);

    void discreteMonteCarlo(float activeEnergy, float targetEnergy, float targetSigma, float acceptanceEnergy, float convergenceFactor);

    DiscretePoint generateUniformRandomPointInSphere(float radius, DiscretePoint center, bool allowZero);

    void setAtomInRandomSphere(DiscreteCluster& cluster, int atomIndex, float radius, DiscretePoint center, bool allowZero);

    void generateInitialCluster(DiscreteCluster& cluster, float radius);

    int getRandomAtomByWeights(std::vector<float>& atomWeights);
};
*/

#endif // FUZZY_GLOBAL_OPTIMIZER_H