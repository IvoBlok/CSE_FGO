#ifndef FUZZY_GLOBAL_OPTIMIZER_H
#define FUZZY_GLOBAL_OPTIMIZER_H

#include "dataStructures.hpp"

#include <iostream>
#include <cmath>
#include <random>
#include <set>
#include <vector>
#include <string>
#include <vector>
#include <chrono>

# define M_PI           3.14159265358979323846

struct FGOParameters {
    size_t numberOfAtoms;

    float gridSpacing = 0.02f;
    int32_t cutoffDistance = ((int32_t)(2.1f / gridSpacing));

    float gradientStepSize = 0.001f;

    struct DMCParameters {
        float invActiveEnergy;
        float targetEnergy;
        float inv2Sigma2;
        float invAcceptanceEnergy;
        float convergenceFactor;

        DMCParameters(float activeEnergy, float targetEnergy, float targetSigma, float acceptanceEnergy, float convergenceFactor)
         :  invActiveEnergy(1.0f / activeEnergy),
            targetEnergy(targetEnergy),
            inv2Sigma2(-0.5f / (targetSigma * targetSigma)),
            invAcceptanceEnergy(1.0f / acceptanceEnergy),
            convergenceFactor(convergenceFactor) {}
    };
    DMCParameters dmcLayer1 = DMCParameters(1.0f, -4.1f, 1.25f, 0.4f, 2.5f);
    DMCParameters dmcLayer2 = DMCParameters(1.0f, -11.0f, 1.3f, 0.3f, 1.5f);

    float spawningRadiusFactor = 0.4f;

    size_t maxRealOptimizationIterations = 2000;
    float realOptimizationTolerance = 1e-6f;
};

struct SingleRunResult {
    Cluster bestCluster;
    float bestEnergy = std::numeric_limits<float>::max();

    std::vector<std::pair<Cluster, float>> candidates;

    // optionally more, to be used for debugging / performance analysis
    std::chrono::microseconds totalTime{0};
    std::chrono::microseconds dmc1Time{0};
    std::chrono::microseconds dmc2Time{0};
    std::chrono::microseconds realOptTime{0};
};

struct MultiRunResult {
    std::vector<SingleRunResult> allRuns;

    // optionally more, for debug / performance analysis or just statistics across the runs
};


class FuzzyGlobalOptimizer {
private: 
    FGOParameters params;
    std::mt19937 rng;

    std::uniform_real_distribution<float> uniformDist{0.0f, 1.0f};
    std::uniform_real_distribution<float> thetaDist{0.0f, 2.0f * M_PI};
    std::uniform_real_distribution<float> phiDist{0.0f, M_PI};

    alignas(64) std::vector<float> lookup;

    struct RunState {
        std::vector<std::pair<Cluster, float>> candidates;
        size_t bestIndex = 0;

        Cluster DMCWalker;
    };

    RealLJCalculator fastLJ;

public:
    explicit FuzzyGlobalOptimizer(const FGOParameters& params);

    SingleRunResult runSingle();
    SingleRunResult runSingle(std::mt19937& rng);
    SingleRunResult runSingleWithSeed(uint32_t seed);

    MultiRunResult runMultiple(size_t numRuns);

//private: 
    void initializeCluster(DiscreteCluster& cluster, std::mt19937& rng);
    void runDMCLayer(RunState& state, const FGOParameters::DMCParameters& dmcParams, std::mt19937& rng);
    void localDiscreteOptimization(DiscreteCluster& cluster);
    void localRealOptimization(std::pair<Cluster, float>& candidate);

    // helper functions for the main algorithm steps above
    void setPointInBall(DiscreteCluster& cluster, int spacingMultiple, size_t index, std::mt19937& rng, float radius, int16_t cx, int16_t cy, int16_t cz, bool allowZero = true);
    void setPointOnSphere(DiscreteCluster& cluster, int spacingMultiple, size_t index, std::mt19937& rng, float radius, int16_t cx, int16_t cy, int16_t cz);

    size_t localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t frozenIndex, const std::vector<uint64_t>& neighbours);
};

#endif // FUZZY_GLOBAL_OPTIMIZER_H