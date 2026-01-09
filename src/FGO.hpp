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
    float gridSpacingSquared = gridSpacing * gridSpacing;
    int32_t cutoffDistance = ((int32_t)(2.1f / gridSpacing));
    int32_t cellDistance = ((int32_t)(1.9f / gridSpacing)); // recommended to be close to cutoffDistance (going smaller loses discrete energy accuracy, and even slightly larger can double the computational time). Ideally it is also divisable by 2

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

    float spawningRadiusFactor = 0.5f;

    size_t maxRealOptimizationIterations = 2000;
    float realOptimizationTolerance = 1e-6f;
};

struct SingleRunResult {
    std::vector<std::pair<DiscretePoints, float>> discCandidates;
    std::vector<std::pair<Cluster, float>> contCandidates;

    // optionally more, to be used for debugging / performance analysis
    std::chrono::microseconds totalTime{0};
    std::chrono::microseconds dmc1Time{0};
    std::chrono::microseconds dmc2Time{0};
    std::chrono::microseconds realOptTime{0};
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
        std::vector<std::pair<DiscretePoints, float>> discCandidates;
        std::vector<std::pair<Cluster, float>> contCandidates;
    };

public:
    explicit FuzzyGlobalOptimizer(const FGOParameters& params);

    SingleRunResult runSingle();
    SingleRunResult runSingle(std::mt19937& rng);
    SingleRunResult runSingleWithSeed(uint32_t seed);
    std::vector<SingleRunResult> runMultiple(size_t numRuns);

private: 
    std::chrono::microseconds initializeCluster(DiscreteCluster& cluster, std::mt19937& rng);
    std::chrono::microseconds runDMCLayer(RunState& state, const DiscreteCluster& startCluster, const FGOParameters::DMCParameters& dmcParams, std::mt19937& rng);
    std::chrono::microseconds runRealOptimization(RunState& state, const float acceptanceThreshold);

    // helper functions for the main algorithm steps above
    DiscreteCoord getPointInBall(std::mt19937& rng, float radius, DiscreteCoord center, bool allowZero = true);
    DiscreteCoord getPointOnSphere(std::mt19937& rng, float radius, DiscreteCoord center);

    void localDiscreteOptimization(DiscreteCluster& cluster);
    void localRealOptimization(std::pair<Cluster, float>& candidate);

    size_t localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t freeIndex);
    float localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t freeIndex, DiscreteCoord freePoint);
};

#endif // FUZZY_GLOBAL_OPTIMIZER_H