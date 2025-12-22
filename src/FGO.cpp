#include "FGO.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>
#include <list>

# define M_PI           3.14159265358979323846  /* pi */

FuzzyGlobalOptimizer::FuzzyGlobalOptimizer(const FGOParameters& params)
    : params(params), atomSelector(DiscreteDistribution(params.numberOfAtoms)), rng(std::random_device{}()) {}

FuzzyGlobalOptimizer::FuzzyGlobalOptimizer(FGOParameters&& params)
    : params(std::move(params)), atomSelector(DiscreteDistribution(params.numberOfAtoms)), rng(std::random_device{}()) {}

SingleRunResult FuzzyGlobalOptimizer::runSingle() {
    return runSingle(rng);
}

SingleRunResult FuzzyGlobalOptimizer::runSingleWithSeed(uint32_t seed) {
    std::mt19937 seededRng(seed);
    return runSingle(seededRng);
}

SingleRunResult FuzzyGlobalOptimizer::runSingle(std::mt19937& rng) {
    RunState state;
    DiscreteCluster& cluster = state.discreteCandidates.emplace_back(DiscreteCluster());
    state.bestDiscreteIndex = 0;

    initializeCluster(cluster, rng);
    localDiscreteOptimization(cluster);
    state.bestEnergy = cluster.getClusterEnergy();

    runDMCLayer(state, params.dmcLayer1, rng);

    // TODO DMC2, SMC

    SingleRunResult result;
    result.bestCluster = ContinuousCluster(state.discreteCandidates[state.bestDiscreteIndex], params.discreteGridSteps);
    result.bestEnergy = state.bestEnergy;
    result.discreteCandidates = std::move(state.discreteCandidates);
    result.continuousCandidates = std::move(state.continuousCandidates);
    
    return result;
}

MultiRunResult FuzzyGlobalOptimizer::runMultiple(size_t numRuns) {
    MultiRunResult multiResult;
    multiResult.allRuns.reserve(numRuns);

    for (size_t i = 0; i < numRuns; i++)
    {
        std::mt19937 runRng(std::random_device{}());
        auto singleResult = runSingle(runRng);
        multiResult.allRuns.emplace_back(std::move(singleResult));

        if (singleResult.bestEnergy < multiResult.globalBestEnergy) {
            multiResult.globalBestEnergy = singleResult.bestEnergy;
            multiResult.globalBestCluster = singleResult.bestCluster;
        }
    }
    
    return multiResult;
}


// private functions
// ===============================================
void FuzzyGlobalOptimizer::initializeCluster(DiscreteCluster& cluster, std::mt19937& rng) {
    cluster = DiscreteCluster(params.discreteGridSteps * params.discreteGridSteps, params.numberOfAtoms);

    float spawningRadius = params.spawningRadiusFactor * std::pow(params.numberOfAtoms, 0.33f);

    for (auto& point : cluster.points) {
        point = getPointInSphere(rng, spawningRadius, DiscretePoint(0.f));
    }
}

void FuzzyGlobalOptimizer::localDiscreteOptimization(DiscreteCluster& cluster) {
    std::list<size_t> activeList;
    for (size_t i = 0; i < cluster.size(); i++)
        activeList.emplace_back(i);

    while (!activeList.empty())
        activeList.remove_if([&cluster, this](int i){ return localDiscreteFrozenOptimization(cluster, i) == 0; });
}

void FuzzyGlobalOptimizer::runDMCLayer(RunState& state, const FGOParameters::DMCParameters& dmcParams, std::mt19937& rng) {
    size_t stepsSinceImprovement = 0;

    DiscreteCluster candidate{params.discreteGridSteps * params.discreteGridSteps, params.numberOfAtoms};

    std::vector<float> atomEnergies(params.numberOfAtoms);
    std::vector<float> activeWeights(params.numberOfAtoms);
    std::vector<float> targetWeights(params.numberOfAtoms);

    while (stepsSinceImprovement < (size_t)(params.numberOfAtoms * params.numberOfAtoms * dmcParams.convergenceFactor)) {
        const DiscreteCluster& currentCluster = state.discreteCandidates.back();

        for (size_t i = 0; i < params.numberOfAtoms; i++)
        {
            atomEnergies[i] = currentCluster.getAtomEnergy(i);

            activeWeights[i] = std::exp(atomEnergies[i] / dmcParams.activeEnergy);
            targetWeights[i] = std::exp(-0.5 * std::pow(atomEnergies[i] - dmcParams.targetEnergy, 2) / std::pow(dmcParams.targetSigma, 2));
        }
        
        atomSelector.updateDistribution(activeWeights);
        size_t activeAtom = atomSelector.generate(rng);
        atomSelector.updateDistribution(targetWeights);
        size_t targetAtom = atomSelector.generate(rng);

        currentCluster.copyTo(candidate);
        candidate.getPoint(activeAtom) = getPointInSphere(rng, 1.0f, candidate.getPoint(targetAtom), false);

        localDiscreteFrozenOptimization(candidate, activeAtom);

        float deltaAtomEnergy = candidate.getAtomEnergy(activeAtom) - atomEnergies[activeAtom];

        stepsSinceImprovement++;
        float acceptanceThreshold = uniformDist(rng);
        if (deltaAtomEnergy < 0.0f || acceptanceThreshold < std::exp(-deltaAtomEnergy / dmcParams.acceptanceEnergy)) {
            localDiscreteOptimization(candidate);

            float candidateEnergy = candidate.getClusterEnergy();
            if(candidateEnergy < state.bestEnergy) {
                state.bestEnergy = candidateEnergy;
                state.bestDiscreteIndex = state.discreteCandidates.size();
                state.discreteCandidates.emplace_back(candidate);
                stepsSinceImprovement = 0;
            }
        }
    }
}

// helper functions
DiscretePoint FuzzyGlobalOptimizer::getPointInSphere(std::mt19937& rng, const float radius, const DiscretePoint& center, const bool allowZero) {
    float u, theta, phi, r;
    int x, y, z;

    // generate random spherical coordinates
    u = uniformDist(rng);
    r = radius * std::cbrt(u);
    theta = thetaDist(rng);
    phi = phiDist(rng);

    // convert to cartesian coordinates        
    x = static_cast<int>(std::round(r * std::sin(phi) * std::cos(theta) / params.discreteGridSteps));
    y = static_cast<int>(std::round(r * std::sin(phi) * std::sin(theta) / params.discreteGridSteps));
    z = static_cast<int>(std::round(r * std::cos(phi) / params.discreteGridSteps));

    if(!allowZero && x == 0 && y == 0 && z == 0)
        getPointInSphere(rng, radius, center, allowZero);

    return DiscretePoint{center.x + x, center.y + y, center.z + z};
}

size_t FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t freeIndex) {
    float oldAtomEnergy = cluster.getAtomEnergy(freeIndex);
    int stepsSinceChange, numChanges, axis;
    stepsSinceChange = numChanges = axis = 0;

    DiscretePoint& freePoint = cluster.getPoint(freeIndex);

    while (stepsSinceChange < 3) {
        axis = (++axis) % 3;
        freePoint[axis] += 1;
        float newAtomEnergy = cluster.getAtomEnergy(freeIndex);

        if (newAtomEnergy - oldAtomEnergy < 0.0f) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        freePoint[axis] -= 2;
        newAtomEnergy = cluster.getAtomEnergy(freeIndex);

        if (newAtomEnergy - oldAtomEnergy < 0.0f) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        freePoint[axis] += 1;
        stepsSinceChange++;
    }

    return numChanges;
}