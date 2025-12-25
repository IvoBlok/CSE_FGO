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

    auto startTotal = std::chrono::high_resolution_clock::now();

    auto& startCandidate = state.candidates.emplace_back(Cluster(), std::numeric_limits<float>::infinity());

    initializeCluster(startCandidate.first, rng);
    localDiscreteOptimization(startCandidate.first);
    startCandidate.second = startCandidate.first.getClusterEnergy();

    auto startDMC = std::chrono::high_resolution_clock::now();
    runDMCLayer(state, params.dmcLayer1, rng);
    auto endDMC = std::chrono::high_resolution_clock::now();

    // TODO DMC2

    auto startRealOpt = std::chrono::high_resolution_clock::now();
    /*
    state.bestIndex = 0; // reset bestIndex, to fix in issue in the rare scenario that the real optimization leads to a worse energy
    for (size_t i = 0; i < state.candidates.size(); i++)
    {
        auto& candidate = state.candidates[i];

        if (candidate.second < state.candidates[state.bestIndex].second + 2.0f) {
            localRealOptimization(candidate);

            if (candidate.second < state.candidates[state.bestIndex].second)
                state.bestIndex = i;
        }
    }
    */
    auto endRealOpt = std::chrono::high_resolution_clock::now();

    // TODO SMC

    // TODO local real optimization of best continuous clusters

    auto endTotal = std::chrono::high_resolution_clock::now();

    SingleRunResult result;
    result.bestCluster = state.candidates[state.bestIndex].first;
    result.bestEnergy = state.candidates[state.bestIndex].second;
    result.candidates = std::move(state.candidates);
    
    result.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTotal - startTotal);
    result.dmcTime = std::chrono::duration_cast<std::chrono::microseconds>(endDMC - startDMC);
    result.realOptTime = std::chrono::duration_cast<std::chrono::microseconds>(endRealOpt - startRealOpt);

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

        multiResult.totalTime += singleResult.totalTime;

        if (singleResult.bestEnergy < multiResult.globalBestEnergy) {
            multiResult.globalBestEnergy = singleResult.bestEnergy;
            multiResult.globalBestCluster = singleResult.bestCluster;
        }
    }

    if (numRuns > 0)
        multiResult.averageTime = multiResult.totalTime / numRuns;
    
    return multiResult;
}


// private functions
// ===============================================
void FuzzyGlobalOptimizer::initializeCluster(Cluster& cluster, std::mt19937& rng) {
    cluster = Cluster(params.numberOfAtoms);

    float spawningRadius = params.spawningRadiusFactor * std::pow(params.numberOfAtoms, 0.33f);

    for (size_t i = 0; i < params.numberOfAtoms; i++)
        setPointInSphere(cluster, i, rng, spawningRadius, 0.f, 0.f, 0.f);
}

void FuzzyGlobalOptimizer::runDMCLayer(RunState& state, const FGOParameters::DMCParameters& dmcParams, std::mt19937& rng) {
    size_t stepsSinceImprovement = 0;

    Cluster candidate{params.numberOfAtoms};

    std::vector<float> atomEnergies(params.numberOfAtoms);
    std::vector<float> activeWeights(params.numberOfAtoms);
    std::vector<float> targetWeights(params.numberOfAtoms);

    while (stepsSinceImprovement < (size_t)(params.numberOfAtoms * params.numberOfAtoms * dmcParams.convergenceFactor)) {
        const auto& currCandidate = state.candidates.back();

        for (size_t i = 0; i < params.numberOfAtoms; i++)
        {
            atomEnergies[i] = currCandidate.first.getAtomEnergy(i);

            activeWeights[i] = std::exp(atomEnergies[i] / dmcParams.activeEnergy);
            targetWeights[i] = std::exp(-0.5 * std::pow(atomEnergies[i] - dmcParams.targetEnergy, 2) / std::pow(dmcParams.targetSigma, 2));
        }
        
        atomSelector.updateDistribution(activeWeights);
        size_t activeAtom = atomSelector.generate(rng);
        atomSelector.updateDistribution(targetWeights);
        size_t targetAtom = atomSelector.generate(rng);

        currCandidate.first.copyTo(candidate);
        setPointInSphere(candidate, activeAtom, rng, 1.0f, candidate.x[targetAtom], candidate.y[targetAtom], candidate.z[targetAtom], false);

        localDiscreteFrozenOptimization(candidate, activeAtom);

        float deltaAtomEnergy = candidate.getAtomEnergy(activeAtom) - atomEnergies[activeAtom];

        stepsSinceImprovement++;
        float acceptanceThreshold = uniformDist(rng);
        if (deltaAtomEnergy < 0.0f || acceptanceThreshold < std::exp(-deltaAtomEnergy / dmcParams.acceptanceEnergy)) {
            localDiscreteOptimization(candidate);

            float candidateEnergy = candidate.getClusterEnergy();
            if(candidateEnergy < state.candidates[state.bestIndex].second) {
                state.bestIndex = state.candidates.size();
                state.candidates.emplace_back(candidate, candidateEnergy);
                stepsSinceImprovement = 0;
            }
        }
    }
}

void FuzzyGlobalOptimizer::localDiscreteOptimization(Cluster& cluster) {
    std::list<size_t> activeList;
    for (size_t i = 0; i < cluster.size(); i++)
        activeList.emplace_back(i);

    while (!activeList.empty())
        activeList.remove_if([&cluster, this](int i){ return localDiscreteFrozenOptimization(cluster, i) == 0; });
}

// helper functions
void FuzzyGlobalOptimizer::setPointInSphere(Cluster& cluster, size_t index, std::mt19937& rng, float radius, float cx, float cy, float cz, bool allowZero) {
    // generate random spherical coordinates
    const float u = uniformDist(rng);
    const float r = radius * std::cbrt(u);
    const float theta = thetaDist(rng);
    const float phi = phiDist(rng);

    // convert to cartesian coordinates    
    const int dx = static_cast<int>(std::round(r * std::sin(phi) * std::cos(theta) / params.gridSpacing));
    const int dy = static_cast<int>(std::round(r * std::sin(phi) * std::sin(theta) / params.gridSpacing));
    const int dz = static_cast<int>(std::round(r * std::cos(phi) / params.gridSpacing));

    if(!allowZero && dx == 0 && dy == 0 && dz == 0) {
        setPointInSphere(cluster, index, rng, radius, cx, cy, cz, allowZero);
        return;
    }

    cluster.x[index] = cx + dx * params.gridSpacing;
    cluster.y[index] = cy + dy * params.gridSpacing;
    cluster.z[index] = cz + dz * params.gridSpacing;
}

size_t FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(Cluster& cluster, const size_t freeIndex) {
    float oldAtomEnergy = cluster.getAtomEnergy(freeIndex);
    int stepsSinceChange, numChanges, axis;
    stepsSinceChange = numChanges = axis = 0;

    float& x = cluster.x[freeIndex];
    float& y = cluster.y[freeIndex];
    float& z = cluster.z[freeIndex];

    while (stepsSinceChange < 3) {
        axis = (++axis) % 3;

        float& coord = (axis == 0) ? cluster.x[freeIndex] : (axis == 1) ? cluster.y[freeIndex] : cluster.z[freeIndex];
        coord += params.gridSpacing;

        float newAtomEnergy = cluster.getAtomEnergy(freeIndex);

        if (newAtomEnergy - oldAtomEnergy < 0.0f) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        coord -= 2 * params.gridSpacing;
        newAtomEnergy = cluster.getAtomEnergy(freeIndex);

        if (newAtomEnergy - oldAtomEnergy < 0.0f) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        coord += 1 * params.gridSpacing;
        stepsSinceChange++;
    }

    return numChanges;
}