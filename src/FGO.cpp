#include "FGO.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>
#include <list>

# define M_PI           3.14159265358979323846  /* pi */

inline float fast_exp(float x)
{
    x = std::max(-50.0f, std::min(50.0f, x));
    x *= 1.4426950408889634f;

    int i = static_cast<int>(x);
    float f = x - i;

    float p = 1.0f + f * (0.69314718f + f * (0.24022651f + f * 0.05550411f));
    return std::ldexp(p, i);
}


FuzzyGlobalOptimizer::FuzzyGlobalOptimizer(const FGOParameters& params)
    : params(params), rng(std::random_device{}()) {}

FuzzyGlobalOptimizer::FuzzyGlobalOptimizer(FGOParameters&& params)
    : params(std::move(params)), rng(std::random_device{}()) {}

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
    startCandidate.second = startCandidate.first.getClusterEnergyAVX(fastLJ);

    auto startDMC1 = std::chrono::high_resolution_clock::now();
    state.DMCWalker = startCandidate.first;
    runDMCLayer(state, params.dmcLayer1, rng);
    auto endDMC1 = std::chrono::high_resolution_clock::now();
    runDMCLayer(state, params.dmcLayer2, rng);
    auto endDMC2 = std::chrono::high_resolution_clock::now();

    auto startRealOpt = std::chrono::high_resolution_clock::now();
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
    auto endRealOpt = std::chrono::high_resolution_clock::now();

    // TODO SMC

    // TODO local real optimization of best continuous clusters

    auto endTotal = std::chrono::high_resolution_clock::now();

    SingleRunResult result;
    result.bestCluster = state.candidates[state.bestIndex].first;
    result.bestEnergy = state.candidates[state.bestIndex].second;
    result.candidates = std::move(state.candidates);
    
    result.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTotal - startTotal);
    result.dmc1Time = std::chrono::duration_cast<std::chrono::microseconds>(endDMC1 - startDMC1);
    result.dmc2Time = std::chrono::duration_cast<std::chrono::microseconds>(endDMC2 - endDMC1);
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
    
    Cluster proposal{params.numberOfAtoms};
    Cluster& walker = state.DMCWalker;

    std::vector<float> atomEnergies(params.numberOfAtoms);
    std::vector<float> activeWeights(params.numberOfAtoms);
    std::vector<float> targetWeights(params.numberOfAtoms);


    while (stepsSinceImprovement < (size_t)(params.numberOfAtoms * params.numberOfAtoms * dmcParams.convergenceFactor)) {
        // calculate distribution weights
        float sumActive = 0.f, sumTarget = 0.f;
        for (size_t i = 0; i < params.numberOfAtoms; i++)
        {
            const float E = walker.getAtomEnergyAVX(i, fastLJ);
            atomEnergies[i] = E;
            
            const float wa = fast_exp(E * dmcParams.invActiveEnergy);
            const float d = E - dmcParams.targetEnergy;
            const float wt = fast_exp(d * d * dmcParams.inv2Sigma2);

            activeWeights[i] = wa;
            targetWeights[i] = wt;

            sumActive += wa;
            sumTarget += wt;
        }

        // sample distributions, once for both
        float uniform = uniformDist(rng) * sumActive;
        float accumulate = 0.f;
        size_t activeAtom = 0;
        for (; activeAtom < params.numberOfAtoms; activeAtom++) {
            accumulate += activeWeights[activeAtom];
            if (uniform <= accumulate) break;
        }

        uniform = uniformDist(rng) * sumTarget;
        accumulate = 0.f;
        size_t targetAtom = 0;
        for (; targetAtom < params.numberOfAtoms; targetAtom++) {
            accumulate += targetWeights[targetAtom];
            if (uniform <= accumulate) break;
        }

        walker.copyTo(proposal);
        setPointInSphere(proposal, activeAtom, rng, 1.0f, proposal.x[targetAtom], proposal.y[targetAtom], proposal.z[targetAtom], false);

        localDiscreteFrozenOptimization(proposal, activeAtom);

        float deltaAtomEnergy = proposal.getAtomEnergyAVX(activeAtom, fastLJ) - atomEnergies[activeAtom];

        stepsSinceImprovement++;
        if (deltaAtomEnergy < 0.0f || uniformDist(rng) < fast_exp(-deltaAtomEnergy * dmcParams.invAcceptanceEnergy)) {
            localDiscreteOptimization(proposal);

            float candidateEnergy = proposal.getClusterEnergyAVX(fastLJ);
            if(candidateEnergy < state.candidates.back().second) { // the back is guaranteed to be the best
                state.bestIndex = state.candidates.size();
                state.candidates.emplace_back(proposal, candidateEnergy);
                stepsSinceImprovement = 0;
            }
            walker = proposal; // copy data from proposal into walker, regardless of if proposal is a new best
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

void FuzzyGlobalOptimizer::localRealOptimization(std::pair<Cluster, float>& candidate) {
    Cluster& cluster = candidate.first;
    const size_t n = params.numberOfAtoms;
    const float step = params.gradientStepSize;

    std::vector<float> gradX(n);
    std::vector<float> gradY(n);
    std::vector<float> gradZ(n);

    static thread_local Cluster gradCluster1, gradCluster2;
    if (gradCluster1.n != n) gradCluster1 = Cluster(n);
    if (gradCluster2.n != n) gradCluster2 = Cluster(n);

    for (size_t iter = 0; iter < params.maxRealOptimizationIterations; iter++)
    {
        cluster.getClusterGradient(gradX, gradY, gradZ, fastLJ);

        // line search
        for (size_t i = 0; i < n; ++i) {
            gradCluster1.x[i] = cluster.x[i] - gradX[i] * step;
            gradCluster1.y[i] = cluster.y[i] - gradY[i] * step;
            gradCluster1.z[i] = cluster.z[i] - gradZ[i] * step;
            
            gradCluster2.x[i] = cluster.x[i] - gradX[i] * (2.0f * step);
            gradCluster2.y[i] = cluster.y[i] - gradY[i] * (2.0f * step);
            gradCluster2.z[i] = cluster.z[i] - gradZ[i] * (2.0f * step);
        }

        const float E0 = candidate.second;
        const float E1 = gradCluster1.getClusterEnergyAVX(fastLJ);
        const float E2 = gradCluster2.getClusterEnergyAVX(fastLJ);
        
        // interpolate quadratic equation
        const float denom = 2.0f * E2 - 4.0f * E1 + 2.0f * E0;
        if (std::abs(denom) < 1e-7f) break;

        const float alpha = (-3.0f * E0 + 4.0f * E1 - E2) / denom;
        
        // move to minimum of fitted quadratic
        for (size_t i = 0; i < n; ++i) {
            cluster.x[i] += gradX[i] * (step * alpha);
            cluster.y[i] += gradY[i] * (step * alpha);
            cluster.z[i] += gradZ[i] * (step * alpha);
        }

        // convergence condition
        candidate.second = cluster.getClusterEnergyAVX(fastLJ);
        if (std::abs(candidate.second - E0) < 1e-10f) break;
    }
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
    float oldAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, fastLJ);
    int stepsSinceChange, numChanges, axis;
    stepsSinceChange = numChanges = axis = 0;

    while (stepsSinceChange < 3) {
        axis = (++axis) % 3;

        float& coord = (axis == 0) ? cluster.x[freeIndex] : (axis == 1) ? cluster.y[freeIndex] : cluster.z[freeIndex];
        coord += params.gridSpacing;

        float newAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, fastLJ);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        coord -= 2 * params.gridSpacing;
        newAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, fastLJ);

        if (newAtomEnergy < oldAtomEnergy) {
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