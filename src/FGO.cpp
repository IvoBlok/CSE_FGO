#include "FGO.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>
#include <list>

FuzzyGlobalOptimizer::FuzzyGlobalOptimizer(const FGOParameters& params)
    : params(params), rng(std::random_device{}()) {
    
    // generate lookup
    const float squaredGridSpacing = params.gridSpacing * params.gridSpacing;
    const size_t maxSquaredDistance = params.cutoffDistance * params.cutoffDistance + 16; // + 16 is just a safety measure, such that I'm sure AVX ops don't load unwanted data
    lookup.reserve(maxSquaredDistance);
    
    lookup.emplace_back(std::numeric_limits<float>::infinity()); // element one is set to positive infinity, though technically for LJ it is undefined at r^2 = 0
    for (size_t i = 1; i < maxSquaredDistance - 2; i++) {
        const float invr2 = 1.0f / (i * squaredGridSpacing);
        const float invr6 = invr2 * invr2 * invr2;

        lookup.emplace_back(invr6 * invr6 - 2.0f * invr6);
    }
}

SingleRunResult FuzzyGlobalOptimizer::runSingle() {
    return runSingle(rng);
}

SingleRunResult FuzzyGlobalOptimizer::runSingleWithSeed(uint32_t seed) {
    std::mt19937 seededRng(seed);
    return runSingle(seededRng);
}

SingleRunResult FuzzyGlobalOptimizer::runSingle(std::mt19937& rng) {
    auto startTotal = std::chrono::high_resolution_clock::now();

    RunState state;
    SingleRunResult result;
    auto startCluster = DiscreteCluster();

    // step 1 (from paper)
    initializeCluster(startCluster, rng);
    localDiscreteOptimization(startCluster);
    state.discCandidates.emplace_back(startCluster.exportPoints(), startCluster.getClusterEnergy(lookup));

    // step 2
    result.dmc1Time = runDMCLayer(state, startCluster, params.dmcLayer1, rng);
    float spawningRadius = params.spawningRadiusFactor * std::pow(params.numberOfAtoms, 0.33f);
    DiscreteCluster startDMC2(state.discCandidates.back().first, (int)(spawningRadius / params.gridSpacing), params.cutoffDistance * params.cutoffDistance, params.cellDistance);
    result.dmc2Time = runDMCLayer(state, startDMC2, params.dmcLayer2, rng); // initialize DMC2 with the best candidate from DMC1

    // step 3
    result.realOptTime = runRealOptimization(state, 2.0f);

    // step 4
    // TODO SMC

    // step 5
    // TODO REAL OPTIMIZATION

    result.discCandidates = std::move(state.discCandidates);
    result.contCandidates = std::move(state.contCandidates);
    result.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTotal);
    return result;
}

std::vector<SingleRunResult> FuzzyGlobalOptimizer::runMultiple(size_t numRuns) {
    std::vector<SingleRunResult> multiResult;
    multiResult.reserve(numRuns);

    for (size_t i = 0; i < numRuns; i++)
    {
        std::mt19937 runRng(std::random_device{}());
        auto singleResult = runSingle(runRng);
        multiResult.emplace_back(std::move(singleResult));
    }

    return multiResult;
}

// private functions
// ===============================================
std::chrono::microseconds FuzzyGlobalOptimizer::initializeCluster(DiscreteCluster& cluster, std::mt19937& rng) {
    const auto startTime = std::chrono::high_resolution_clock::now();

    float spawningRadius = params.spawningRadiusFactor * std::pow(params.numberOfAtoms, 0.33f);

    DiscretePoints randomPoints;
    randomPoints.x.reserve(params.numberOfAtoms);
    randomPoints.y.reserve(params.numberOfAtoms);
    randomPoints.z.reserve(params.numberOfAtoms);

    for (size_t i = 0; i < params.numberOfAtoms; i++) {
        auto point = getPointInBall(rng, spawningRadius, {0, 0, 0});
        randomPoints.x.push_back(point[0]);
        randomPoints.y.push_back(point[1]);
        randomPoints.z.push_back(point[2]);
    }

    cluster = DiscreteCluster{randomPoints, (int)(spawningRadius / params.gridSpacing), params.cutoffDistance * params.cutoffDistance, params.cellDistance};

    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime);
}

std::chrono::microseconds FuzzyGlobalOptimizer::runDMCLayer(RunState& state, const DiscreteCluster& startCluster, const FGOParameters::DMCParameters& dmcParams, std::mt19937& rng) {
    const auto startTime = std::chrono::high_resolution_clock::now();

    DiscreteCluster walker{startCluster};

    std::vector<float> atomEnergies(params.numberOfAtoms);
    std::vector<float> activeWeights(params.numberOfAtoms);
    std::vector<float> targetWeights(params.numberOfAtoms);
    float sumActive, sumTarget;

    bool recomputeWeights = true;
    size_t stepsSinceImprovement = 0;

    while (stepsSinceImprovement < (size_t)(params.numberOfAtoms * params.numberOfAtoms * dmcParams.convergenceFactor)) {
        if (recomputeWeights) {
            // calculate distribution weights
            sumActive = 0, sumTarget = 0;
            for (size_t i = 0; i < params.numberOfAtoms; i++)
            {
                const float E = walker.getAtomEnergySlow(i, lookup);
                atomEnergies[i] = E;
                
                const float wa = std::exp(E * dmcParams.invActiveEnergy);
                const float d = E - dmcParams.targetEnergy;
                const float wt = std::exp(d * d * dmcParams.inv2Sigma2);

                activeWeights[i] = wa;
                targetWeights[i] = wt;

                sumActive += wa;
                sumTarget += wt;
            }
            recomputeWeights = false;
        }

        size_t activeAtom, targetAtom;

        // sample the distributions, once for the target atom, once for the active atom
        float uniform = uniformDist(rng) * sumActive;
        float accumulate = 0.f;
        for (activeAtom = 0; activeAtom < params.numberOfAtoms; activeAtom++) {
            accumulate += activeWeights[activeAtom];
            if (uniform <= accumulate) break;
        }

        uniform = uniformDist(rng) * sumTarget;
        accumulate = 0.f;
        for (targetAtom = 0; targetAtom < params.numberOfAtoms; targetAtom++) {
            accumulate += targetWeights[targetAtom];
            if (uniform <= accumulate) break;
        }

        DiscreteCoord oldPoint = walker.getAtom(activeAtom);
        DiscreteCoord newPoint = getPointInBall(rng, 1.0f, walker.getAtom(targetAtom));
        float deltaAtomEnergy = localDiscreteFrozenOptimization(walker, activeAtom, newPoint) - atomEnergies[activeAtom]; // optimize the activeAtom, starting from newPoint. the optimal spot then gets stored in walker

        if (deltaAtomEnergy < 0.0f || uniformDist(rng) < std::exp(-deltaAtomEnergy * dmcParams.invAcceptanceEnergy)) {
            // accept move, update walker and weights
            recomputeWeights = true;
            localDiscreteOptimization(walker);
            stepsSinceImprovement += 1;

            float candidateEnergy = walker.getClusterEnergy(lookup); 
            if(candidateEnergy < state.discCandidates.back().second) {
                // accept candidate
                state.discCandidates.emplace_back(walker.exportPoints(), candidateEnergy);
                stepsSinceImprovement = 0;
            }
        } else {
            // reverse the change to the unchanged walker
            walker.updateAtom(activeAtom, oldPoint); //TODO this can also largely be saved; modify 'localDiscreteFrozenOptimization' to not update the point at the end, but to return both the energy and final point; then only update if the E < 0 / metropolis hits, which is drastically less often then the reverse
        }
    }

    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime);
}

std::chrono::microseconds FuzzyGlobalOptimizer::runRealOptimization(RunState& state, const float acceptanceThreshold) {
    const auto startTime = std::chrono::high_resolution_clock::now();

    for (const auto& [cluster, energy] : state.discCandidates)
    {
        if (energy < state.discCandidates.back().second + acceptanceThreshold) {
            auto& contCandidate = state.contCandidates.emplace_back(Cluster(cluster, params.gridSpacing), std::numeric_limits<float>::infinity());
            contCandidate.second = contCandidate.first.getClusterEnergyAVX();
            localRealOptimization(contCandidate);
        }
    }

    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime);
}

// helper functions
DiscreteCoord FuzzyGlobalOptimizer::getPointInBall(std::mt19937& rng, float radius, DiscreteCoord center, bool allowZero) {
    // generate random spherical coordinates
    const float u = uniformDist(rng);
    const float r = radius * std::cbrt(u);
    const float theta = thetaDist(rng);
    const float phi = phiDist(rng);

    // convert to cartesian coordinates    
    const int32_t dx = static_cast<int32_t>(std::round(r * std::sin(phi) * std::cos(theta) / params.gridSpacing));
    const int32_t dy = static_cast<int32_t>(std::round(r * std::sin(phi) * std::sin(theta) / params.gridSpacing));
    const int32_t dz = static_cast<int32_t>(std::round(r * std::cos(phi) / params.gridSpacing));

    if(!allowZero && dx == 0 && dy == 0 && dz == 0) {
        return getPointInBall(rng, radius, center, false);
    }
    
    return {center[0] + dx, center[1] + dy, center[2] + dz};
}

DiscreteCoord FuzzyGlobalOptimizer::getPointOnSphere(std::mt19937& rng, float radius, DiscreteCoord center) {
    const float theta = thetaDist(rng);
    const float phi = phiDist(rng);

    const float x = radius * std::sin(phi) * std::cos(theta);
    const float y = radius * std::sin(phi) * std::sin(theta);
    const float z = radius * std::cos(phi);
    
    const int32_t dx = static_cast<int32_t>(std::round(x / params.gridSpacing));
    const int32_t dy = static_cast<int32_t>(std::round(y / params.gridSpacing));
    const int32_t dz = static_cast<int32_t>(std::round(z / params.gridSpacing));
    
    return {center[0] + dx, center[0] + dy, center[0] + dz};
}

void FuzzyGlobalOptimizer::localDiscreteOptimization(DiscreteCluster& cluster) {
    std::list<size_t> activeList;
    for (size_t i = 0; i < cluster.n; i++)
        activeList.push_back(i);

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
        cluster.getClusterGradient(gradX, gradY, gradZ);

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
        const float E1 = gradCluster1.getClusterEnergyAVX();
        const float E2 = gradCluster2.getClusterEnergyAVX();
        
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
        candidate.second = cluster.getClusterEnergyAVX();
        if (std::abs(candidate.second - E0) < 1e-12f) break;
    }
}

size_t FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t freeIndex) {
    DiscreteCoord freePoint = cluster.getAtom(freeIndex);
    auto buffer = cluster.gatherNeighbourBuffer(freeIndex, freePoint);
    float oldAtomEnergy = cluster.getAtomEnergy(freePoint, buffer, lookup);

    int stepsSinceChange, numChanges, axis;
    stepsSinceChange = numChanges = axis = 0;

    while (stepsSinceChange < 3) {
        axis = (++axis) % 3;

        if (numChanges % params.neighbourUpdateInterval == params.neighbourUpdateInterval - 1)
            buffer = cluster.gatherNeighbourBuffer(freeIndex, freePoint);
            
        freePoint[axis] += 1;
        float newAtomEnergy = cluster.getAtomEnergy(freePoint, buffer, lookup);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        freePoint[axis] -= 2;
        newAtomEnergy = cluster.getAtomEnergy(freePoint, buffer, lookup);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        freePoint[axis] += 1;
        stepsSinceChange++;
    }

    // confirm the changes to the cluster
    cluster.updateAtom(freeIndex, freePoint);
    return numChanges;
}

float FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t freeIndex, DiscreteCoord freePoint) {
    auto buffer = cluster.gatherNeighbourBuffer(freeIndex, freePoint);
    float oldAtomEnergy = cluster.getAtomEnergy(freePoint, buffer, lookup);

    int stepsSinceChange, numChanges, axis;
    stepsSinceChange = numChanges = axis = 0;

    while (stepsSinceChange < 3) {
        axis = (++axis) % 3;

        if (numChanges % params.neighbourUpdateInterval == params.neighbourUpdateInterval - 1)
            buffer = cluster.gatherNeighbourBuffer(freeIndex, freePoint);

        freePoint[axis] += 1;
        float newAtomEnergy = cluster.getAtomEnergy(freePoint, buffer, lookup);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        freePoint[axis] -= 2;
        newAtomEnergy = cluster.getAtomEnergy(freePoint, buffer, lookup);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        freePoint[axis] += 1;
        stepsSinceChange++;
    }

    // confirm the changes to the cluster
    cluster.updateAtom(freeIndex, freePoint);
    return oldAtomEnergy;
}

