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
    : params(params), rng(std::random_device{}()) {
    
    // generate lookup
    const float squaredGridSpacing = params.gridSpacing * params.gridSpacing;
    const size_t maxSquaredDistance = params.cutoffDistance * params.cutoffDistance + 16; // + 16 is just a safety measure, such that I'm sure AVX ops don't load unwanted data
    lookup.reserve(maxSquaredDistance);

    lookup.emplace_back(0.0f);
    for (size_t i = 1; i < maxSquaredDistance; i++) {
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
    RunState state;
    auto& startCandidate = state.discreteCandidates.emplace_back(DiscreteCluster(), std::numeric_limits<float>::infinity());

    // step 1 (from paper)
    state.DMCWalker = state.discreteCandidates.front().first;
    initializeCluster(startCandidate.first, rng);
    localDiscreteOptimization(startCandidate.first);
    startCandidate.second = startCandidate.first.getClusterEnergy(params.gridSpacingSquared);
    std::cout << "SLO found: " << startCandidate.second << "\n";

    // step 2
    state.DMCWalker = startCandidate.first;
    runDMCLayer(state, params.dmcLayer1, rng);
    //runDMCLayer(state, params.dmcLayer2, rng);
    
    //TODO

    return SingleRunResult{};
}

MultiRunResult FuzzyGlobalOptimizer::runMultiple(size_t numRuns) {
    MultiRunResult multiResult;
    multiResult.allRuns.reserve(numRuns);

    for (size_t i = 0; i < numRuns; i++)
    {
        std::mt19937 runRng(std::random_device{}());
        auto singleResult = runSingle(runRng);
        multiResult.allRuns.emplace_back(std::move(singleResult));
    }

    return multiResult;
}

// private functions
// ===============================================
void FuzzyGlobalOptimizer::initializeCluster(DiscreteCluster& cluster, std::mt19937& rng) {
    // create a cluster of points, where the points lay on random points within some radius of 0 on a discrete grid. No two distinct point sit on the same discrete position
    cluster = DiscreteCluster(params.numberOfAtoms, params.cutoffDistance);

    float spawningRadius = params.spawningRadiusFactor * std::pow(params.numberOfAtoms, 0.33f);

    setPointInBall(cluster, 2, 0, rng, spawningRadius, 0, 0, 0);
    for (size_t i = 1; i < params.numberOfAtoms; i++) {
        do {
            setPointInBall(cluster, 2, i, rng, spawningRadius, 0, 0, 0);
        } while (cluster.doesPointOverlap(i, i-1));
    }
}

void FuzzyGlobalOptimizer::runDMCLayer(RunState& state, const FGOParameters::DMCParameters& dmcParams, std::mt19937& rng) {
    size_t stepsSinceImprovement = 0;
    
    DiscreteCluster proposal{params.numberOfAtoms, params.cutoffDistance};
    DiscreteCluster& walker = state.DMCWalker;

    std::vector<float> atomEnergies(params.numberOfAtoms);
    std::vector<float> activeWeights(params.numberOfAtoms);
    std::vector<float> targetWeights(params.numberOfAtoms);


    while (stepsSinceImprovement < (size_t)(params.numberOfAtoms * params.numberOfAtoms * dmcParams.convergenceFactor)) {
        // calculate distribution weights
        float sumActive = 0.f, sumTarget = 0.f;
        for (size_t i = 0; i < params.numberOfAtoms; i++)
        {
            const float E = walker.getAtomEnergyAVX(i, lookup); // how are we gonna do neighbours here? 
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
        setPointInBall(proposal, 1, activeAtom, rng, 1.0f, proposal.points[4*targetAtom], proposal.points[4*targetAtom+1], proposal.points[4*targetAtom+2], false); // TODO CRUCIAL somehow this should not put the point 1 next to any of the existing points

        localDiscreteFrozenOptimization(proposal, activeAtom);

        float deltaAtomEnergy = proposal.getAtomEnergyAVX(activeAtom, lookup) - atomEnergies[activeAtom];

        stepsSinceImprovement++;
        if (deltaAtomEnergy < 0.0f || uniformDist(rng) < fast_exp(-deltaAtomEnergy * dmcParams.invAcceptanceEnergy)) {
            localDiscreteOptimization(proposal);

            //TODO the cost of this could be removed, by modifying localDiscreteOptimization to keep track of the total sum of changes from improvements in getAtomEnergyAVX
            // hence after localDiscreteOptimization, we would know how much the discreteOptimization steps (an swap) changed the walker cluster energy, saving us a clusterEnergy call at the cost of some float operations
            float candidateEnergy = proposal.getClusterEnergy(params.gridSpacingSquared); 
            
            if(candidateEnergy < state.discreteCandidates.back().second) { // the back is guaranteed to be the best
                state.bestDistcrete = state.discreteCandidates.size();
                state.discreteCandidates.emplace_back(proposal, candidateEnergy);
                std::cout << "DMC found: " << candidateEnergy << "\n";
                stepsSinceImprovement = 0;
            }
            walker = proposal; // copy data from proposal into walker, regardless of if proposal is a new best
        }
    }
}

void FuzzyGlobalOptimizer::localDiscreteOptimization(DiscreteCluster& cluster) {
    std::list<size_t> activeList;
    alignas(64) std::vector<std::vector<uint64_t>> neighbourLists;

    const int32_t squaredCutoffDistance = params.cutoffDistance * params.cutoffDistance;

    for (size_t i = 0; i < cluster.n; i++) {
        activeList.emplace_back(i);
        neighbourLists.emplace_back(cluster.getNeighbours(i,squaredCutoffDistance));
    }

    while (!activeList.empty())
        activeList.remove_if([&cluster, &neighbourLists, this](int i){ return localDiscreteFrozenOptimization(cluster, i, neighbourLists[i]) == 0; });
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
        if (std::abs(candidate.second - E0) < 1e-12f) break;
    }
}

// helper functions
void FuzzyGlobalOptimizer::setPointInBall(DiscreteCluster& cluster, int spacingMultiple, size_t index, std::mt19937& rng, float radius, int16_t cx, int16_t cy, int16_t cz, bool allowZero) {
    // generate random spherical coordinates
    const float u = uniformDist(rng);
    const float r = radius * std::cbrt(u);
    const float theta = thetaDist(rng);
    const float phi = phiDist(rng);

    // convert to cartesian coordinates    
    const int dx = static_cast<int>(std::round(r * std::sin(phi) * std::cos(theta) / (spacingMultiple * params.gridSpacing))) * spacingMultiple;
    const int dy = static_cast<int>(std::round(r * std::sin(phi) * std::sin(theta) / (spacingMultiple * params.gridSpacing))) * spacingMultiple;
    const int dz = static_cast<int>(std::round(r * std::cos(phi) / (spacingMultiple * params.gridSpacing))) * spacingMultiple;

    if(!allowZero && dx == 0 && dy == 0 && dz == 0) {
        setPointInBall(cluster, spacingMultiple, index, rng, radius, cx, cy, cz, false);
        return;
    }

    cluster.points[4*index+0] = cx + dx;
    cluster.points[4*index+1] = cy + dy;
    cluster.points[4*index+2] = cz + dz;
}

void FuzzyGlobalOptimizer::setPointOnSphere(DiscreteCluster& cluster, int spacingMultiple, size_t index, std::mt19937& rng, float radius, int16_t cx, int16_t cy, int16_t cz) {
    const float theta = thetaDist(rng);
    const float phi = phiDist(rng);

    const float x = radius * std::sin(phi) * std::cos(theta);
    const float y = radius * std::sin(phi) * std::sin(theta);
    const float z = radius * std::cos(phi);
    
    const int16_t dx = static_cast<int16_t>(std::round(x / (spacingMultiple * params.gridSpacing))) * spacingMultiple;
    const int16_t dy = static_cast<int16_t>(std::round(y / (spacingMultiple * params.gridSpacing))) * spacingMultiple;
    const int16_t dz = static_cast<int16_t>(std::round(z / (spacingMultiple * params.gridSpacing))) * spacingMultiple;
    
    cluster.points[4*index+0] = cx + dx;
    cluster.points[4*index+1] = cy + dy;
    cluster.points[4*index+2] = cz + dz;
}

size_t FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t freeIndex, const std::vector<uint64_t>& neighbours) {
    float oldAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, neighbours, lookup);
    int stepsSinceChange, numChanges, axis;
    stepsSinceChange = numChanges = axis = 0;

    while (stepsSinceChange < 3) {
        axis = (++axis) % 3;

        cluster.points[4*freeIndex+axis] += 1;

        float newAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, neighbours, lookup);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        cluster.points[4*freeIndex+axis] -= 2;

        newAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, neighbours, lookup);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        cluster.points[4*freeIndex+axis] += 1;
        stepsSinceChange++;
    }

    return numChanges;
}

size_t FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(DiscreteCluster& cluster, const size_t freeIndex) {
    float oldAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, lookup);
    int stepsSinceChange, numChanges, axis;
    stepsSinceChange = numChanges = axis = 0;

    while (stepsSinceChange < 3) {
        axis = (++axis) % 3;

        cluster.points[4*freeIndex+axis] += 1;

        float newAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, lookup);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        cluster.points[4*freeIndex+axis] -= 2;

        newAtomEnergy = cluster.getAtomEnergyAVX(freeIndex, lookup);

        if (newAtomEnergy < oldAtomEnergy) {
            oldAtomEnergy = newAtomEnergy;
            stepsSinceChange = 0;
            numChanges++;
            continue;
        }

        cluster.points[4*freeIndex+axis] += 1;
        stepsSinceChange++;
    }

    return numChanges;
}