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
    auto& candidate = state.discreteCandidates.emplace_back(DiscreteCluster(), std::numeric_limits<float>::infinity());

    initializeCluster(candidate.first, rng);
    localDiscreteOptimization(candidate.first);
    candidate.second = candidate.first.getClusterEnergy();

    runDMCLayer(state, params.dmcLayer1, rng);

    // TODO DMC2

    for (const auto& candidate : state.discreteCandidates)
    {
        if (candidate.second < state.discreteCandidates[state.bestDiscrete].second + 2.0f) {
            state.continuousCandidates.emplace_back(ContinuousCluster(candidate.first, params.discreteGridSteps), candidate.second);
            localRealOptimization(state.continuousCandidates.back());

            if (state.continuousCandidates.back().second < state.continuousCandidates[state.bestContinuous].second)
                state.bestContinuous = state.continuousCandidates.size() - 1;
        }
    }

    // TODO SMC

    // TODO local real optimization of best continuous clusters

    SingleRunResult result;
    result.bestCluster = state.continuousCandidates[state.bestContinuous].first;
    result.bestEnergy = state.continuousCandidates[state.bestContinuous].second;
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
        const auto& currCandidate = state.discreteCandidates.back();

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
        candidate.getPoint(activeAtom) = getPointInSphere(rng, 1.0f, candidate.getPoint(targetAtom), false);

        localDiscreteFrozenOptimization(candidate, activeAtom);

        float deltaAtomEnergy = candidate.getAtomEnergy(activeAtom) - atomEnergies[activeAtom];

        stepsSinceImprovement++;
        float acceptanceThreshold = uniformDist(rng);
        if (deltaAtomEnergy < 0.0f || acceptanceThreshold < std::exp(-deltaAtomEnergy / dmcParams.acceptanceEnergy)) {
            localDiscreteOptimization(candidate);

            float candidateEnergy = candidate.getClusterEnergy();
            if(candidateEnergy < state.discreteCandidates[state.bestDiscrete].second) {
                state.bestDiscrete = state.discreteCandidates.size();
                state.discreteCandidates.emplace_back(candidate, candidateEnergy);
                stepsSinceImprovement = 0;
            }
        }
    }
}

void FuzzyGlobalOptimizer::localRealOptimization(std::pair<ContinuousCluster, float>& candidate) {
    std::vector<ContinuousPoint> gradient(params.numberOfAtoms);

    float distance, distanceSquared;
    ContinuousPoint direction;

    ContinuousCluster gradientCluster1 = ContinuousCluster(params.numberOfAtoms);
    ContinuousCluster gradientCluster2 = ContinuousCluster(params.numberOfAtoms);

    float newEnergy1, newEnergy2;
    float lastOriginalEnergy = std::numeric_limits<float>::infinity();

    for (size_t iter = 0; iter < params.maxRealOptimizationIterations; iter++)
    {
        for (size_t i = 0; i < params.numberOfAtoms; i++)
        {
            gradient[i] = ContinuousPoint(0.0f);

            for (size_t j = 0; j < params.numberOfAtoms; j++)
            {
                if (j != i) {
                    distanceSquared = candidate.first.getDistanceSquared(i, j);
                    distance = std::sqrt(distanceSquared);

                    direction = candidate.first.getPoint(i) - candidate.first.getPoint(j);
                    direction = (direction / distance) * lennardJonesDerivative(distance);
                    gradient[i] = gradient[i] + direction;
                }
            }
            // limit the length of each gradient element to a max of 100 (TODO fix this, such that it uses the full gradient length instead of element-wise)
            gradient[i] = (gradient[i] / std::sqrt(gradient[i].lengthSquared())) * std::fmin(100.f, std::sqrt(gradient[i].lengthSquared()));
            
        }
        
        for (size_t i = 0; i < params.numberOfAtoms; i++)
        {
            gradientCluster1.getPoint(i) = candidate.first.getPoint(i) - gradient[i] * params.gradientStepSize;
            gradientCluster2.getPoint(i) = candidate.first.getPoint(i) - gradient[i] * (2.0f * params.gradientStepSize);
        }

        newEnergy1 = gradientCluster1.getClusterEnergy();
        newEnergy2 = gradientCluster2.getClusterEnergy();
        
        if (std::abs(2*newEnergy2 - 4*newEnergy1 + 2*candidate.second) < 1e-7f)
            break;

        float optimalDeflectionFactor = (4*newEnergy1 - newEnergy2 - 3*candidate.second) / (-4*newEnergy1 + 2*newEnergy2 + 2*candidate.second);
        candidate.first.addToPoints(gradient, params.gradientStepSize * optimalDeflectionFactor);

        float oldEnergy = candidate.second;
        candidate.second = candidate.first.getClusterEnergy();

        if (std::abs(candidate.second - oldEnergy) < 1e-6f)
            break;
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