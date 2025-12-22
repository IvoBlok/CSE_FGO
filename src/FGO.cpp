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

    initializeCluster(state.currentDiscrete, rng);
    localDiscreteOptimization(state.currentDiscrete);
    state.bestEnergy = state.currentDiscrete.getClusterEnergy();

    runDMCLayer(state, params.dmcLayer1, rng);

    // TODO DMC2, SMC

    SingleRunResult result;
    result.bestCluster = ContinuousCluster(state.discreteCandidates[state.bestDiscreteIndex], params.discreteGridSteps);
    result.bestEnergy = state.bestEnergy;
    
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

    state.discreteCandidates.emplace_back(state.currentDiscrete);
    state.bestDiscreteIndex = 0;
    
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


/*
void FuzzyGlobalOptimizer::runFGO() {
    // ===============================================
    // STEP 1: create initial cluster
    currentCluster = DiscreteCluster{discreteGridSteps*discreteGridSteps, numberOfAtoms};
    generateInitialCluster(currentCluster, spawningRadius);
    localDiscreteOptimization(currentCluster);
    bestClusterEnergy = currentCluster.getClusterEnergy(LJLookup);

    // ===============================================
    // STEP 2: run two distinct DMC layers, with experimentally determined hyperparameters as inputs
    discreteMonteCarlo(1.0f, -4.1f, 1.25f, 0.4f, 2.5f);

    if(bestClusterIndex != -1)
        candidateClusters[bestClusterIndex].copyTo(currentCluster);

    discreteMonteCarlo(1.0f, -11.0f, 1.3f, 0.3f, 1.5f);

    // ===============================================
    // STEP 3: locally optimize all candidate clusters in the real space

    // get all candidates with low energies
    std::vector<ContinuousCluster> goodCandidates;

    for (size_t i = 0; i < candidateClusters.size(); i++)
    {
        if (candidateClusters[i].getClusterEnergy(LJLookup) < bestClusterEnergy + 2.f) {
            goodCandidates.emplace_back(ContinuousCluster{candidateClusters[i], discreteGridSteps});

            // optimize the appropriate candidates now in real 3D space
            localRealOptimization(goodCandidates.back());
        }
    }

    // TEMP: For debugging / development purposes, we retrieve the best real-optimized candidate
    for (size_t i = 0; i < goodCandidates.size(); i++)
    {   
        float energy = goodCandidates[i].getClusterEnergy(lennardJonesSquaredPotential);
        if (energy < bestClusterEnergy)
            bestClusterEnergy = energy;
    }

    // ===============================================
    // STEP 4, 5: Surface Monte Carlo (SMC). mainly important for larger clusters (>200)
    // TODO SMC
}

void FuzzyGlobalOptimizer::discreteMonteCarlo(float activeEnergy, float targetEnergy, float targetSigma, float acceptanceEnergy, float convergenceFactor) {
    int lastSinceImprovement = 0;

    DiscreteCluster candidateCluster{discreteGridSteps*discreteGridSteps, numberOfAtoms};

    std::vector<float> atomEnergies(numberOfAtoms);
    std::vector<float> atomActiveWeights(numberOfAtoms);
    std::vector<float> atomTargetWeights(numberOfAtoms);

    while (lastSinceImprovement < (int)(numberOfAtoms*numberOfAtoms*convergenceFactor))
    {
        for (int i = 0; i < numberOfAtoms; i++) {
            atomEnergies[i] = currentCluster.getAtomEnergy(LJLookup, i);

            atomActiveWeights[i] = std::exp(atomEnergies[i]/activeEnergy);
            atomTargetWeights[i] = std::exp(-std::pow(atomEnergies[i] - targetEnergy, 2)/(2*std::pow(targetSigma, 2)));
        }

        int activeAtom = getRandomAtomByWeights(atomActiveWeights);
        int targetAtom = getRandomAtomByWeights(atomTargetWeights);

        // make a new candidate cluster, with the active atomed moved to the area around the target atom, in a sphere of radius 1.
        // Since the problem is tackled in reduced units, a distance of 1 ( or 2^(1/6)) is the optimum distance between two atoms (assuming no other atoms are in the cluster).
        currentCluster.copyTo(candidateCluster);
        setAtomInRandomSphere(candidateCluster, activeAtom, 1.0f, currentCluster.getPoint(targetAtom), false);

        // locally optimize the modified cluster in the discrete space, while holding the rest of the cluster still
        localDiscreteFrozenOptimization(candidateCluster, activeAtom);

        // if the local energy of the moved atom improved, we directly accept the new candidate
        float deltaLocalAtomEnergy = candidateCluster.getAtomEnergy(LJLookup, activeAtom) - atomEnergies[activeAtom];

        float randomExpAcceptanceThreshold = dist(gen);

        if(deltaLocalAtomEnergy < 0.f || randomExpAcceptanceThreshold < std::exp(-deltaLocalAtomEnergy/acceptanceEnergy)) {
            
            localDiscreteOptimization(candidateCluster);
            
            // if the discrete local optimization (DLO) found a new best cluster, keep the candidate
            float candidateEnergy = candidateCluster.getClusterEnergy(LJLookup);
            if(candidateEnergy < bestClusterEnergy) {
                bestClusterEnergy = candidateEnergy;
                bestClusterIndex = candidateClusters.size();
                candidateClusters.emplace_back(candidateCluster);
                lastSinceImprovement = 0;
            } else {
                lastSinceImprovement++;
            }

            // now that we have a probably better cluster, make it the base cluster for the next iteration
            candidateCluster.copyTo(currentCluster);
        } else {
            // the move - local optimization combo chosen here didn't work out. keep the original cluster
            lastSinceImprovement++;
        }
    }
}

void FuzzyGlobalOptimizer::localRealOptimization(ContinuousCluster& cluster) {
    // gradient is initialized with 0 vector
    std::vector<ContinuousPoint> gradient;
    gradient.reserve(cluster.numberOfPoints);
    for (int i = 0; i < cluster.numberOfPoints; i++)
        gradient.emplace_back(ContinuousPoint(0.f, 0.f, 0.f));

    float distanceSquared;
    float distance;
    ContinuousPoint direction;

    ContinuousCluster gradientCluster1 = ContinuousCluster(cluster.numberOfPoints);
    ContinuousCluster gradientCluster2 = ContinuousCluster(cluster.numberOfPoints);

    float originalEnergy, newEnergy1, newEnergy2;
    float lastOriginalEnergy = std::numeric_limits<float>::infinity();

    for (int iteration = 0; iteration < 1000; iteration++)
    {
        // fill the gradient, by iterating over each atom pair
        for (int i = 0; i < cluster.numberOfPoints; i++)
        {   
            gradient[i] = ContinuousPoint(0.f, 0.f, 0.f);

            for (int j = 0; j < cluster.numberOfPoints; j++)
            {
                if(j != i) {
                    distanceSquared = cluster.getDistanceSquared(i, j);
                    distance = std::sqrt(distanceSquared);

                    direction = cluster.getPoint(i) - cluster.getPoint(j);
                    direction = (direction * (1.0f / distance)) * lennardJonesDerivative(distance);
                    gradient[i] = gradient[i] + direction;
                }
            }
            // limit the gradient length to a max of 100
            gradient[i] = gradient[i] * (1.0f / std::sqrt(gradient[i].lengthSquared())) * std::fmin(100.f, std::sqrt(gradient[i].lengthSquared()));
        }
        
        // create two new clusters, by moving in the opposite direction of the gradient
        for (int i = 0; i < cluster.numberOfPoints; i++)
        {
            gradientCluster1.getPoint(i) = cluster.getPoint(i) - gradient[i] * gradientStepSize;
            gradientCluster2.getPoint(i) = cluster.getPoint(i) - gradient[i] * (2.f * gradientStepSize);
        }
        
        // calculate the total cluster energies of these 3 clusters
        originalEnergy = cluster.getClusterEnergy(lennardJonesSquaredPotential);
        newEnergy1 = gradientCluster1.getClusterEnergy(lennardJonesSquaredPotential);
        newEnergy2 = gradientCluster2.getClusterEnergy(lennardJonesSquaredPotential);
        
        if ((2 * newEnergy2 - 4 * newEnergy1 + 2 * originalEnergy) == 0.f)
            break;
        float optimalDeflectionFactor = -(newEnergy2 - 4 * newEnergy1 + 3 * originalEnergy)/(2 * newEnergy2 - 4 * newEnergy1 + 2 * originalEnergy);

        // we now have the values at 3 points along the gradient direction. Fitting these points with a quadratic function yields an approximate optimal new cluster
        cluster.addToPoints(gradient, gradientStepSize * optimalDeflectionFactor);

        float newEnergy = cluster.getClusterEnergy(lennardJonesSquaredPotential);
        if(std::abs(newEnergy - originalEnergy) < 1e-6f)
            break;
    }
}

int FuzzyGlobalOptimizer::getRandomAtomByWeights(std::vector<float>& atomWeights) {
    discreteDistribution.updateDistribution(atomWeights);

    return discreteDistribution.generate(gen);
}
*/