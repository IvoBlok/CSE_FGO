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

    auto& candidate = state.candidates.emplace_back(Cluster(), std::numeric_limits<float>::infinity());

    initializeCluster(candidate.first, rng);
    localDiscreteOptimization(candidate.first);
    candidate.second = candidate.first.getClusterEnergy();

    auto startDMC = std::chrono::high_resolution_clock::now();
    runDMCLayer(state, params.dmcLayer1, rng);
    auto endDMC = std::chrono::high_resolution_clock::now();

    // TODO DMC2

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

void FuzzyGlobalOptimizer::localDiscreteOptimization(Cluster& cluster) {
    std::list<size_t> activeList;
    for (size_t i = 0; i < cluster.size(); i++)
        activeList.emplace_back(i);

    while (!activeList.empty())
        activeList.remove_if([&cluster, this](int i){ return localDiscreteFrozenOptimization(cluster, i) == 0; });
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
        // calculate gradient
        std::fill(gradX.begin(), gradX.end(), 0.0f);
        std::fill(gradY.begin(), gradY.end(), 0.0f);
        std::fill(gradZ.begin(), gradZ.end(), 0.0f);

        for (size_t i = 0; i < n; i++)
        {
            const float xi = cluster.x[i];
            const float yi = cluster.y[i];
            const float zi = cluster.z[i];

            for (size_t j = i + 1; j < n; j++)
            {
                const float dx = xi - cluster.x[j];
                const float dy = yi - cluster.y[j];
                const float dz = zi - cluster.z[j];

                const float r = dx*dx + dy*dy + dz*dz;
                const float inv_r = 1.0 / (std::sqrt(r));
                const float force = lennardJonesDerivative(r) * inv_r;

                gradX[i] += dx * force;
                gradY[i] += dy * force;
                gradZ[i] += dz * force;

                gradX[j] -= dx * force;
                gradY[j] -= dy * force;
                gradZ[j] -= dz * force;
            }
        }

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
        const float E1 = gradCluster1.getClusterEnergy();
        const float E2 = gradCluster2.getClusterEnergy();
        
        // interpolate quadratic equation
        const float denom = 2.0f * E2 - 4.0f * E1 + 2.0f * E0;
        if (std::abs(denom) < 1e-7f) break;

        const float alpha = (-3.0f * E0 + 4.0f * E1 - E2) / denom;
        
        // move to minimum of fitted quadratic
        for (size_t i = 0; i < n; ++i) {
            cluster.x[i] -= gradX[i] * (step * alpha);
            cluster.y[i] -= gradY[i] * (step * alpha);
            cluster.z[i] -= gradZ[i] * (step * alpha);
        }

        // convergence condition
        candidate.second = cluster.getClusterEnergy();
        if (std::abs(candidate.second - E0) < 1e-6f) break;
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
    float& x = cluster.x[freeIndex];
    float& y = cluster.y[freeIndex];
    float& z = cluster.z[freeIndex];

    float currentEnergy = cluster.getAtomEnergy(freeIndex);
    size_t numChanges = 0;
    bool improved = true;

    const float moves[6][3] = {
        {params.gridSpacing, 0, 0},
        {-params.gridSpacing, 0, 0},
        {0, params.gridSpacing, 0},
        {0, -params.gridSpacing, 0},
        {0, 0, params.gridSpacing},
        {0, 0, -params.gridSpacing}
    };

    while (improved) {
        improved = false;

        for (int moveIdx = 0; moveIdx < 6; ++moveIdx) {
            x += moves[moveIdx][0];
            y += moves[moveIdx][1];
            z += moves[moveIdx][2];

            float newEnergy = cluster.getAtomEnergy(freeIndex);

            if (newEnergy - currentEnergy < 0.0f) {
                currentEnergy = newEnergy;
                improved = true;
                numChanges++;
                break;
            } else {
                x -= moves[moveIdx][0];
                y -= moves[moveIdx][1];
                z -= moves[moveIdx][2];
            }
        }
    }

    return numChanges;
}