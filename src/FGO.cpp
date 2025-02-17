#include "FGO.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>
#include <list>

# define M_PI           3.14159265358979323846  /* pi */

FuzzyGlobalOptimizer::FuzzyGlobalOptimizer(int numberOfAtoms, const std::vector<float>& LJLookup, float discreteGridSteps, float discreteCutoffDistance, float gradientStepSize) 
    :   numberOfAtoms(numberOfAtoms), 
        LJLookup(LJLookup),
        discreteGridSteps(discreteGridSteps), 
        discreteCutoffDistance(discreteCutoffDistance),
        gradientStepSize(gradientStepSize),
        lowestEnergyFound(0.f)
{ 
    // calculate initial cluster spawning radius, i.e. in what sphere of volume the atoms are 'spawned'
    spawningRadius = 0.4f * std::pow(numberOfAtoms, 0.333f);
    
    // sphere based random point generation initialization
    gen = std::minstd_rand0(rd());
    dist = std::uniform_real_distribution<>(0.0, 1.0);         // For uniform sampling
    distTheta = std::uniform_real_distribution<>(0.0, 2.f * M_PI); // Azimuthal angle
    distPhi = std::uniform_real_distribution<>(0.0, M_PI);     // Polar angle

    discreteDistribution = DiscreteDistribution{numberOfAtoms};
}

void FuzzyGlobalOptimizer::runFGO() {
    // ===============================================
    // STEP 1: create initial cluster
    currentCluster = DiscreteCluster{discreteGridSteps*discreteGridSteps, numberOfAtoms};
    generateInitialCluster(currentCluster, spawningRadius);
    localDiscreteOptimization(currentCluster);
    lowestEnergyFound = currentCluster.getClusterEnergy(LJLookup);

    // ===============================================
    // STEP 2: run first rough DMC layer, with experimentally determined hyperparameters as inputs
    discreteMonteCarlo(1.0f, -4.1f, 1.25f, 0.4f, 2.5f);

    // ===============================================
    // STEP 3: run second finer DMC layer, with experimentally determined hyperparameters as inputs
    discreteMonteCarlo(1.0f, -11.0f, 1.3f, 0.3f, 1.5f);

    // ===============================================
    // STEP 4: locally optimize all candidate clusters in the real space

    // get all candidates with low energies
    std::vector<ContinuousCluster> goodCandidates;

    for (size_t i = 0; i < candidateClusters.size(); i++)
    {
        if (candidateClusters[i].getClusterEnergy(LJLookup) < lowestEnergyFound + 2.f) {
            goodCandidates.emplace_back(ContinuousCluster{candidateClusters[i], discreteGridSteps});

            // optimize the appropriate candidates now in real 3D space
            localRealOptimization(goodCandidates.back());
        }
    }
    //std::cout << "candidates / low energy candidates: " << candidateClusters.size() << " / " << goodCandidates.size() << "\n";

    // TEMP: For debugging / development purposes, we retrieve the best real-optimized candidate
    for (size_t i = 0; i < goodCandidates.size(); i++)
    {   
        float energy = goodCandidates[i].getClusterEnergy(LeonardJonesSquaredPotential);
        if (energy < lowestEnergyFound)
            lowestEnergyFound = energy;
    }

    // ===============================================
    // STEP 5: Surface Monte Carlo (SMC). mainly important for larger clusters (>100)
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
        // Since the problem is tackled in reduced units, a distance of 1 is the optimum distance between two atoms (assuming no other atoms are in the cluster).
        currentCluster.copyInto(candidateCluster);
        setAtomInRandomSphere(candidateCluster, activeAtom, 1.f, currentCluster.getPoint(targetAtom), false);

        // locally optimize the modified cluster in the discrete space, while holding the rest of the cluster still
        localDiscreteFrozenOptimization(candidateCluster, activeAtom);

        // if the local energy of the moved atom improved, we directly accept the new candidate
        float deltaLocalAtomEnergy = candidateCluster.getAtomEnergy(LJLookup, activeAtom) - atomEnergies[activeAtom];

        float randomExpAcceptanceThreshold = dist(gen);

        if(deltaLocalAtomEnergy < 0.f || randomExpAcceptanceThreshold < std::exp(-deltaLocalAtomEnergy/acceptanceEnergy)) {
            
            localDiscreteOptimization(candidateCluster);
            
            // if the discrete local optimization (DLO) found a new best cluster, keep the candidate
            float candidateEnergy = candidateCluster.getClusterEnergy(LJLookup);
            if(candidateEnergy < lowestEnergyFound) {
                lowestEnergyFound = candidateEnergy;
                candidateClusters.emplace_back(candidateCluster);
                lastSinceImprovement = 0;
            } else {
                lastSinceImprovement++;
            }

            // now that we have a probably better cluster, make it the base cluster for the next iteration
            candidateCluster.copyInto(currentCluster);
        } else {
            // the move - local optimization combo chosen here didn't work out. keep the original cluster
            lastSinceImprovement++;
        }
    }
}

int FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(DiscreteCluster& cluster, int nonFrozenAtom) {
    // get all atoms within the cutoff range of the non-frozen atom
    std::vector<int> neighbours = cluster.getAtomNeighbours(nonFrozenAtom, std::pow(discreteCutoffDistance / discreteGridSteps, 2));

    return localDiscreteFrozenOptimization(cluster, nonFrozenAtom, neighbours);
}

int FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(DiscreteCluster& cluster, int nonFrozenAtom, std::vector<int>& neighbours) {
    float initialAtomEnergy = cluster.getAtomEnergy(LJLookup, nonFrozenAtom, neighbours);

    // nudge the x,y,z direction individually, untill changes in any result in worse energy
    float oldAtomEnergy = initialAtomEnergy;
    float newAtomEnergy = 0.f;
    int lastMoved = 0;
    int changes = 0;
    int axis = 0;
    DiscretePoint& nonFrozenPoint = cluster.getPoint(nonFrozenAtom);

    while (lastMoved < 3) {
        axis = (++axis) % 3;
        nonFrozenPoint[axis] += 1;
        newAtomEnergy = cluster.getAtomEnergy(LJLookup, nonFrozenAtom, neighbours);

        // if the nudge lowered, i.e. improved, the energy of this atom, accept the new position
        if(oldAtomEnergy - newAtomEnergy > 0.f) {
            oldAtomEnergy = newAtomEnergy;
            lastMoved = 0;
            changes++;
            continue;
        }
        
        // if the energy worsened in the positive direction, try the negative direction
        nonFrozenPoint[axis] -= 2;
        newAtomEnergy = cluster.getAtomEnergy(LJLookup, nonFrozenAtom, neighbours);

        if(oldAtomEnergy - newAtomEnergy > 0.f) {
            oldAtomEnergy = newAtomEnergy;
            lastMoved = 0;
            changes++;
            continue;
        }
        
        // if both directions worsen the outcome, reject the move in this axis, and note down that this axis wasn't succesfully nudged
        nonFrozenPoint[axis] += 1;
        lastMoved++;
    }

    return changes;
}

void FuzzyGlobalOptimizer::localDiscreteOptimization(DiscreteCluster& cluster) {
    // activeList is simply all atoms which will be optimized that iteration. At the start, all atoms are in the list
    std::list<int> activeList;
    for (int i = 0; i < cluster.numberOfPoints; i++)
        activeList.emplace_back(i);

    
    while(activeList.size() > 0) {
        activeList.remove_if([&cluster, this](int n){ return localDiscreteFrozenOptimization(cluster, n) == 0; });
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

    ContinuousCluster gradientCluster1;
    ContinuousCluster gradientCluster2;

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
                    direction = (direction * (1.f / distance)) * LeonardJonesDerivative(distance);
                    gradient[i] = gradient[i] + direction;
                }
            }
            gradient[i] = gradient[i] * (1.f / std::sqrt(gradient[i].lengthSquared())) * std::fmin(100.f, std::sqrt(gradient[i].lengthSquared()));
        }
        
        // create two new clusters, by moving in the opposite direction of the gradient
        gradientCluster1 = ContinuousCluster(cluster.numberOfPoints);
        gradientCluster2 = ContinuousCluster(cluster.numberOfPoints);

        for (int i = 0; i < cluster.numberOfPoints; i++)
        {
            gradientCluster1.getPoint(i) = cluster.getPoint(i) - gradient[i] * gradientStepSize;
            gradientCluster2.getPoint(i) = cluster.getPoint(i) - gradient[i] * (2.f * gradientStepSize);
        }
        
        // calculate the total cluster energies of these 3 clusters
        originalEnergy = cluster.getClusterEnergy(LeonardJonesSquaredPotential);
        newEnergy1 = gradientCluster1.getClusterEnergy(LeonardJonesSquaredPotential);
        newEnergy2 = gradientCluster2.getClusterEnergy(LeonardJonesSquaredPotential);
        
        if ((2 * newEnergy2 - 4 * newEnergy1 + 2 * originalEnergy) == 0.f)
            break;
        float optimalDeflectionFactor = -(newEnergy2 - 4 * newEnergy1 + 3 * originalEnergy)/(2 * newEnergy2 - 4 * newEnergy1 + 2 * originalEnergy);

        // we now have the values at 3 points along the gradient direction. Fitting these points with a quadratic function yields an approximate optimal new cluster
        cluster.addToPoints(gradient, gradientStepSize * optimalDeflectionFactor);
        //cluster.addToPoints(gradient, -gradientStepSize);

        float newEnergy = cluster.getClusterEnergy(LeonardJonesSquaredPotential);
        if(std::abs(newEnergy - originalEnergy) < 1e-6f * std::abs(originalEnergy))
            break;
    }
}

int FuzzyGlobalOptimizer::getRandomAtomByWeights(std::vector<float>& atomWeights) {
    discreteDistribution.updateDistribution(atomWeights);

    return discreteDistribution.generate(gen);
}

DiscretePoint FuzzyGlobalOptimizer::generateUniformRandomPointInSphere(float radius, DiscretePoint center, bool allowZero = true) {

    float u, theta, phi, r;
    int x, y, z;

    // Generate random spherical coordinates
    u = dist(gen);
    r = radius * std::cbrt(u);
    theta = distTheta(gen);
    phi = distPhi(gen);

    // Convert to Cartesian coordinates        
    x = static_cast<int>(std::round(r * std::sin(phi) * std::cos(theta) / discreteGridSteps));
    y = static_cast<int>(std::round(r * std::sin(phi) * std::sin(theta) / discreteGridSteps));
    z = static_cast<int>(std::round(r * std::cos(phi) / discreteGridSteps));

    if(!allowZero && x == 0 && y == 0 && z == 0)
        generateUniformRandomPointInSphere(radius, center, allowZero);

    return DiscretePoint{center.x + x, center.y + y, center.z + z};
}

void FuzzyGlobalOptimizer::setAtomInRandomSphere(DiscreteCluster& cluster, int atomIndex, float radius, DiscretePoint center, bool allowZero = true) {
    DiscretePoint randomPoint = generateUniformRandomPointInSphere(radius, center, allowZero);
    
    cluster.setPoint(atomIndex, randomPoint);
}

void FuzzyGlobalOptimizer::generateInitialCluster(DiscreteCluster& cluster, float radius) {
    // assumed is that the cubic grid has equal radius to the spawning sphere. Thus the center of the cube can be retrieved from the sphere radius
    // here the case of points starting on identical locations is ignored, though this might cause issues later
    for (size_t i = 0; i < numberOfAtoms; i++)
        setAtomInRandomSphere(cluster, i, radius, DiscretePoint{static_cast<int>(std::round(radius / discreteGridSteps))});
}