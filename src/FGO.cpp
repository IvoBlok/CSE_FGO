#include "FGO.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>

float LeonardJonespotential(float distance) {
    // using 'reduced' units, the LJ potential is simply:
    return std::pow(distance, -12) - 2 * std::pow(distance, -6);
}

float LeonardJonesSquaredPotential(float squaredDistance) {
    // defines the LJ potential based on a squared distance input. It saves some computation
    return std::pow(squaredDistance, -6) - 2 * std::pow(squaredDistance, -3);
}

FuzzyGlobalOptimizer::FuzzyGlobalOptimizer(int numberOfAtoms, float discreteGridSteps, float discreteCutoffDistance, float realLOGradientStep) 
    :   numberOfAtoms(numberOfAtoms), 
        discreteGridSteps(discreteGridSteps), 
        discreteCutoffDistance(discreteCutoffDistance),
        realLOGradientStep(realLOGradientStep),
        LJPotentialsLookup(nullptr),
        lowestEnergyFound(0.f)
{ 

    // calculate initial cluster spawning radius, i.e. in what sphere of volume the atoms are 'spawned'
    spawningRadius = 0.4f * std::pow(numberOfAtoms, 0.333f);

    // the discrete grid is stored in a special way. Since all atoms lay on discrete points, we can store their position as integers, refering to which discrete point along the axes the atom is located at
    // Since a large portion of the calculations to be done rely on knowing the Leonard-Jones interaction energy between two atoms, these can be precomputed for all possible distances on the grid
    
    // as a size of the square box of the discrete grid, we use the same radius as used for the spawning process
    discreteGridPointCount = std::floor((2.f * spawningRadius) / discreteGridSteps) + 1;

    // Since the LJ potential only depends on the distance between two points, we can precalculate all options. To keep the access time low, we simply create an array, 
    // with all distances 0 up to 3*(discreteGridPointCount - 1)^2; i.e. the max distance squared possible in the grid. 
    // Though there are only (2 * discreteGridPointCount - 1) unique distances, to keep code simple, and access time quick, all integers in the range have their respective LJ potential calculated
    // TODO: In the future, this is probably the easiest to improve on. Even for smaller numbers of atoms, the fraction of existing distances, and the amount of integers in the range, is absurd. 
    maxGridSquaredDistance = 3 * std::pow(discreteGridPointCount - 1, 3);
    LJPotentialsLookup = new float[maxGridSquaredDistance + 1];

    // fill the array. Each index corresponds to the LJ potential with distance sqrt(index)
    LJPotentialsLookup[0] = std::numeric_limits<float>::infinity();
    for (int i = 1; i < maxGridSquaredDistance + 1; i++)
        LJPotentialsLookup[i] = LeonardJonesSquaredPotential(i*discreteGridSteps*discreteGridSteps);
}

void FuzzyGlobalOptimizer::runFGO() {

    // TESTING: 
    currentCluster = DiscreteCluster(numberOfAtoms);
    generateInitialCluster(currentCluster, spawningRadius);

    ContinuousCluster continuousCluster = ContinuousCluster(currentCluster, discreteGridSteps);
    localRealOptimization(continuousCluster);
    lowestEnergyFound = continuousCluster.getClusterEnergy(LeonardJonesSquaredPotential);
    /*
    // ===============================================
    // STEP 1: create initial cluster

    currentCluster = DiscreteCluster(numberOfAtoms);
    generateInitialCluster(currentCluster, spawningRadius);
    localDiscreteOptimization(currentCluster);

    // ===============================================
    // STEP 2: run first rough DMC layer, with experimentally determined hyperparameters as inputs
    discreteMonteCarlo(1.0f, -4.1f, 1.25f, 0.4f, 3.5f);

    // ===============================================
    // STEP 3: run second finer DMC layer, with experimentally determined hyperparameters as inputs
    discreteMonteCarlo(1.0f, -11.0f, 1.3f, 0.3f, 1.5f);

    // ===============================================
    // STEP 4: locally optimize all candidate clusters in the real space

    // get all candidates with low energies
    std::vector<ContinuousCluster> goodCandidates;

    for (size_t i = 0; i < candidateClusters.size(); i++)
    {
        if (candidateClusters[i].getClusterEnergy(LJPotentialsLookup) < lowestEnergyFound + 2.f) {
            goodCandidates.emplace_back(ContinuousCluster{candidateClusters[i], discreteGridSteps});

            // optimize the appropriate candidates now in real 3D space
            localRealOptimization(goodCandidates.back());
        }
    }

    // TEMP: For debugging / development purposes, we retrieve the best real-optimized candidate
    float newLowestEnergy = lowestEnergyFound;
    for (size_t i = 0; i < goodCandidates.size(); i++)
    {   
        float energy = goodCandidates[i].getClusterEnergy(LeonardJonesSquaredPotential);
        if (energy < newLowestEnergy) {
            newLowestEnergy = energy;
            goodCandidates[i].writeClusterToFile("../results/bestCandidate.xyz");
        }
    }
    std::cout << "E_opt_bef: " << lowestEnergyFound << " E_opt_aft: " << newLowestEnergy << "\n";
    lowestEnergyFound = newLowestEnergy;    
    */
    // ===============================================
    // STEP 5: Surface Monte Carlo (SMC). mainly important for larger clusters (>100)
    // TODO SMC
}

void FuzzyGlobalOptimizer::discreteMonteCarlo(float activeEnergy, float targetEnergy, float targetSigma, float acceptanceEnergy, float convergenceFactor) {
    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_real_distribution<float> distribution(0.f, 1.f);

    // currentCluster.writeClusterToFile(LJPotentialsLookup, "../results/initialCluster.xyz");


    int lastSinceImprovement = 0;
    int iteration = 0;

    DiscreteCluster candidateCluster{numberOfAtoms};

    while (lastSinceImprovement < (int)(numberOfAtoms*numberOfAtoms*convergenceFactor))
    {
        iteration++;
        // calculate individual atom energy for each atom in the cluster
        float* atomEnergies = new float[numberOfAtoms];
        int squaredDistance;

        for (size_t i = 0; i < numberOfAtoms; i++)
            atomEnergies[i] = currentCluster.getAtomEnergy(LJPotentialsLookup, i);

        // calculate probabilities of the atoms being chosen as active or as target
        std::vector<float> atomActiveWeights;
        std::vector<float> atomTargetWeights;
        atomActiveWeights.resize(numberOfAtoms);
        atomTargetWeights.resize(numberOfAtoms);

        (numberOfAtoms);
        for (size_t i = 0; i < numberOfAtoms; i++)
        {
            atomActiveWeights[i] = std::exp(atomEnergies[i]/activeEnergy);
            atomTargetWeights[i] = std::exp(-std::pow(atomEnergies[i] - targetEnergy, 2)/(2*std::pow(targetSigma, 2)));
        }

        // get an active and target atom
        int activeAtom = getRandomAtomByWeights(atomActiveWeights);
        int targetAtom = getRandomAtomByWeights(atomTargetWeights);

        // make a new candidate cluster, with the active atomed moved to the area around the target atom, in a sphere of radius 1.
        // Since the problem is tackled in reduced units, a distance of 1 is the optimum distance between two atoms (assuming no other atoms are in the cluster).
        currentCluster.copyInto(candidateCluster);
        setAtomInRandomSphere(candidateCluster, activeAtom, 1.f, currentCluster.getPoint(targetAtom), false);

        // locally optimize the modified cluster in the discrete space, while holding the rest of the cluster still
        localDiscreteFrozenOptimization(candidateCluster, activeAtom);

        // if the local energy of the moved atom improved, we directly accept the new candidate
        float deltaLocalAtomEnergy = candidateCluster.getAtomEnergy(LJPotentialsLookup, activeAtom) - atomEnergies[activeAtom];

        float randomExpAcceptanceThreshold = distribution(generator);

        if(deltaLocalAtomEnergy < -0.f || randomExpAcceptanceThreshold < std::exp(-deltaLocalAtomEnergy/acceptanceEnergy)) {
            
            localDiscreteOptimization(candidateCluster);
            
            // if the discrete local optimization (DLO) found a new best cluster, keep the candidate
            float candidateEnergy = candidateCluster.getClusterEnergy(LJPotentialsLookup);
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

    // currentCluster.writeClusterToFile(LJPotentialsLookup, "../results/finalCluster.xyz");
    //std::cout << "DMC finished in " << iteration << " iterations\n";
    //std::cout << "N =" << numberOfAtoms << " Final Cluster Energy: " << currentCluster.getClusterEnergy(LJPotentialsLookup) << "\n";
}

void FuzzyGlobalOptimizer::localDiscreteFrozenOptimization(DiscreteCluster& cluster, int nonFrozenAtom) {

    // get all atoms within the cutoff range of the non-frozen atom
    std::vector<int> neighbours = cluster.getAtomNeighbours(nonFrozenAtom, std::pow(discreteCutoffDistance / discreteGridSteps, 2));

    float initialAtomEnergy = cluster.getAtomEnergy(LJPotentialsLookup, nonFrozenAtom, neighbours);

    // nudge the x,y,z direction individually, untill changes in any result in worse energy
    float oldAtomEnergy = initialAtomEnergy;
    float newAtomEnergy = 0.f;
    int lastMoved = 0;
    int axis = 0;
    DiscretePoint& nonFrozenPoint = cluster.getPoint(nonFrozenAtom);

    while (lastMoved < 3) {
        axis = (++axis) % 3;
        nonFrozenPoint[axis] += 1;
        newAtomEnergy = cluster.getAtomEnergy(LJPotentialsLookup, nonFrozenAtom, neighbours);

        // if the nudge lowered, i.e. improved, the energy of this atom, accept the new position
        if(oldAtomEnergy - newAtomEnergy > 0.f) {
            oldAtomEnergy = newAtomEnergy;
            lastMoved = 0;
            continue;
        }
        
        // if the energy worsened in the positive direction, try the negative direction
        nonFrozenPoint[axis] -= 2;
        newAtomEnergy = cluster.getAtomEnergy(LJPotentialsLookup, nonFrozenAtom, neighbours);

        if(oldAtomEnergy - newAtomEnergy > 0.f) {
            oldAtomEnergy = newAtomEnergy;
            lastMoved = 0;
            continue;
        }
        
        // if both directions worsen the outcome, reject the move in this axis, and note down that this axis wasn't succesfully nudged
        nonFrozenPoint[axis] += 1;
        lastMoved++;
    }
}

void FuzzyGlobalOptimizer::localDiscreteOptimization(DiscreteCluster& cluster) {
    // the paper isn't very clear about the logic in this function. For now I'll assume the most basic interpretation; This just runs 'frozen' discrete optimization once for each atom in the cluster
    // TODO: investigate if this is actually what the paper intended, and separately, what then an improvement can be here
    for (int i = 0; i < cluster.numberOfPoints; i++)
        localDiscreteFrozenOptimization(cluster, i);
}

void FuzzyGlobalOptimizer::localRealOptimization(ContinuousCluster& cluster) {
    // gradient is initialized with 0 vector
    ContinuousPoint* gradient = new ContinuousPoint[cluster.numberOfPoints];
    float distanceSquared;
    float distance;
    ContinuousPoint direction;

    ContinuousCluster gradientCluster1;
    ContinuousCluster gradientCluster2;

    float originalEnergy, newEnergy1, newEnergy2;
    float lastOriginalEnergy = 1000000000.f;

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
                    direction = (direction * (1.f / distance)) * (-12.f * (std::pow(distance, -13) - std::pow(distance, -7)));
                    //direction = (direction * (1.f / distance)) * std::fmax(-100.f, (-12.f * (std::pow(distance, -13) - std::pow(distance, -7))));
                    //direction = direction * (-12.f * (std::pow(distanceSquared, -7) - std::pow(distanceSquared, -4)));
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
            gradientCluster1.getPoint(i) = cluster.getPoint(i) - gradient[i] * realLOGradientStep;
            gradientCluster2.getPoint(i) = cluster.getPoint(i) - gradient[i] * (2.f * realLOGradientStep);
        }
        
        // calculate the total cluster energies of these 3 clusters
        originalEnergy = cluster.getClusterEnergy(LeonardJonesSquaredPotential);
        newEnergy1 = gradientCluster1.getClusterEnergy(LeonardJonesSquaredPotential);
        newEnergy2 = gradientCluster2.getClusterEnergy(LeonardJonesSquaredPotential);

        float optimalDeflectionFactor = -(newEnergy2 - 4 * newEnergy1 + 3 * originalEnergy)/(2 * newEnergy2 - 4 * newEnergy1 + 2 * originalEnergy);

        // we now have the values at 3 points along the gradient direction. Fitting these points with a quadratic function yields an approximate optimal new cluster
        cluster.addToPoints(gradient, realLOGradientStep * optimalDeflectionFactor);
        //cluster.addToPoints(gradient, -realLOGradientStep);

        float newEnergy = cluster.getClusterEnergy(LeonardJonesSquaredPotential);
        if(std::abs(newEnergy - originalEnergy) < 0.001)
            break;
    }
}

int FuzzyGlobalOptimizer::getRandomAtomByWeights(std::vector<float>& atomWeights) {
    std::default_random_engine generator;
    std::discrete_distribution<int> distribution(atomWeights.begin(), atomWeights.end());

    return distribution(generator);
}

DiscretePoint FuzzyGlobalOptimizer::generateUniformRandomPointInSphere(float radius, DiscretePoint center, bool allowZero = true) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);         // For uniform sampling
    std::uniform_real_distribution<> distTheta(0.0, 2 * M_PI); // Azimuthal angle
    std::uniform_real_distribution<> distPhi(0.0, M_PI);     // Polar angle

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