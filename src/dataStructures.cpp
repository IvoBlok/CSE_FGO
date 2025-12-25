#include "dataStructures.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>



float lennardJonesPotential(float distance) {
    if(distance == 0.f)
        return std::numeric_limits<float>::infinity();

    // using 'reduced' units, the LJ potential is simply:
    return std::pow(distance, -12) - 2.f * std::pow(distance, -6);
}

float lennardJonesSquaredPotential(float squaredDistance) {
    if(squaredDistance < 0.002f)
        return std::numeric_limits<float>::infinity();

    // defines the LJ potential based on a squared distance input. It saves some computation
    return std::pow(squaredDistance, -6) - 2.f * std::pow(squaredDistance, -3);
}

float lennardJonesDerivative(float distance) {
    return -12.f * (std::pow(distance, -13) - std::pow(distance, -7));
}



// Cluster Implementation
// ===================================================================================
Cluster::Cluster(const size_t numberOfPoints) : x(numberOfPoints), y(numberOfPoints), z(numberOfPoints), n(numberOfPoints) {}

void Cluster::setPoint(const size_t atomIndex, const float xVal, const float yVal, const float zVal) {
    x[atomIndex] = xVal;
    y[atomIndex] = yVal;
    z[atomIndex] = zVal;
}

float Cluster::getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const {
    const float dx = x[atomIndex1] - x[atomIndex2];
    const float dy = y[atomIndex1] - y[atomIndex2];
    const float dz = z[atomIndex1] - z[atomIndex2];
    return dx*dx + dy*dy + dz*dz;
}

float Cluster::getAtomEnergy(size_t atomIndex) const {
    float total = 0.f;

    for (size_t j = 0; j < n; j++)
    {   
        if (atomIndex != j) {
            const float squaredDistance = getDistanceSquared(atomIndex, j);
            total += lennardJonesSquaredPotential(squaredDistance);
        }
    }
    return total;
}

float Cluster::getAtomEnergy(size_t atomIndex, const std::vector<size_t>& atomsToConsider) const {
    float total = 0.f;

    for (const size_t& atom : atomsToConsider)
    {
        if (atomIndex != atom) {
            const float squaredDistance = getDistanceSquared(atomIndex, atom);
            total += lennardJonesSquaredPotential(squaredDistance);
        }
    }
    return total;
}

float Cluster::getClusterEnergy() const {
    float total = 0.f;

    for (size_t i = 0; i < n; i++)
        total += getAtomEnergy(i);

    return total * 0.5f;    
}

void Cluster::copyTo(Cluster& otherCluster) const {
    otherCluster.x = x;
    otherCluster.y = y;
    otherCluster.z = z;
    otherCluster.n = n;
}