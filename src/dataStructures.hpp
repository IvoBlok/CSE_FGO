#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <cstddef>

float lennardJonesPotential(float distance);
float lennardJonesSquaredPotential(float squaredDistance);
float lennardJonesDerivative(float distance);


struct Cluster {
    std::vector<float> x, y, z;
    size_t n = 0;
    
    Cluster() = default;
    explicit Cluster(const size_t numberOfPoints);

    Cluster(const Cluster& other) = default;
    Cluster& operator=(const Cluster& other) = default;

    void setPoint(const size_t atomIndex, const float xVal, const float yVal, const float zVal);

    float getDistanceSquared(size_t atomIndex1, size_t atomIndex2) const;

    float getAtomEnergy(size_t atomIndex) const;
    float getAtomEnergy(size_t atomIndex, const std::vector<size_t>& atomsToConsider) const;

    float getClusterEnergy() const;
    
    void copyTo(Cluster& otherCluster) const;

    size_t size() const { return n; };
};

#endif // DATA_STRUCTURES_H