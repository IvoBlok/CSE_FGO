#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <cstddef>

float lennardJonesPotential(float distance);
float lennardJonesSquaredPotential(float squaredDistance);
float lennardJonesDerivative(float distance);


struct DiscretePoint {
    int x, y, z;

    DiscretePoint() = default;
    DiscretePoint(const int x, const int y, const int z);
    explicit DiscretePoint(const int val);

    DiscretePoint operator+(const DiscretePoint& other) const;
    DiscretePoint operator-(const DiscretePoint& other) const;

    int& operator[](const size_t index);
    const int& operator[](const size_t index) const;

    int lengthSquared() const;
};


struct DiscreteCluster {
    std::vector<DiscretePoint> points;
    float gridStepSizeSquared;

    DiscreteCluster() = default;
    DiscreteCluster(const float gridStepSizeSquared, const size_t numberOfPoints);
    
    DiscreteCluster(const DiscreteCluster& other) = default;
    DiscreteCluster& operator=(const DiscreteCluster& other) = default;

    void setPoint(const size_t atomIndex, const int x, const int y, const int z);
    void setPoint(const size_t atomIndex, const DiscretePoint& point);

    DiscretePoint& getPoint(const size_t atomIndex);
    const DiscretePoint& getPoint(const size_t atomIndex) const;

    int getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const;

    float getAtomEnergy(const size_t atomIndex) const;
    float getAtomEnergy(const size_t atomIndex, const std::vector<size_t>& atomsToConsider) const;
    float getClusterEnergy() const;

    std::vector<size_t> getAtomNeighbours(const size_t atomIndex, const int cutoffDistanceSquared) const;

    void copyTo(DiscreteCluster& otherCluster) const;

    size_t size() const;
};


struct ContinuousPoint {
    float x, y, z;

    ContinuousPoint() = default;
    ContinuousPoint(const float x, const float y, const float z);
    explicit ContinuousPoint(const float val);
    explicit ContinuousPoint(const DiscretePoint& discretePoint, const float gridStepSize);

    ContinuousPoint operator+(const ContinuousPoint& other) const;
    ContinuousPoint operator-(const ContinuousPoint& other) const;
    ContinuousPoint operator*(float scalar) const;

    float& operator[](const size_t index);
    const float& operator[](const size_t index) const;

    float lengthSquared() const;
};


struct ContinuousCluster {
    std::vector<ContinuousPoint> points;
    
    ContinuousCluster() = default;
    explicit ContinuousCluster(const size_t numberOfPoints);
    explicit ContinuousCluster(const DiscreteCluster& discreteCluster, const float gridStepSize);

    ContinuousCluster(const ContinuousCluster& other) = default;
    ContinuousCluster& operator=(const ContinuousCluster& other) = default;

    void setPoint(const size_t atomIndex, const float x, const float y, const float z);
    void setPoint(const size_t atomIndex, const ContinuousPoint& point);

    ContinuousPoint& getPoint(const size_t atomIndex);
    const ContinuousPoint& getPoint(const size_t atomIndex) const;

    void addToPoints(const std::vector<ContinuousPoint>& deltas, const float factor);

    float getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const;

    float getAtomEnergy(const size_t atomIndex) const;
    float getAtomEnergy(const size_t atomIndex, const std::vector<size_t>& atomsToConsider) const;

    float getClusterEnergy() const;
    
    void copyTo(ContinuousCluster& otherCluster) const;

    size_t size() const;
};

#endif // DATA_STRUCTURES_H