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



// DiscretePoint Implementation
// ===================================================================================
DiscretePoint::DiscretePoint(const int x, const int y, const int z) : x(x), y(y), z(z) {}

DiscretePoint::DiscretePoint(const int val) : x(val), y(val), z(val) {}

DiscretePoint DiscretePoint::operator+(const DiscretePoint& other) const {
    return DiscretePoint(x + other.x, y + other.y, z + other.z);
}

DiscretePoint DiscretePoint::operator-(const DiscretePoint& other) const {
    return DiscretePoint(x - other.x, y - other.y, z - other.z);
}

int& DiscretePoint::operator[](const size_t index) {
    if (index >= 3) 
        throw std::out_of_range("Index out of range. Valid indices are 0, 1, and 2\n");

    return (&x)[index];
}

const int& DiscretePoint::operator[](const size_t index) const {
    if (index >= 3) 
        throw std::out_of_range("Index out of range. Valid indices are 0, 1, and 2\n");

    return (&x)[index];
}

int DiscretePoint::lengthSquared() const {
    return x*x + y*y + z*z;
}



// DiscreteCluster Implementation
// ===================================================================================
DiscreteCluster::DiscreteCluster(const float gridStepSizeSquared, const size_t numberOfPoints) : gridStepSizeSquared(gridStepSizeSquared), points(numberOfPoints) {}

void DiscreteCluster::setPoint(const size_t atomIndex, const int x, const int y, const int z) {
    points[atomIndex] = {x, y, z};
}

void DiscreteCluster::setPoint(const size_t atomIndex, const DiscretePoint& point) {
    points[atomIndex] = point;
}

DiscretePoint& DiscreteCluster::getPoint(const size_t atomIndex) {
    return points[atomIndex];
}

const DiscretePoint& DiscreteCluster::getPoint(const size_t atomIndex) const {
    return points[atomIndex];
}

int DiscreteCluster::getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const {
    const auto& p1 = points[atomIndex1];
    const auto& p2 = points[atomIndex2];
    const int dx = p1.x - p2.x;
    const int dy = p1.y - p2.y;
    const int dz = p1.z - p2.z;
    return dx*dx + dy*dy + dz*dz;
}

float DiscreteCluster::getAtomEnergy(const size_t atomIndex) const {
    float total = 0.f;

    for (size_t j = 0; j < points.size(); j++)
    {   
        if (atomIndex != j) {
            const int squaredDistance = getDistanceSquared(atomIndex, j);
            total += lennardJonesSquaredPotential(squaredDistance * gridStepSizeSquared);
        }
    }
    return total;
}

float DiscreteCluster::getAtomEnergy(const size_t atomIndex, const std::vector<size_t>& atomsToConsider) const {
    float total = 0.f;

    for (const size_t& atom : atomsToConsider)
    {
        if (atomIndex != atom) {
            const int squaredDistance = getDistanceSquared(atomIndex, atom);
            total += lennardJonesSquaredPotential(squaredDistance * gridStepSizeSquared);
        }
    }
    return total;
}

float DiscreteCluster::getClusterEnergy() const {
    float total = 0.f;

    for (size_t i = 0; i < points.size(); i++)
        total += getAtomEnergy(i);

    return total * 0.5f; // each pair gets counted twice, so half the total to account for this
}

std::vector<size_t> DiscreteCluster::getAtomNeighbours(const size_t atomIndex, const int cutoffDistanceSquared) const {
    std::vector<size_t> neighbours;
    neighbours.reserve(points.size());

    for (size_t i = 0; i < points.size(); i++) {
        if (i == atomIndex) continue;
        if (getDistanceSquared(atomIndex, i) < cutoffDistanceSquared)
            neighbours.emplace_back(i);
    }

    return neighbours;
}

void DiscreteCluster::copyTo(DiscreteCluster& otherCluster) const {
    otherCluster.gridStepSizeSquared = gridStepSizeSquared;
    otherCluster.points = points;
}

size_t DiscreteCluster::size() const { return points.size(); }



// ContinuousPoint Implementation
// ===================================================================================
ContinuousPoint::ContinuousPoint(const float x, const float y, const float z) : x(x), y(y), z(z) {}

ContinuousPoint::ContinuousPoint(const float val) : x(val), y(val), z(val) {}

ContinuousPoint::ContinuousPoint(const DiscretePoint& discretePoint, const float gridStepSize) 
    : x(discretePoint.x * gridStepSize), y(discretePoint.y * gridStepSize), z(discretePoint.z * gridStepSize) {}

ContinuousPoint ContinuousPoint::operator+(const ContinuousPoint& other) const {
    return ContinuousPoint(x + other.x, y + other.y, z + other.z);
}

ContinuousPoint ContinuousPoint::operator-(const ContinuousPoint& other) const {
    return ContinuousPoint(x - other.x, y - other.y, z - other.z);
}

ContinuousPoint ContinuousPoint::operator*(float scalar) const {
    return ContinuousPoint(x * scalar, y * scalar, z * scalar);
}

ContinuousPoint ContinuousPoint::operator/(float scalar) const {
    return ContinuousPoint(x / scalar, y / scalar, z / scalar);
}


float& ContinuousPoint::operator[](const size_t index) {
    if (index >= 3) 
        throw std::out_of_range("Index out of range. Valid indices are 0, 1, and 2\n");

    return (&x)[index];
}

const float& ContinuousPoint::operator[](const size_t index) const {
    if (index >= 3) 
        throw std::out_of_range("Index out of range. Valid indices are 0, 1, and 2\n");

    return (&x)[index];
}

float ContinuousPoint::lengthSquared() const {
    return x*x + y*y + z*z;
}



// ContinuousCluster Implementation
// ===================================================================================
ContinuousCluster::ContinuousCluster(const size_t numberOfPoints) : points(numberOfPoints) {}

ContinuousCluster::ContinuousCluster(const DiscreteCluster& discreteCluster, const float gridStepSize) : points(discreteCluster.points.size()) {
    for (size_t i = 0; i < points.size(); i++)
        points[i] = ContinuousPoint{discreteCluster.getPoint(i), gridStepSize};
}

void ContinuousCluster::setPoint(const size_t atomIndex, const float x, const float y, const float z) {
    points[atomIndex] = {x, y, z};
}

void ContinuousCluster::setPoint(const size_t atomIndex, const ContinuousPoint& point) {
    points[atomIndex] = point;
}

ContinuousPoint& ContinuousCluster::getPoint(const size_t atomIndex) {
    return points[atomIndex];
}

const ContinuousPoint& ContinuousCluster::getPoint(const size_t atomIndex) const {
    return points[atomIndex];
}

void ContinuousCluster::addToPoints(const std::vector<ContinuousPoint>& deltas, const float factor) {
    if (points.size() != deltas.size()) 
        throw std::invalid_argument("addToPoints: size mismatch");

    for (size_t i = 0; i < points.size(); i++)
        points[i] = points[i] + deltas[i] * factor;
}

float ContinuousCluster::getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const {
    const auto& p1 = points[atomIndex1];
    const auto& p2 = points[atomIndex2];
    const float dx = p1.x - p2.x;
    const float dy = p1.y - p2.y;
    const float dz = p1.z - p2.z;
    return dx*dx + dy*dy + dz*dz;
}

float ContinuousCluster::getAtomEnergy(const size_t atomIndex) const {
    float total = 0.f;

    for (size_t j = 0; j < points.size(); j++)
    {   
        if (atomIndex != j) {
            const float squaredDistance = getDistanceSquared(atomIndex, j);
            total += lennardJonesSquaredPotential(squaredDistance);
        }
    }
    return total;
}

float ContinuousCluster::getAtomEnergy(const size_t atomIndex, const std::vector<size_t>& atomsToConsider) const {
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

float ContinuousCluster::getClusterEnergy() const {
    float total = 0.f;

    for (size_t i = 0; i < points.size(); i++)
        total += getAtomEnergy(i);

    return total * 0.5f;    
}

void ContinuousCluster::copyTo(ContinuousCluster& otherCluster) const {
    otherCluster.points = points;
}

size_t ContinuousCluster::size() const { return points.size(); }