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

// Point Implementation
// ===================================================================================
Point::Point(const float x, const float y, const float z) : x(x), y(y), z(z) {}

Point::Point(const float val) : x(val), y(val), z(val) {}

Point Point::operator+(const Point& other) const {
    return Point(x + other.x, y + other.y, z + other.z);
}

Point Point::operator-(const Point& other) const {
    return Point(x - other.x, y - other.y, z - other.z);
}

Point Point::operator*(float scalar) const {
    return Point(x * scalar, y * scalar, z * scalar);
}

Point Point::operator/(float scalar) const {
    return Point(x / scalar, y / scalar, z / scalar);
}


float& Point::operator[](const size_t index) {
    if (index >= 3) 
        throw std::out_of_range("Index out of range. Valid indices are 0, 1, and 2\n");

    return (&x)[index];
}

const float& Point::operator[](const size_t index) const {
    if (index >= 3) 
        throw std::out_of_range("Index out of range. Valid indices are 0, 1, and 2\n");

    return (&x)[index];
}

float Point::lengthSquared() const {
    return x*x + y*y + z*z;
}



// Cluster Implementation
// ===================================================================================
Cluster::Cluster(const size_t numberOfPoints) : points(numberOfPoints) {}

void Cluster::setPoint(const size_t atomIndex, const float x, const float y, const float z) {
    points[atomIndex] = {x, y, z};
}

void Cluster::setPoint(const size_t atomIndex, const Point& point) {
    points[atomIndex] = point;
}

Point& Cluster::getPoint(const size_t atomIndex) {
    return points[atomIndex];
}

const Point& Cluster::getPoint(const size_t atomIndex) const {
    return points[atomIndex];
}

void Cluster::addToPoints(const std::vector<Point>& deltas, const float factor) {
    if (points.size() != deltas.size()) 
        throw std::invalid_argument("addToPoints: size mismatch");

    for (size_t i = 0; i < points.size(); i++)
        points[i] = points[i] + deltas[i] * factor;
}

float Cluster::getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const {
    const auto& p1 = points[atomIndex1];
    const auto& p2 = points[atomIndex2];
    const float dx = p1.x - p2.x;
    const float dy = p1.y - p2.y;
    const float dz = p1.z - p2.z;
    return dx*dx + dy*dy + dz*dz;
}

float Cluster::getAtomEnergy(const size_t atomIndex) const {
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

float Cluster::getAtomEnergy(const size_t atomIndex, const std::vector<size_t>& atomsToConsider) const {
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

    for (size_t i = 0; i < points.size(); i++)
        total += getAtomEnergy(i);

    return total * 0.5f;    
}

void Cluster::copyTo(Cluster& otherCluster) const {
    otherCluster.points = points;
}

size_t Cluster::size() const { return points.size(); }