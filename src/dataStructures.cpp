#include "dataStructures.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>

float LeonardJonespotential(float distance) {
    if(distance = 0.f)
        return std::numeric_limits<float>::infinity();

    // using 'reduced' units, the LJ potential is simply:
    return 4.f * (std::pow(distance, -12) - std::pow(distance, -6));
}

float LeonardJonesSquaredPotential(float squaredDistance) {
    if(squaredDistance < 0.002f)
        return std::numeric_limits<float>::infinity();

    // defines the LJ potential based on a squared distance input. It saves some computation
    return 4.f * (std::pow(squaredDistance, -6) - std::pow(squaredDistance, -3));
}


ContinuousPoint::ContinuousPoint(float x, float y, float z) : x(x), y(y), z(z) { }

ContinuousPoint::ContinuousPoint(float x) : x(x), y(x), z(x) { }

ContinuousPoint::ContinuousPoint(DiscretePoint& discretePoint, float gridStepSize) {
    x = discretePoint.x * gridStepSize;
    y = discretePoint.y * gridStepSize;
    z = discretePoint.z * gridStepSize;
}

ContinuousPoint ContinuousPoint::operator+(const ContinuousPoint& other) const {
    return ContinuousPoint(x + other.x, y + other.y, z + other.z);
}

ContinuousPoint ContinuousPoint::operator-(const ContinuousPoint& other) const {
    return ContinuousPoint(x - other.x, y - other.y, z - other.z);
}

ContinuousPoint ContinuousPoint::operator*(float scalar) const {
    return ContinuousPoint(x * scalar, y * scalar, z * scalar);
}

ContinuousPoint ContinuousPoint::operator*(double scalar) const {
    return ContinuousPoint(x * scalar, y * scalar, z * scalar);
}

ContinuousPoint::ContinuousPoint(const ContinuousPoint& other) : x(other.x), y(other.y), z(other.z) { }

ContinuousPoint& ContinuousPoint::operator=(const ContinuousPoint& other) {
    if (this == &other) return *this;

    x = other.x;
    y = other.y;
    z = other.z;

    return *this;
}

float& ContinuousPoint::operator[](size_t index) {
    if (index >= 3) 
        throw std::out_of_range("Index out of range. Valid indices are 0, 1, and 2\n");

    return (&x)[index];
}

float ContinuousPoint::lengthSquared() {
    return x*x + y*y + z*z;
}



DiscretePoint::DiscretePoint(int x, int y, int z) : x(x), y(y), z(z) { }

DiscretePoint::DiscretePoint(int x) : x(x), y(x), z(x) { }

DiscretePoint DiscretePoint::operator+(const DiscretePoint& other) const {
    return DiscretePoint(x + other.x, y + other.y, z + other.z);
}

DiscretePoint DiscretePoint::operator-(const DiscretePoint& other) const {
    return DiscretePoint(x - other.x, y - other.y, z - other.z);
}

DiscretePoint::DiscretePoint(const DiscretePoint& other) : x(other.x), y(other.y), z(other.z) { }

DiscretePoint& DiscretePoint::operator=(const DiscretePoint& other) {
    if (this == &other) return *this;

    x = other.x;
    y = other.y;
    z = other.z;

    return *this;
}

int& DiscretePoint::operator[](size_t index) {
    if (index >= 3) 
        throw std::out_of_range("Index out of range. Valid indices are 0, 1, and 2\n");

    return (&x)[index];
}

int DiscretePoint::lengthSquared() {
    return x*x + y*y + z*z;
}



DiscreteCluster::DiscreteCluster(float gridStepSizeSquared, int numberOfPoints) : gridStepSizeSquared(gridStepSizeSquared), numberOfPoints(numberOfPoints) { 
    data = new DiscretePoint[numberOfPoints];
}

DiscreteCluster::DiscreteCluster(const DiscreteCluster& other) : gridStepSizeSquared(other.gridStepSizeSquared), numberOfPoints(other.numberOfPoints), data(new DiscretePoint[other.numberOfPoints]) {
    for (int i = 0; i < numberOfPoints; i++)
    {
        data[i] = other.data[i];
    }
}

DiscreteCluster& DiscreteCluster::operator=(const DiscreteCluster& other) {
    if (this == &other) return *this;

    delete[] data;

    gridStepSizeSquared = other.gridStepSizeSquared;
    numberOfPoints = other.numberOfPoints;
    data = new DiscretePoint[numberOfPoints];
    for (int i = 0; i < numberOfPoints; i++)
    {
        data[i] = other.data[i];
    }
    
    return *this;
}

DiscreteCluster::DiscreteCluster(DiscreteCluster&& other) noexcept : gridStepSizeSquared(other.gridStepSizeSquared), numberOfPoints(other.numberOfPoints), data(other.data) {
    other.data = nullptr;
}

DiscreteCluster& DiscreteCluster::operator=(DiscreteCluster&& other) noexcept {
    if (this == &other) return *this;

    delete[] data;

    gridStepSizeSquared = other.gridStepSizeSquared;
    numberOfPoints = other.numberOfPoints;
    data = other.data;
    other.data = nullptr;
    other.numberOfPoints = 0;

    return *this;
}

DiscreteCluster::~DiscreteCluster() {
    delete[] data;
    data = nullptr;
}

void DiscreteCluster::setPoint(int atomIndex, int x, int y, int z) {
    data[atomIndex].x = x;
    data[atomIndex].y = y;
    data[atomIndex].z = z;
}

void DiscreteCluster::setPoint(int atomIndex, DiscretePoint& point) {
    data[atomIndex].x = point.x;
    data[atomIndex].y = point.y;
    data[atomIndex].z = point.z;
}

DiscretePoint& DiscreteCluster::getPoint(int atomIndex) {
    return data[atomIndex];
}

int DiscreteCluster::getDistanceSquared(int atomIndex1, int atomIndex2) {
    DiscretePoint delta = data[atomIndex1] - data[atomIndex2];
    return delta.lengthSquared();
}

float DiscreteCluster::getAtomEnergy(int atomIndex) {
    int squaredDistance;
    float result = 0.f;

    for (size_t j = 0; j < numberOfPoints; j++)
    {   
        if(atomIndex != j) {
            squaredDistance = getDistanceSquared(atomIndex, j);
            result += LeonardJonesSquaredPotential(squaredDistance * gridStepSizeSquared);
        }
    }
    return result;
}

float DiscreteCluster::getAtomEnergy(int atomIndex, std::vector<int> atomsToConsider) {
    int squaredDistance;
    float result = 0.f;

    for (size_t j = 0; j < atomsToConsider.size(); j++)
    {
        if(atomIndex != atomsToConsider[j]) {
            squaredDistance = getDistanceSquared(atomIndex, atomsToConsider[j]);
            result += LeonardJonesSquaredPotential(squaredDistance * gridStepSizeSquared);
        }
    }
    return result;
}

float DiscreteCluster::getClusterEnergy() {
    float result = 0.f;

    for (int i = 0; i < numberOfPoints; i++)
        result += getAtomEnergy(i);

    return result * 0.5f;    
}

std::vector<int> DiscreteCluster::getAtomNeighbours(int atomIndex, int cutoffDistanceSquared) {
    std::vector<int> neighbours;

    for (size_t i = 0; i < numberOfPoints; i++) {
    if ((float)getDistanceSquared(atomIndex, i) < cutoffDistanceSquared)
        neighbours.emplace_back(i);
    }

    return neighbours;
}

void DiscreteCluster::copyInto(DiscreteCluster& otherCluster) {
    if(otherCluster.numberOfPoints != numberOfPoints)
        otherCluster = DiscreteCluster(numberOfPoints);

    std::memcpy(otherCluster.data, data, sizeof(DiscretePoint) * numberOfPoints);
}

void DiscreteCluster::writeClusterToFile(const std::string& filename) {
    std::ofstream outFile(filename);

    // Header line
    outFile << "Interactive 3D Scatter Plot;\n";
    outFile << "::X::Y::Z;\n";

    // Write points
    for (int i = 0; i < numberOfPoints; i++) 
    {
        outFile << "Point A " << i << "::"
                << std::fixed << std::setprecision(2) 
                << data[i].x/5.f << "::"
                << data[i].y/5.f << "::" 
                << data[i].z/5.f << "::"
                << std::clamp(((int)getAtomEnergy(i) - 1) * 50 + 100, 0, 255) << "::" // Example value for the 5th field
                << "40::A::1::0::0::1::0;\n"; // Static additional fields
    }

    outFile.close();
}



ContinuousCluster::ContinuousCluster(int numberOfPoints) : numberOfPoints(numberOfPoints) { 
    data = new ContinuousPoint[numberOfPoints];
}

ContinuousCluster::ContinuousCluster(DiscreteCluster& discreteCluster, float gridStepSize) : numberOfPoints(discreteCluster.numberOfPoints) {
    data = new ContinuousPoint[numberOfPoints];

    for (size_t i = 0; i < numberOfPoints; i++)
    {
        data[i] = ContinuousPoint{discreteCluster.getPoint(i), gridStepSize};
    }
}

ContinuousCluster::ContinuousCluster(const ContinuousCluster& other) : numberOfPoints(other.numberOfPoints), data(new ContinuousPoint[other.numberOfPoints]) {
    for (int i = 0; i < numberOfPoints; i++)
    {
        data[i] = other.data[i];
    }
}

ContinuousCluster& ContinuousCluster::operator=(const ContinuousCluster& other) {
    if (this == &other) return *this;

    delete[] data;

    numberOfPoints = other.numberOfPoints;
    data = new ContinuousPoint[numberOfPoints];
    for (int i = 0; i < numberOfPoints; i++)
    {
        data[i] = other.data[i];
    }
    
    return *this;
}

ContinuousCluster::ContinuousCluster(ContinuousCluster&& other) noexcept : numberOfPoints(other.numberOfPoints), data(other.data) {
    other.data = nullptr;
    other.numberOfPoints = 0;
}

ContinuousCluster& ContinuousCluster::operator=(ContinuousCluster&& other) noexcept {
    if (this == &other) return *this;

    delete[] data;

    numberOfPoints = other.numberOfPoints;
    data = other.data;
    other.data = nullptr;
    other.numberOfPoints = 0;

    return *this;
}

ContinuousCluster::~ContinuousCluster() {
    delete[] data;
    data = nullptr;
}

void ContinuousCluster::setPoint(int atomIndex, float x, float y, float z) {
    data[atomIndex].x = x;
    data[atomIndex].y = y;
    data[atomIndex].z = z;
}

void ContinuousCluster::setPoint(int atomIndex, ContinuousPoint& point) {
    data[atomIndex].x = point.x;
    data[atomIndex].y = point.y;
    data[atomIndex].z = point.z;
}

void ContinuousCluster::addToPoints(ContinuousPoint* points, float factor) {
    for (int i = 0; i < numberOfPoints; i++) {
        data[i] = data[i] + points[i] * factor;
    }
}

ContinuousPoint& ContinuousCluster::getPoint(int atomIndex) {
    return data[atomIndex];
}

float ContinuousCluster::getDistanceSquared(int atomIndex1, int atomIndex2) {
    ContinuousPoint delta = data[atomIndex1] - data[atomIndex2];
    return delta.lengthSquared();
}

float ContinuousCluster::getAtomEnergy(std::function<float(float)> potentialSquared, int atomIndex) {
    float squaredDistance;
    float result = 0.f;

    for (size_t j = 0; j < numberOfPoints; j++)
    {   
        if(atomIndex != j) {
            squaredDistance = getDistanceSquared(atomIndex, j);
            result += potentialSquared(squaredDistance);
        }
    }
    return result;
}

float ContinuousCluster::getAtomEnergy(std::function<float(float)> potentialSquared, int atomIndex, std::vector<int> atomsToConsider) {
    float squaredDistance;
    float result = 0.f;

    for (size_t j = 0; j < atomsToConsider.size(); j++)
    {
        if(atomIndex != atomsToConsider[j]) {
            squaredDistance = getDistanceSquared(atomIndex, atomsToConsider[j]);
            result += potentialSquared(squaredDistance);
        }
    }
    return result;
}

float ContinuousCluster::getClusterEnergy(std::function<float(float)> potentialSquared) {
    float result = 0.f;

    for (int i = 0; i < numberOfPoints; i++)
        result += getAtomEnergy(potentialSquared, i);

    return result * 0.5f;    
}

void ContinuousCluster::copyInto(ContinuousCluster& otherCluster) {
    if(otherCluster.numberOfPoints != numberOfPoints)
        otherCluster = ContinuousCluster(numberOfPoints);

    std::memcpy(otherCluster.data, data, sizeof(ContinuousCluster) * numberOfPoints);
}

void ContinuousCluster::writeClusterToFile(const std::string& filename) {
    std::ofstream outFile(filename);
    // Write points
    for (size_t i = 0; i < numberOfPoints; i++) 
    {
        outFile << std::fixed << std::setprecision(8) 
                << data[i].x << "::"
                << data[i].y << "::" 
                << data[i].z << "\n";

    }
    outFile.close();
}
