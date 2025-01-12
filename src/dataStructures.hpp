#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <iostream>
#include <cmath>
#include <random>
#include <set>
#include <vector>
#include <string>
#include <functional>

struct DiscretePoint {
    int x;
    int y;
    int z;

    DiscretePoint(int x, int y, int z);
    DiscretePoint(int x = 0);

    DiscretePoint operator+(const DiscretePoint& other) const;
    DiscretePoint operator-(const DiscretePoint& other) const;

    DiscretePoint(const DiscretePoint& other);
    DiscretePoint& operator=(const DiscretePoint& other);

    int& operator[](size_t index);

    int lengthSquared();
};


struct DiscreteCluster {
    DiscretePoint* data;
    int numberOfPoints;

    DiscreteCluster(int numberOfPoints = 0);
    
    ~DiscreteCluster();

    DiscreteCluster(const DiscreteCluster& other);
    DiscreteCluster& operator=(const DiscreteCluster& other);

    DiscreteCluster(DiscreteCluster&& other) noexcept;
    DiscreteCluster& operator=(DiscreteCluster&& other) noexcept;

    void setPoint(int atomIndex, int x, int y, int z);
    void setPoint(int atomIndex, DiscretePoint& point);

    DiscretePoint& getPoint(int atomIndex);

    int getDistanceSquared(int atomIndex1, int atomIndex2);

    float getAtomEnergy(float* LJLookup, int atomIndex);
    float getAtomEnergy(float* LJLookup, int atomIndex, std::vector<int> atomsToConsider);

    float getClusterEnergy(float* LJLookup);

    std::vector<int> getAtomNeighbours(int atomIndex, int cutoffDistanceSquared);

    void copyInto(DiscreteCluster& otherCluster);

    void writeClusterToFile(float* LJLookup, const std::string& filename);
};


struct ContinuousPoint {
    float x;
    float y;
    float z;

    ContinuousPoint(float x, float y, float z);
    ContinuousPoint(float x = 0.f);
    ContinuousPoint(DiscretePoint& discretePoint, float gridStepSize);

    ContinuousPoint operator+(const ContinuousPoint& other) const;
    ContinuousPoint operator-(const ContinuousPoint& other) const;

    ContinuousPoint operator*(float scalar) const;
    ContinuousPoint operator*(double scalar) const;

    ContinuousPoint(const ContinuousPoint& other);
    ContinuousPoint& operator=(const ContinuousPoint& other);

    float& operator[](size_t index);

    float lengthSquared();
};


struct ContinuousCluster {
    ContinuousPoint* data;
    int numberOfPoints;

    
    ContinuousCluster(int numberOfPoints = 0);
    ContinuousCluster(DiscreteCluster& discreteCluster, float gridStepSize);

    ~ContinuousCluster();

    ContinuousCluster(const ContinuousCluster& other);
    ContinuousCluster& operator=(const ContinuousCluster& other);

    ContinuousCluster(ContinuousCluster&& other) noexcept;
    ContinuousCluster& operator=(ContinuousCluster&& other) noexcept;

    void setPoint(int atomIndex, float x, float y, float z);
    void setPoint(int atomIndex, ContinuousPoint& point);

    void addToPoints(ContinuousPoint* points, float factor);

    ContinuousPoint& getPoint(int atomIndex);

    float getDistanceSquared(int atomIndex1, int atomIndex2);

    float getAtomEnergy(std::function<float(float)> potentialSquared, int atomIndex);
    float getAtomEnergy(std::function<float(float)> potentialSquared, int atomIndex, std::vector<int> atomsToConsider);

    float getClusterEnergy(std::function<float(float)> potentialSquared);
    
    void copyInto(ContinuousCluster& otherCluster);

    void writeClusterToFile(const std::string& filename);
};


#endif // DATA_STRUCTURES_H