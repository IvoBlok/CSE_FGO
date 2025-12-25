#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <cstddef>

float lennardJonesPotential(float distance);
float lennardJonesSquaredPotential(float squaredDistance);
float lennardJonesDerivative(float distance);


struct Point {
    float x, y, z;

    Point() = default;
    Point(const float x, const float y, const float z);
    explicit Point(const float val);

    Point operator+(const Point& other) const;
    Point operator-(const Point& other) const;
    Point operator*(float scalar) const;
    Point operator/(float scalar) const;

    float& operator[](const size_t index);
    const float& operator[](const size_t index) const;

    float lengthSquared() const;
};


struct Cluster {
    std::vector<Point> points;
    size_t n = 0;
    
    Cluster() = default;
    explicit Cluster(const size_t numberOfPoints);

    Cluster(const Cluster& other) = default;
    Cluster& operator=(const Cluster& other) = default;

    void setPoint(const size_t atomIndex, const float x, const float y, const float z);
    void setPoint(const size_t atomIndex, const Point& point);

    Point& getPoint(const size_t atomIndex);
    const Point& getPoint(const size_t atomIndex) const;

    void addToPoints(const std::vector<Point>& deltas, const float factor);

    float getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const;

    float getAtomEnergy(const size_t atomIndex) const;
    float getAtomEnergy(const size_t atomIndex, const std::vector<size_t>& atomsToConsider) const;

    float getClusterEnergy() const;
    
    void copyTo(Cluster& otherCluster) const;

    size_t size() const { return n; };
};

#endif // DATA_STRUCTURES_H