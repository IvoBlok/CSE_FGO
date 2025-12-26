#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <array>
#include <cstddef>
#include <immintrin.h>

float lennardJonesPotential(float distance);
float lennardJonesSquaredPotential(float squaredDistance);
float lennardJonesDerivative(float distance);


// this class uses basic interpolation to limit the amount of LJ potential computations we do
// this version uses an equally spaced grid in (0, 25) in r^2 'space'. This doesn't create an ideal spacing with many points outside area's of high detail
// TODO improve the interpolation choice / grid choice
class LJCalculator {
private:
    static constexpr size_t TABLE_SIZE = 1024;
    static constexpr float MAX_R2 = 25.0f;
    static constexpr float INV_STEP = (TABLE_SIZE - 1) / MAX_R2;
    
    alignas(64) std::array<float, TABLE_SIZE> ljPotentialTable;

public:
    LJCalculator();
    void buildTables();

    inline float potential(float r2) const;
    __m256 potentialAVX(__m256 r2) const;
};


struct Cluster {
public:
    alignas(64) std::vector<float> x, y, z;
    size_t n = 0;
    
    Cluster() = default;
    explicit Cluster(const size_t numberOfPoints);

    Cluster(const Cluster& other) = default;
    Cluster& operator=(const Cluster& other) = default;

    void setPoint(const size_t atomIndex, const float xVal, const float yVal, const float zVal);

    float getDistanceSquared(size_t atomIndex1, size_t atomIndex2) const;

    float getAtomEnergyAVX(size_t atomIndex, const LJCalculator& lj) const;
    float getAtomEnergy(size_t atomIndex) const;
    float getAtomEnergy(size_t atomIndex, const std::vector<size_t>& atomsToConsider) const;

    float getClusterEnergy() const;
    float getClusterEnergyAVX(const LJCalculator& lj) const;
    
    void copyTo(Cluster& otherCluster) const;

    size_t size() const { return n; };

private:
    static float horizontalSumAVX(__m256 vals);
};

#endif // DATA_STRUCTURES_H