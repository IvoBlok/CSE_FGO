#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <immintrin.h>

inline float lennardJonesDerivative(float distance) {
    return -12.f * (std::pow(distance, -13) - std::pow(distance, -7));
}


// this class uses basic interpolation to limit the amount of LJ potential computations we do
// this version uses an equally spaced grid in (0, 25) in r^2 'space'. This doesn't create an ideal spacing with many points outside area's of high detail
// TODO improve the interpolation choice / grid choice
struct LJCalculator {
private:
    const __m256 TWO = _mm256_set1_ps(2.0f);
    const __m256 EPS = _mm256_set1_ps(1e-10f);
public:
    inline float potential(float r2) const;
    inline __m256 potentialAVX(__m256 r2) const;
};


struct Cluster {
public:
    alignas(64) std::vector<float> x, y, z;
    size_t n;
private:
    size_t nPadded;

public:
    Cluster() = default;
    explicit Cluster(const size_t numberOfPoints);

    Cluster(const Cluster& other) = default;
    Cluster& operator=(const Cluster& other) = default;

    void setPoint(const size_t atomIndex, const float xVal, const float yVal, const float zVal);

    float getDistanceSquared(size_t atomIndex1, size_t atomIndex2) const;

    float getAtomEnergyAVX(size_t atomIndex, const LJCalculator& lj) const;
    float getClusterEnergyAVX(const LJCalculator& lj) const;
    
    void copyTo(Cluster& otherCluster) const;

    size_t size() const { return n; };

private:
    static float horizontalSumAVX(__m256 vals);
};

#endif // DATA_STRUCTURES_H