#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

static inline float horizontalSumAVX(__m256 x) {
    // ( x3+x7, x2+x6, x1+x5, x0+x4 )
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    // ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 )
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    // ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 )
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    // Conversion to float is a no-op on x86-64
    return _mm_cvtss_f32(x32);
}

struct RealLJCalculator {
private:
    const __m256 TWO = _mm256_set1_ps(2.0f);
    const __m256 EPS = _mm256_set1_ps(1e-10f);
    static constexpr float EPSF = 1e-10f;

public:
    inline float potential(float r2) const {
        if (r2 < 1e-10f) return std::numeric_limits<float>::infinity();

        const float inv_r2 = 1.0f / r2;
        const float inv_r6 = inv_r2 * inv_r2 * inv_r2;
        const float inv_r12 = inv_r6 * inv_r6;

        return inv_r12 - 2.0f * inv_r6;
    }

    inline __m256 potentialAVX(__m256 r2) const {
        // add small epsilon to all values to avoid division by zero
        __m256 r2_safe = _mm256_add_ps(r2, EPS);
        
        // approximate 1/r2 by rcp + a Newton-Raphson refinement step
        __m256 inv_r2 = _mm256_rcp_ps(r2_safe);
        inv_r2 = _mm256_mul_ps(inv_r2, _mm256_fnmadd_ps(r2_safe, inv_r2, TWO));
        
        __m256 inv_r4 = _mm256_mul_ps(inv_r2, inv_r2);
        __m256 inv_r6 = _mm256_mul_ps(inv_r4, inv_r2);
        __m256 inv_r12 = _mm256_mul_ps(inv_r6, inv_r6);
        
        return _mm256_fnmadd_ps(inv_r6, TWO, inv_r12);
    }

    inline float force(float r2) const {
        // this effectively calculates the derivative of the potential, divided by sqrt(r2). This is a usefull optimization when calculating the gradient
        if (r2 < EPSF) return std::numeric_limits<float>::infinity();

        const float inv_r2 = 1.0f / r2;
        const float inv_r6 = inv_r2 * inv_r2 * inv_r2;
        const float inv_r8 = inv_r6 * inv_r2;
        const float inv_r14 = inv_r8 * inv_r6;
        return 12.f * (inv_r8 - inv_r14);
    }
};



struct DiscreteCluster {
public:
    alignas(64) std::vector<int16_t> points; // the points are stored like: [x1, y1, z1, 0, x2, y2, z2, 0, ...]
    size_t n;
private:
    size_t nPadded;
    __m512i squaredCutoffSIMD;

public:
    DiscreteCluster();
    explicit DiscreteCluster(const size_t numberOfPoints, const int32_t cutoffSIMD);  

    float getAtomEnergyAVX(uint64_t atomIndex, const std::pair<std::vector<uint64_t>, uint64_t>& neighbours, const std::vector<float>& lookup) const;
    float getAtomEnergyAVX(uint64_t atomIndex, const std::vector<float>& lookup) const;
    float getClusterEnergy(float gridSpacingSquared) const;

    std::pair<std::vector<uint64_t>, uint64_t> getNeighbours(uint64_t atomIndex, uint32_t squaredCutoff) const;
    bool doesPointOverlap(uint64_t atomIndex, uint64_t maxIncludedIndex) const;

    void copyTo(DiscreteCluster& otherCluster) const;
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
    explicit Cluster(const DiscreteCluster& discreteCluster, float gridSpacing);

    Cluster(const Cluster& other) = default;
    Cluster& operator=(const Cluster& other) = default;

    void setPoint(const size_t atomIndex, const float xVal, const float yVal, const float zVal);

    float getDistanceSquared(size_t atomIndex1, size_t atomIndex2) const;

    float getAtomEnergyAVX(size_t atomIndex, const RealLJCalculator& lj) const;
    float getClusterEnergyAVX(const RealLJCalculator& lj) const;
    
    void getClusterGradient(std::vector<float>& gradX, std::vector<float>& gradY, std::vector<float>& gradZ, const RealLJCalculator& lj) const;

    void copyTo(Cluster& otherCluster) const;

    size_t size() const { return n; };
};

#endif // DATA_STRUCTURES_H