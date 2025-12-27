#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <immintrin.h>

// this class uses basic interpolation to limit the amount of LJ potential computations we do
// this version uses an equally spaced grid in (0, 25) in r^2 'space'. This doesn't create an ideal spacing with many points outside area's of high detail
// TODO improve the interpolation choice / grid choice
struct LJCalculator {
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
    
    void getClusterGradient(std::vector<float>& gradX, std::vector<float>& gradY, std::vector<float>& gradZ, const LJCalculator& lj) const;

    void copyTo(Cluster& otherCluster) const;

    size_t size() const { return n; };

private:
    static float horizontalSumAVX(__m256 vals);
};

#endif // DATA_STRUCTURES_H