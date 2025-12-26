#include "dataStructures.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>


// LJCalculator Implementation
// ===================================================================================
inline float LJCalculator::potential(float r2) const {
    if (r2 < 1e-10f) return std::numeric_limits<float>::infinity();

    const float inv_r2 = 1.0f / r2;
    const float inv_r6 = inv_r2 * inv_r2 * inv_r2;
    const float inv_r12 = inv_r6 * inv_r6;

    return inv_r12 - 2.0f * inv_r6;
}

inline __m256 LJCalculator::potentialAVX(__m256 r2) const {
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



// Cluster Implementation
// ===================================================================================
// make the vectors with lengts of a multiple of 8, such that SIMD instructions can be optimally used in getAtomEnergyAVX
Cluster::Cluster(const size_t numberOfPoints) : x((numberOfPoints + 7) & ~size_t(7)), y((numberOfPoints + 7) & ~size_t(7)), z((numberOfPoints + 7) & ~size_t(7)), n(numberOfPoints), nPadded((numberOfPoints + 7) & ~size_t(7)) {}

void Cluster::setPoint(const size_t atomIndex, const float xVal, const float yVal, const float zVal) {
    x[atomIndex] = xVal;
    y[atomIndex] = yVal;
    z[atomIndex] = zVal;
}

float Cluster::getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const {
    const float dx = x[atomIndex1] - x[atomIndex2];
    const float dy = y[atomIndex1] - y[atomIndex2];
    const float dz = z[atomIndex1] - z[atomIndex2];
    return dx*dx + dy*dy + dz*dz;
}

float Cluster::getAtomEnergyAVX(size_t atomIndex, const LJCalculator& lj) const {
    alignas(32) float energiesArr[8];

    __m256 totalVec = _mm256_setzero_ps();

    __m256 xi = _mm256_set1_ps(x[atomIndex]);
    __m256 yi = _mm256_set1_ps(y[atomIndex]);
    __m256 zi = _mm256_set1_ps(z[atomIndex]);

    for (size_t j = 0; j < nPadded; j += 8) {
        // load 8 points
        __m256 xj = _mm256_loadu_ps(&x[j]);
        __m256 yj = _mm256_loadu_ps(&y[j]);
        __m256 zj = _mm256_loadu_ps(&z[j]);

        // compute squared distances
        __m256 dx = _mm256_sub_ps(xi, xj);
        __m256 dy = _mm256_sub_ps(yi, yj);
        __m256 dz = _mm256_sub_ps(zi, zj);

        dx = _mm256_mul_ps(dx, dx);
        dy = _mm256_mul_ps(dy, dy);
        dz = _mm256_mul_ps(dz, dz);

        __m256 r2 = _mm256_add_ps(dx, _mm256_add_ps(dy, dz));
        __m256 energies = lj.potentialAVX(r2);

        // if atomIndex is in this block, set its energy to 0
        if (atomIndex >= j && atomIndex < j + 8) {
            _mm256_store_ps(energiesArr, energies); 
            energiesArr[atomIndex - j] = 0.0f;
            energies = _mm256_load_ps(energiesArr);
        }
        // if this block was partially in the invalid padded part, negate the contribution of the padded entries
        if (j + 8 > n) {
            _mm256_store_ps(energiesArr, energies); 
            for (size_t k = n; k < j + 8; k++)
                energiesArr[k - j] = 0.0f;
            energies = _mm256_load_ps(energiesArr);
        }

        totalVec = _mm256_add_ps(totalVec, energies);
    }
    return horizontalSumAVX(totalVec);
};

float Cluster::getClusterEnergyAVX(const LJCalculator& lj) const {
    float total = 0.f;

    for (size_t i = 0; i < n; i++)
        total += getAtomEnergyAVX(i, lj);

    return total * 0.5f;
}

void Cluster::copyTo(Cluster& otherCluster) const {
    otherCluster.x = x;
    otherCluster.y = y;
    otherCluster.z = z;
    otherCluster.n = n;
    otherCluster.nPadded = nPadded;
}

float Cluster::horizontalSumAVX(__m256 vals) {
    // sum the last half on the first half
    __m128 vLow = _mm256_castps256_ps128(vals);
    __m128 vHigh = _mm256_extractf128_ps(vals, 1);
    vLow = _mm_add_ps(vLow, vHigh);

    // vLow: [x0, x1, x2, x3]
    // shuf: [x1, x1, x3, x3]
    // sums: [x0 + x1, 2 x1, x2 + x3, 2 x3]
    __m128 shuf = _mm_movehdup_ps(vLow);
    __m128 sums = _mm_add_ps(vLow, shuf);

    // shuf: [ x2 + x3, 2 x3, x3, x3]
    shuf = _mm_movehl_ps(shuf, sums);
    // sums: [x0 + x1 + x2 + x3, ...]
    sums = _mm_add_ss(sums, shuf);

    // return the first value of sums
    return _mm_cvtss_f32(sums);
}
