#include "dataStructures.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>


float lennardJonesPotential(float distance) {
    if(distance < 1e-7f)
        return 0.0f;

    // using 'reduced' units, the LJ potential is simply:
    return std::pow(distance, -12) - 2.f * std::pow(distance, -6);
}

float lennardJonesSquaredPotential(float squaredDistance) {
    if(squaredDistance < 1e-14f)
        return 0.0f;

    // defines the LJ potential based on a squared distance input. It saves some computation
    return std::pow(squaredDistance, -6) - 2.f * std::pow(squaredDistance, -3);
}

float lennardJonesDerivative(float distance) {
    return -12.f * (std::pow(distance, -13) - std::pow(distance, -7));
}



// LJCalculator Implementation
// ===================================================================================
LJCalculator::LJCalculator() {
    buildTables();
}

void LJCalculator::buildTables() {
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        float r2 = i * (1.0f / INV_STEP);

        // special case to ensure that if the LJ potential between an atom and itself is 0
        if (i <= 1) {
            ljPotentialTable[i] = 0.0f;
        } else {
            float r6_inv = 1.0f / (r2 * r2 * r2);
            float r12_inv = r6_inv * r6_inv;

            ljPotentialTable[i] = r12_inv - 2.0f * r6_inv;
        }
    }
}

inline float LJCalculator::potential(float r2) const {
    size_t index0 = static_cast<size_t>(r2 * INV_STEP);
    index0 = std::min(index0, TABLE_SIZE - 2);

    float val0 = ljPotentialTable[index0];
    float val1 = ljPotentialTable[index0 + 1];
    float t = (r2 * INV_STEP) - (float)index0;

    return val0 + t * (val1 - val0);
}

inline __m256 LJCalculator::potentialAVX(__m256 r2) const {
    __m256 pos = _mm256_mul_ps(r2, _mm256_set1_ps(INV_STEP));

    __m256i index0 = _mm256_cvttps_epi32(pos);
    index0 = _mm256_min_epi32(index0, _mm256_set1_epi32(TABLE_SIZE - 2));
    __m256i index1 = _mm256_add_epi32(index0, _mm256_set1_epi32(1));

    __m256 val0 = _mm256_i32gather_ps(ljPotentialTable.data(), index0, 4);
    __m256 val1 = _mm256_i32gather_ps(ljPotentialTable.data(), index1, 4);

    __m256 t = _mm256_sub_ps(pos, _mm256_cvtepi32_ps(index0));

    return _mm256_fmadd_ps(t, _mm256_sub_ps(val1, val0), val0); // linear interpolation: t*(val1 - val0) + val0
}




// Cluster Implementation
// ===================================================================================
Cluster::Cluster(const size_t numberOfPoints) : x(numberOfPoints), y(numberOfPoints), z(numberOfPoints), n(numberOfPoints) {}

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
    __m256 totalVec = _mm256_setzero_ps();

    __m256 xi = _mm256_set1_ps(x[atomIndex]);
    __m256 yi = _mm256_set1_ps(y[atomIndex]);
    __m256 zi = _mm256_set1_ps(z[atomIndex]);

    for (size_t j = 0; j + 8 <= n; j += 8) {
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

        // LJ SIMD table lookup
        __m256 energies = lj.potentialAVX(r2);

        totalVec = _mm256_add_ps(totalVec, energies);
    }
    float total = horizontalSumAVX(totalVec);

    // remainder
    for (size_t j = n - (n % 8); j < n; j++) {
        float dx = x[atomIndex] - x[j];
        float dy = y[atomIndex] - y[j];
        float dz = z[atomIndex] - z[j];

        float r2 = dx*dx + dy*dy + dz*dz;
        // LJ table lookup
        total += lj.potential(r2);
    }

    return total;
};

float Cluster::getAtomEnergy(size_t atomIndex) const {
    float total = 0.f;

    for (size_t j = 0; j < n; j++)
    {   
        if (atomIndex != j) {
            const float squaredDistance = getDistanceSquared(atomIndex, j);
            total += lennardJonesSquaredPotential(squaredDistance);
        }
    }
    return total;
}

float Cluster::getAtomEnergy(size_t atomIndex, const std::vector<size_t>& atomsToConsider) const {
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

    for (size_t i = 0; i < n; i++)
        total += getAtomEnergy(i);

    return total * 0.5f;    
}

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
