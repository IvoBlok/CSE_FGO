#include "dataStructures.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>

// DiscreteCluster Implementation
// ===================================================================================
float DiscreteCluster::getAtomEnergyAVX(uint64_t atomIndex, const std::vector<uint64_t> neighbours, const std::vector<float> lookup) {
    // points, neighbours is required to have a multiple of 8 elements

    __m256 energyTotal = _mm256_setzero_ps();
    __m512i atom = _mm512_set1_epi64(points[atomIndex]); // broadcast main atom

    for (size_t i = 0; i < neighbours.size(); i+=8)
    {
        __m512i indices = _mm512_load_epi64(&neighbours[i]);
        __m512i neighbours = _mm512_i64gather_epi64(indices, points.data(), 8);

        __m512i diff = _mm512_sub_epi16(atom, neighbours);
        __m512i squaredXY = _mm512_madd_epi16(diff, diff);

        __m512i shiftedLeft = _mm512_alignr_epi32(squaredXY, squaredXY, 1);
        __m512i squaredDistances = _mm512_add_epi32(squaredXY, shiftedLeft);

        squaredDistances = _mm512_maskz_mov_epi32(0xAAAA, squaredDistances); // set every second element to zero
        squaredDistances = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(squaredDistances)); //TODO this should be avoidable, if we make slight changes above; This is only needed because we first have 32 bits of usefull stuff, then 32 bit of zeros. Instead we need it switched around
        
        __m256 energies = _mm512_i64gather_ps(squaredDistances, lookup.data(), 4);
        energyTotal = _mm256_add_ps(energyTotal, energies);
    }

    return horizontalSumAVX(energyTotal);
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

float Cluster::getAtomEnergyAVX(size_t atomIndex, const RealLJCalculator& lj) const {
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

float Cluster::getClusterEnergyAVX(const RealLJCalculator& lj) const {
    float total = 0.f;

    for (size_t i = 0; i < n; i++)
        total += getAtomEnergyAVX(i, lj);

    return total * 0.5f;
}

void Cluster::getClusterGradient(std::vector<float>& gradX, std::vector<float>& gradY, std::vector<float>& gradZ, const RealLJCalculator& lj) const {
    if (gradX.size() < n || gradY.size() < n || gradZ.size() < n)
        throw std::invalid_argument("given gradient output vectors are of invalid size!");
    
    std::fill(gradX.begin(), gradX.end(), 0.0f);
    std::fill(gradY.begin(), gradY.end(), 0.0f);
    std::fill(gradZ.begin(), gradZ.end(), 0.0f);

    for (size_t i = 0; i < n; i++)
    {
        const float xi = x[i];
        const float yi = y[i];
        const float zi = z[i];

        for (size_t j = i + 1; j < n; j++)
        {
            const float dx = xi - x[j];
            const float dy = yi - y[j];
            const float dz = zi - z[j];

            const float r2 = dx*dx + dy*dy + dz*dz;
            float force = lj.force(r2);
            
            if (force > 1e10f) force = 1e10f;
            if (force < -1e10f) force = -1e10f;

            gradX[i] += dx * force;
            gradY[i] += dy * force;
            gradZ[i] += dz * force;

            gradX[j] -= dx * force;
            gradY[j] -= dy * force;
            gradZ[j] -= dz * force;
        }
    }
}


void Cluster::copyTo(Cluster& otherCluster) const {
    otherCluster.x = x;
    otherCluster.y = y;
    otherCluster.z = z;
    otherCluster.n = n;
    otherCluster.nPadded = nPadded;
}