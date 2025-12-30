#include "dataStructures.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>


// DiscreteCluster Implementation
// ===================================================================================
DiscreteCluster::DiscreteCluster() : n(0), nPadded(0), squaredCutoffSIMD(_mm512_setzero_si512()) {}

DiscreteCluster::DiscreteCluster(const size_t numberOfPoints, const int32_t cutoffSIMD)
     : points(4*((numberOfPoints + 7) & ~size_t(7))), 
       n(numberOfPoints), 
       nPadded((numberOfPoints + 7) & ~size_t(7)),
       squaredCutoffSIMD(_mm512_set1_epi32(cutoffSIMD * cutoffSIMD)) {}

float DiscreteCluster::getAtomEnergyAVX(uint64_t atomIndex, const std::pair<std::vector<uint64_t>, uint64_t>& neighbours, const std::vector<float>& lookup) const {
    // neighbours is required to have a multiple of 8 elements, where extra entries can be added by using the same index as that of the main atom.
    // numNeighbours is the number of atoms that represent actual neighbours; the length of neighbours without the padding.
    // TODO this might be improvable by loading 2 blocks of neighbours each iteration, combining them somehow at the squaredDistances step where currently only half of the elements are used (rest is 0). Then use a 32bit index gather instruction instead, such that we can gather twice the values in the one (expensive) gather instruction.

    __m256 energiesTotal = _mm256_setzero_ps();
    __m256 lastEnergies = _mm256_setzero_ps();
    __m512i atom = _mm512_set1_epi64(*reinterpret_cast<const int64_t*>(&points[4*atomIndex]));

    for (size_t i = 0; i < neighbours.second; i+=8)
    {
        energiesTotal = _mm256_add_ps(energiesTotal, lastEnergies);

        __m512i indices = _mm512_loadu_epi64(&neighbours.first[i]);
        __m512i neighbourPoints = _mm512_i64gather_epi64(indices, points.data(), 8);

        __m512i diff = _mm512_sub_epi16(atom, neighbourPoints);
        __m512i squaredXY = _mm512_madd_epi16(diff, diff);

        __m512i shiftedLeft = _mm512_alignr_epi32(squaredXY, squaredXY, 1);
        __m512i squaredDistances = _mm512_add_epi32(squaredXY, shiftedLeft);

        squaredDistances = _mm512_maskz_mov_epi32(0x5555, squaredDistances); // set every second element to zero, such that the next line can interpret it directly as 8 64bit (unsigned) ints
        
        // not all neighbours might still be within the cutoff range, and hence be in the range of the lookup array. So set any that go over the limit to 0, such that their energy contribution is 0.
        // this is due to us keeping the same neighbours during discreteOptimization, while these neighbours do get moved, potentially out of the cutoff range they were selected with at the start.
        __mmask16 outOfRangeMask = _mm512_cmpgt_epi32_mask(squaredCutoffSIMD, squaredDistances); // bit is 0 if the distance element is larger , 1 if smaller or equal
        squaredDistances = _mm512_maskz_mov_epi32(outOfRangeMask, squaredDistances);

        lastEnergies = _mm512_i64gather_ps(squaredDistances, lookup.data(), 4);
    }
    // remove energy contribution of the entries corresponding to padded neighbours
    size_t padding = neighbours.first.size() - neighbours.second;
    __mmask8 mask = (0xFF >> padding);
    lastEnergies = _mm256_maskz_mov_ps(mask, lastEnergies);

    energiesTotal = _mm256_add_ps(energiesTotal, lastEnergies);
    return horizontalSumAVX(energiesTotal);
}

float DiscreteCluster::getAtomEnergyAVX(uint64_t atomIndex, const std::vector<float>& lookup) const {
    __m256 energiesTotal = _mm256_setzero_ps();
    __m256 lastEnergies = _mm256_setzero_ps();
    __m512i atom = _mm512_set1_epi64(*reinterpret_cast<const int64_t*>(&points[4*atomIndex]));

    for (size_t i = 0; i < nPadded; i += 8)
    {
        // this ordering, of only adding the energies of the last loop to the total at the start of the next loop, allows us to avoid any branches in the main loop.
        // We can then, after the last iteration, when the block with irrelevant potential padded elements got done, remove those incorrect entries efficiently after the main loop.
        energiesTotal = _mm256_add_ps(energiesTotal, lastEnergies);

        __m512i pointData = _mm512_loadu_epi64(reinterpret_cast<const int64_t*>(&points[4*i]));

        __m512i diff = _mm512_sub_epi16(atom, pointData);
        __m512i squaredXY = _mm512_madd_epi16(diff, diff);
        __m512i shiftedLeft = _mm512_alignr_epi32(squaredXY, squaredXY, 1);
        __m512i squaredDistances = _mm512_add_epi32(squaredXY, shiftedLeft);
        squaredDistances = _mm512_maskz_mov_epi32(0x5555, squaredDistances);

        // only calculate energy for those within the cutoff range
        __mmask16 outOfRangeMask = _mm512_cmpgt_epi32_mask(squaredCutoffSIMD, squaredDistances);
        squaredDistances = _mm512_maskz_mov_epi32(outOfRangeMask, squaredDistances);

        lastEnergies = _mm512_i64gather_ps(squaredDistances, lookup.data(), 4);

        // remove the contribution between atomIndex and itself (+inf). 
        // [1, 1, 1, 1, 1, 1, 1, 1] if atomIndex is not in the range[i, i+8]. Otherwise the (7 - atomIndex % 8) element is 0. 
        __mmask8 selfMask = (atomIndex >= i && atomIndex < i + 8) ? (0xFF ^ (1 << (atomIndex % 8))) : 0xFF;
        lastEnergies = _mm256_maskz_mov_ps(selfMask, lastEnergies);
    }
    
    // remove energy contribution of the entries corresponding to padded points
    size_t padding = nPadded - n;
    __mmask8 mask = (0xFF >> padding);
    lastEnergies = _mm256_maskz_mov_ps(mask, lastEnergies);

    energiesTotal = _mm256_add_ps(energiesTotal, lastEnergies);
    return horizontalSumAVX(energiesTotal);
}

float DiscreteCluster::getClusterEnergy(float gridSpacingSquared) const {
    //TODO slow? basic (exact) implementation to get cluster energy. 
    // this again assumes no two distinct points in the cluster sit on the same point (woul)
    float totalEnergy = 0.0f;

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = i + 1; j < n; j++)
        {
            const int32_t dx = points[4*i] - points[4*j];
            const int32_t dy = points[4*i+1] - points[4*j+1];
            const int32_t dz = points[4*i+2] - points[4*j+2];

            const int32_t sum = dx*dx + dy*dy + dz*dz;
            if (sum == 0)
                return std::numeric_limits<float>::infinity();

            const float invr2 = 1.0f/(gridSpacingSquared * sum);

            const float invr6 = invr2 * invr2 * invr2;
            totalEnergy += invr6 * invr6 - 2 * invr6;
        }
    }
    
    return totalEnergy;
}

std::pair<std::vector<uint64_t>, uint64_t> DiscreteCluster::getNeighbours(uint64_t atomIndex, uint32_t squaredCutoff) const {
    std::vector<uint64_t> neighbours;

    const int16_t xi = points[4*atomIndex+0];
    const int16_t yi = points[4*atomIndex+1];
    const int16_t zi = points[4*atomIndex+2];

    for (uint64_t i = 0; i < n; i++)
    {
        if(i == atomIndex) continue; // skip self
        const int32_t dx = xi - points[4*i+0];
        const int32_t dy = yi - points[4*i+1];
        const int32_t dz = zi - points[4*i+2];

        const int32_t r2 = dx*dx + dy*dy + dz*dz;
        if(r2 < squaredCutoff)
            neighbours.emplace_back(i);
    }

    // ensure the vector has a length of 8, where the padding is filled with atomIndex, such that later getAtomEnergyAVX can safely load blocks of 8 neighbours at a time
    size_t trueNeighbourCount = neighbours.size();
    size_t remainder = trueNeighbourCount % 8;
    if (remainder != 0) {
        size_t padding = 8 - remainder;
        for (size_t i = 0; i < padding; i++)
            neighbours.emplace_back(atomIndex);
    }

    return {neighbours, trueNeighbourCount};
}

bool DiscreteCluster::doesPointOverlap(uint64_t atomIndex, uint64_t maxIncludedIndex) const {
    for (uint64_t i = 0; i <= maxIncludedIndex; i++)
    {
        if(points[4*atomIndex] - points[4*i] == 0 && points[4*atomIndex+1] - points[4*i+1] == 0 && points[4*atomIndex+2] - points[4*i+2] == 0)
            return true;
    }
    return false;
}

void DiscreteCluster::copyTo(DiscreteCluster& otherCluster) const {
    otherCluster.points = points;
    otherCluster.n = n;
    otherCluster.nPadded = nPadded;
    otherCluster.squaredCutoffSIMD = squaredCutoffSIMD;
}



// Cluster Implementation
// ===================================================================================
// make the vectors with lengts of a multiple of 8, such that SIMD instructions can be optimally used in getAtomEnergyAVX
Cluster::Cluster(const size_t numberOfPoints) : x((numberOfPoints + 7) & ~size_t(7)), y((numberOfPoints + 7) & ~size_t(7)), z((numberOfPoints + 7) & ~size_t(7)), n(numberOfPoints), nPadded((numberOfPoints + 7) & ~size_t(7)) {}

Cluster::Cluster(const DiscreteCluster& discreteCluster, float gridSpacing) : Cluster(discreteCluster.n) {
    for (size_t i = 0; i < discreteCluster.n; i++)
    {
        x[i] = static_cast<float>(gridSpacing * discreteCluster.points[4*i]);
        y[i] = static_cast<float>(gridSpacing * discreteCluster.points[4*i+1]);
        z[i] = static_cast<float>(gridSpacing * discreteCluster.points[4*i+2]);
    }
}

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