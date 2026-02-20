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
DiscreteCluster::DiscreteCluster(
    DiscretePoints& points,
    int DISCRETE_RADIUS,
    int CUTOFF2,
    int CELL_SIZE
) : n(points.x.size()), CUTOFF2(CUTOFF2), INV_CELL_SIZE(1.0f / (float)CELL_SIZE) {
    // find bounding box (with margin)
    const double R = 1.3f * DISCRETE_RADIUS;
    
    // number of cells from center to one side (including 5 layers of padding)
    int k = static_cast<int>(std::ceil(R / CELL_SIZE + 5));
    nCells = 2 * k + 1;

    // center cell is exactly centered at the origin
    minCells = -(k + 0.5f) * CELL_SIZE;
    const int numCells = nCells * nCells * nCells;


    // copy point data
    x = points.x;
    y = points.y;
    z = points.z;

    // sort each point into the appropriate cell
    cells.resize(numCells);
    cellNeighbours.resize(numCells);
    atomCell.resize(n);

    for (size_t i = 0; i < n; i++) {
        int c = cellIndexFromCoord(x[i], y[i], z[i]);
        cells[c].push_back(i);
        atomCell[i] = c;
    }

    // set up cell neighbours
    for (int cx = 0; cx < nCells; cx++)
    for (int cy = 0; cy < nCells; cy++)
    for (int cz = 0; cz < nCells; cz++) {
        int c = (cx * nCells + cy) * nCells + cz;

        auto& neighbours = cellNeighbours[c];
        neighbours.clear();

        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++) {
            int nx = cx + dx;
            int ny = cy + dy;
            int nz = cz + dz;
            
            if (nx >= 0 && nx < nCells && ny >= 0 && ny < nCells && nz >= 0 && nz < nCells) {
                int neighbourIndex = (nx * nCells + ny) * nCells + nz;
                neighbours.push_back(neighbourIndex);
            }
        }
    }
}

DiscreteCoord DiscreteCluster::getAtom(size_t atom) const {
    return {x[atom], y[atom], z[atom]};
}

void DiscreteCluster::updateAtom(size_t atom, DiscreteCoord point) {
    auto oldCell = atomCell[atom];
    auto newCell = cellIndexFromCoord(point);

    x[atom] = point[0];
    y[atom] = point[1];
    z[atom] = point[2];

    if (oldCell != newCell) {
        cells[oldCell].erase(std::remove(cells[oldCell].begin(), cells[oldCell].end(), atom), cells[oldCell].end());
        cells[newCell].push_back(atom);
        atomCell[atom] = newCell;
    }
}

DiscretePoints DiscreteCluster::gatherNeighbourBuffer(size_t freeAtom, DiscreteCoord point) const {
    DiscretePoints buffer;
    buffer.x.reserve(n);
    buffer.y.reserve(n);
    buffer.z.reserve(n);
    
    //TODO check distances here (efficiently), and only accept those within the cutoff range. This should drastically lower the number of points in the buffer (~1/0,15 = 6,7x smaller)
    //TODO reuse the same buffer each time, and probably even keep a large amount stored. Then maybe return how many multiples of 16 are in it, and only loop over that integer amount in getAtomEnergy

    int centerCell = cellIndexFromCoord(point);

    for (const auto& neighbourCell : cellNeighbours[centerCell])
    {
        for (const auto& atom : cells[neighbourCell])
        {
            if (freeAtom == atom) continue; // skip self, such that energy calculations don't take the contribution from self relative to self into consideration

            buffer.x.push_back(x[atom]);
            buffer.y.push_back(y[atom]);
            buffer.z.push_back(z[atom]);
        }
    }

    // add padding to ensure the buffer size per axis is a multiple of 16 for easy loading with simd instructions
    size_t nNeighbours = buffer.x.size();
    size_t padded = (nNeighbours + 15) & ~15;

    for (size_t i = nNeighbours; i < padded; i++) {
        buffer.x.push_back(1 << 12);
        buffer.y.push_back(1 << 12);
        buffer.z.push_back(1 << 12);
    }
    
    return buffer;
}

float DiscreteCluster::getAtomEnergy(DiscreteCoord point, const DiscretePoints& neighbours, const std::vector<float>& lookup) const {
    const __m512i vpx = _mm512_set1_epi32(point[0]);
    const __m512i vpy = _mm512_set1_epi32(point[1]);
    const __m512i vpz = _mm512_set1_epi32(point[2]);

    const __m512i vCutoff2 = _mm512_set1_epi32(CUTOFF2);
    __m512 accumulate = _mm512_setzero_ps();

    for (size_t i = 0; i < neighbours.x.size(); i += 16) {
        __m512i x = _mm512_loadu_epi32(&neighbours.x[i]);
        __m512i y = _mm512_loadu_epi32(&neighbours.y[i]);
        __m512i z = _mm512_loadu_epi32(&neighbours.z[i]);

        __m512i dx = _mm512_sub_epi32(x, vpx);
        __m512i dy = _mm512_sub_epi32(y, vpy);
        __m512i dz = _mm512_sub_epi32(z, vpz);

        __m512i dx2 = _mm512_mullo_epi32(dx, dx);
        __m512i dy2 = _mm512_mullo_epi32(dy, dy);
        __m512i dz2 = _mm512_mullo_epi32(dz, dz);

        __m512i r2 = _mm512_add_epi32(_mm512_add_epi32(dx2, dy2), dz2);

        __mmask16 cutoffMask = _mm512_cmple_epi32_mask(r2, vCutoff2); // exclude points outside lookup cutoff

        __m512 energies = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), cutoffMask, r2, lookup.data(), 4);
        accumulate = _mm512_add_ps(accumulate, energies);
    }
    return _mm512_reduce_add_ps(accumulate);
}

float DiscreteCluster::getAtomEnergy(size_t atom, const DiscretePoints& neighbours, const std::vector<float>& lookup) const {
    return getAtomEnergy(getAtom(atom), neighbours, lookup);
}

float DiscreteCluster::getAtomEnergySlow(size_t atom, const std::vector<float>& lookup) const {
    auto buffer = gatherNeighbourBuffer(atom, getAtom(atom));
    return getAtomEnergy(getAtom(atom), buffer, lookup);
}

float DiscreteCluster::getClusterEnergy(const std::vector<float>& lookup) const {
    //TODO fix this :)
    float total = 0.0f;
    for (size_t i = 0; i < n; i++) {
        auto buffer = gatherNeighbourBuffer(i, getAtom(i));
        total += getAtomEnergy(i, buffer, lookup);
    }

    return total * 0.5f;
}

DiscretePoints DiscreteCluster::exportPoints() const {
    DiscretePoints result;

    result.x = x;
    result.y = y;
    result.z = z;
    
    return result;
}



// Cluster Implementation
// ===================================================================================
// make the vectors with lengts of a multiple of 8, such that SIMD instructions can be optimally used in getAtomEnergyAVX
Cluster::Cluster(const size_t numberOfPoints) : x((numberOfPoints + 7) & ~size_t(7)), y((numberOfPoints + 7) & ~size_t(7)), z((numberOfPoints + 7) & ~size_t(7)), n(numberOfPoints), nPadded((numberOfPoints + 7) & ~size_t(7)) {}

Cluster::Cluster(const DiscretePoints& discretePoints, float gridSpacing) : Cluster(discretePoints.x.size()) {
    for (size_t i = 0; i < n; i++)
    {
        x[i] = static_cast<float>(gridSpacing * discretePoints.x[i]);
        y[i] = static_cast<float>(gridSpacing * discretePoints.y[i]);
        z[i] = static_cast<float>(gridSpacing * discretePoints.z[i]);
    }
}

float Cluster::getDistanceSquared(const size_t atomIndex1, const size_t atomIndex2) const {
    const float dx = x[atomIndex1] - x[atomIndex2];
    const float dy = y[atomIndex1] - y[atomIndex2];
    const float dz = z[atomIndex1] - z[atomIndex2];
    return dx*dx + dy*dy + dz*dz;
}

float Cluster::getAtomEnergyAVX(size_t atomIndex) const {
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
        // ======================================
        __m256 dx = _mm256_sub_ps(xi, xj);
        __m256 dy = _mm256_sub_ps(yi, yj);
        __m256 dz = _mm256_sub_ps(zi, zj);

        dx = _mm256_mul_ps(dx, dx);
        dy = _mm256_mul_ps(dy, dy);
        dz = _mm256_mul_ps(dz, dz);

        __m256 r2 = _mm256_add_ps(dx, _mm256_add_ps(dy, dz));

        // calculate energies
        // ======================================
        // add small epsilon to all values to avoid division by zero
        __m256 r2_safe = _mm256_add_ps(r2, _mm256_set1_ps(1e-10f));
        
        // approximate 1/r2 by rcp + a Newton-Raphson refinement step
        __m256 inv_r2 = _mm256_rcp_ps(r2_safe);
        inv_r2 = _mm256_mul_ps(inv_r2, _mm256_fnmadd_ps(r2_safe, inv_r2, _mm256_set1_ps(2.0f)));
        
        __m256 inv_r4 = _mm256_mul_ps(inv_r2, inv_r2);
        __m256 inv_r6 = _mm256_mul_ps(inv_r4, inv_r2);
        __m256 inv_r12 = _mm256_mul_ps(inv_r6, inv_r6);
        
        __m256 energies =  _mm256_fnmadd_ps(inv_r6, _mm256_set1_ps(2.0f), inv_r12);

        // catch edgecases
        // ======================================
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

float Cluster::getClusterEnergyAVX() const {
    float total = 0.f;

    for (size_t i = 0; i < n; i++)
        total += getAtomEnergyAVX(i);

    return total * 0.5f;
}

void Cluster::getClusterGradient(std::vector<float>& gradX, std::vector<float>& gradY, std::vector<float>& gradZ) const {
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
            
            float force;
            if (r2 < 1e-10f) {
                force = std::numeric_limits<float>::infinity();
            } else {
                const float inv_r2 = 1.0f / r2;
                const float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                const float inv_r8 = inv_r6 * inv_r2;
                const float inv_r14 = inv_r8 * inv_r6;
                force = 12.f * (inv_r8 - inv_r14);
            }
            
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
