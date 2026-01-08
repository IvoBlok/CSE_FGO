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
    const size_t numberOfPoints, 
    const std::vector<DiscreteCoord>& points,
    int DISCRETE_RADIUS,
    int CUTOFF2,
    int CELL_SIZE
) : n(numberOfPoints), CUTOFF2(CUTOFF2), CELL_SIZE(CELL_SIZE) {
    // find bounding box (with margin)
    const double margin = 1.3;
    const double R = margin * DISCRETE_RADIUS;
    
    // number of cells from center to one side (including 2 layers of padding)
    int k = static_cast<int>(std::ceil(R / CELL_SIZE + 5.5));
    nx = ny = nz = 2 * k + 1;

    // center cell is exactly centered at the origin
    minX = -(k + 0.5f) * CELL_SIZE;
    maxX =  (k + 0.5f) * CELL_SIZE;

    minY = minX;
    maxY = maxX;
    minZ = minX;
    maxZ = maxX;

    const int numCells = nx * ny * nz;

    // fill cellData and cellLengths
    cellData = std::vector<CellData>(numCells);
    cellLengths = std::vector<int>(numCells, 0);
    atomIndices.resize(numberOfPoints);

    for (size_t i = 0; i < n; i++)
    {
        int c = cellIndexFromCoord(points[i]);

        if (cellLengths[c] < 15) {
            cellData[c].cellX[cellLengths[c]] = points[i][0];
            cellData[c].cellY[cellLengths[c]] = points[i][1];
            cellData[c].cellZ[cellLengths[c]] = points[i][2];
            cellData[c].cellID[cellLengths[c]] = static_cast<int32_t>(i);

            atomIndices[i].cell = c; // this is the index of the X coordinate of atom i, starting from the start of std::vector<CellData> celldata.
            atomIndices[i].slot = cellLengths[c];

            cellLengths[c]++;
        } else {
            throw std::runtime_error("given starting cluster is too dense for memory layout!");
        }
    }

    // fill padded elements with data such that they are ignored in energy calculation
    for (int c = 0; c < numCells; c++)
    {
        for (int i = cellLengths[c]; i < 16; i++) {
            cellData[c].cellX[i] = 1 << 20;
            cellData[c].cellY[i] = 1 << 20;
            cellData[c].cellZ[i] = 1 << 20;
            cellData[c].cellID[i] = -1;
        }
    }

    // calculate the cells neighbouring each cell, only including non-empty ones
    cellNeighbours.resize(numCells);
    initializeCellNeighboursList();
}

DiscreteCoord DiscreteCluster::getAtom(size_t atom) const {
    auto index = atomIndices[atom];
    return {cellData[index.cell].cellX[index.slot], cellData[index.cell].cellY[index.slot], cellData[index.cell].cellZ[index.slot]};
}

void DiscreteCluster::updateAtom(size_t atom, DiscreteCoord point) {
    auto oldIndex = atomIndices[atom];
    int newCell = cellIndexFromCoord(point);

    // if changing the atom keeps it within the same cell, just update the old point location
    if (oldIndex.cell == newCell) {
        cellData[oldIndex.cell].cellX[oldIndex.slot] = point[0];
        cellData[oldIndex.cell].cellY[oldIndex.slot] = point[1];
        cellData[oldIndex.cell].cellZ[oldIndex.slot] = point[2];
        return;
    } else {
        // if the change moved the atom out of its old cell, set its old location as padding and insert it into the new one
        cellData[oldIndex.cell].cellX[oldIndex.slot] = 1 << 20;
        cellData[oldIndex.cell].cellY[oldIndex.slot] = 1 << 20;
        cellData[oldIndex.cell].cellZ[oldIndex.slot] = 1 << 20;
        cellData[oldIndex.cell].cellID[oldIndex.slot] = -1;
        cellLengths[oldIndex.cell]--;
    }

    int insertIndex = -1;
    CellData& c = cellData[newCell];
    for (int i = 0; i < 16; i++)
    {
        if (c.cellID[i] == -1) {
            insertIndex = i;
            break;
        }
    }

    if (insertIndex == -1) {
        throw std::runtime_error("Cluster is too dense for memory layout, atom move doesn't fit!");
    } else {
        cellData[newCell].cellX[insertIndex] = point[0];
        cellData[newCell].cellY[insertIndex] = point[1];
        cellData[newCell].cellZ[insertIndex] = point[2];
        cellData[newCell].cellID[insertIndex] = static_cast<int32_t>(atom);

        atomIndices[atom].cell = newCell;
        atomIndices[atom].slot = insertIndex;

        cellLengths[newCell]++;
    }
}

float DiscreteCluster::getAtomEnergy(size_t atom, DiscreteCoord point, const std::vector<float>& lookup) const {
    const __m512i vpx = _mm512_set1_epi32(point[0]);
    const __m512i vpy = _mm512_set1_epi32(point[1]);
    const __m512i vpz = _mm512_set1_epi32(point[2]);

    const __m512i vAtom = _mm512_set1_epi32((int)atom);
    const __m512i vMinusOne = _mm512_set1_epi32(-1);
    const __m512i vCutoff2 = _mm512_set1_epi32(CUTOFF2);

    __m512 accumulate = _mm512_setzero_ps();

    int centerCell = cellIndexFromCoord(point);

    for (const int c : cellNeighbours[centerCell]) {
        if (cellLengths[c] == 0) continue;

        const CellData& cell = cellData[c];

        __m512i x = _mm512_load_epi32(&cell.cellX[0]);
        __m512i y = _mm512_load_epi32(&cell.cellY[0]);
        __m512i z = _mm512_load_epi32(&cell.cellZ[0]);
        __m512i id = _mm512_load_epi32(&cell.cellID[0]);

        __m512i dx = _mm512_sub_epi32(x, vpx);
        __m512i dy = _mm512_sub_epi32(y, vpy);
        __m512i dz = _mm512_sub_epi32(z, vpz);

        __m512i dx2 = _mm512_mullo_epi32(dx, dx);
        __m512i dy2 = _mm512_mullo_epi32(dy, dy);
        __m512i dz2 = _mm512_mullo_epi32(dz, dz);

        __m512i r2 = _mm512_add_epi32(_mm512_add_epi32(dx2, dy2), dz2);

        __mmask16 mValid = _mm512_cmpneq_epi32_mask(id, vMinusOne); // exclude padding points
        __mmask16 mNotSelf = _mm512_cmpneq_epi32_mask(id, vAtom); // exclude self-interaction
        __mmask16 mCutoff = _mm512_cmple_epi32_mask(r2, vCutoff2); // exclude points outside lookup cutoff

        __mmask16 mask = mValid & mNotSelf & mCutoff;

        __m512 energies = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, r2, lookup.data(), 4);
        accumulate = _mm512_add_ps(accumulate, energies);
    }
    return _mm512_reduce_add_ps(accumulate);
}

float DiscreteCluster::getAtomEnergy(size_t atom, const std::vector<float>& lookup) const {
    return getAtomEnergy(atom, getAtom(atom), lookup);
}

float DiscreteCluster::getClusterEnergy(const std::vector<float>& lookup) const {
    //TODO fix this :)
    float total = 0.0f;
    for (size_t i = 0; i < n; i++)
        total += getAtomEnergy(i, lookup);

    return total * 0.5f;
}

void DiscreteCluster::initializeCellNeighboursList() {
    for (int cx = 0; cx < nx; cx++) {
        for (int cy = 0; cy < ny; cy++) {
            for (int cz = 0; cz < nz; cz++) {
                int cellIndex = (cx * ny + cy) * nz + cz;
                auto& neighbours = cellNeighbours[cellIndex];
                neighbours.clear();
                
                for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) {
                    int nx_ = cx + dx;
                    int ny_ = cy + dy;
                    int nz_ = cz + dz;
                    
                    if (nx_ >= 0 && nx_ < nx && ny_ >= 0 && ny_ < ny && nz_ >= 0 && nz_ < nz) {
                        int neighbourIndex = (nx_ * ny + ny_) * nz + nz_;
                        neighbours.push_back(neighbourIndex);
                    }
                }
            }
        }
    }
}


// Cluster Implementation
// ===================================================================================
// make the vectors with lengts of a multiple of 8, such that SIMD instructions can be optimally used in getAtomEnergyAVX
Cluster::Cluster(const size_t numberOfPoints) : x((numberOfPoints + 7) & ~size_t(7)), y((numberOfPoints + 7) & ~size_t(7)), z((numberOfPoints + 7) & ~size_t(7)), n(numberOfPoints), nPadded((numberOfPoints + 7) & ~size_t(7)) {}

Cluster::Cluster(const DiscreteCluster& discreteCluster, float gridSpacing) : Cluster(discreteCluster.n) {
    for (size_t i = 0; i < discreteCluster.n; i++)
    {
        auto index = discreteCluster.atomIndices[i];
        x[i] = static_cast<float>(gridSpacing * discreteCluster.cellData[index.cell].cellX[index.slot]); // TODO can be faster, since x values of points are next to each other in (small) blocks
        y[i] = static_cast<float>(gridSpacing * discreteCluster.cellData[index.cell].cellY[index.slot]);
        z[i] = static_cast<float>(gridSpacing * discreteCluster.cellData[index.cell].cellZ[index.slot]);
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
