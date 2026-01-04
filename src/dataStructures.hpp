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


using DiscreteCoord = std::array<int32_t, 3>;

struct Cell {
    int start;
    int count; // padded to multiple of 16
};

struct DiscreteCluster {
public:
    //TODO if we can ensure there are no less then 16 points in each cell, we can improve memory layout by storing a single std::vector<CellData>, where each CellData stores X[16],Y[16],Z[16],ID[16] next to each other
    alignas(64) std::vector<int32_t> cellX;
    alignas(64) std::vector<int32_t> cellY;
    alignas(64) std::vector<int32_t> cellZ;
    alignas(64) std::vector<int32_t> cellID;

    std::vector<Cell> cells;

    std::vector<int> atomIndices; // go from atom i [0, n] to an index of that atom into cellX,Y,Z,ID
    size_t n;

    int32_t minX, minY, minZ, maxX, maxY, maxZ;
    int nx, ny, nz;

    int CELL_SIZE;
    int CUTOFF2;

public:
    DiscreteCluster() = default;
    explicit DiscreteCluster(const size_t numberOfPoints, const std::vector<DiscreteCoord>& points, int DISCRETE_RADIUS , int CUTOFF2, int CELL_SIZE);

    DiscreteCoord getAtom(size_t atom) const;
    void updateAtom(size_t atom, DiscreteCoord point);

    float getAtomEnergy(size_t atom, DiscreteCoord point, const std::vector<float>& lookup) const;
    float getAtomEnergy(size_t atom, const std::vector<float>& lookup) const;
    float getClusterEnergy(const std::vector<float>& lookup) const;

private:
    inline int cellIndexFromCoord(int32_t px, int32_t py, int32_t pz) const {
        int cx = (px - minX) / CELL_SIZE;
        int cy = (py - minY) / CELL_SIZE;
        int cz = (pz - minZ) / CELL_SIZE;

        // Optional safety (recommended at least in debug)
        cx = std::min(std::max(cx, 0), nx - 1);
        cy = std::min(std::max(cy, 0), ny - 1);
        cz = std::min(std::max(cz, 0), nz - 1);

        return (cx * ny + cy) * nz + cz;
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
    explicit Cluster(const DiscreteCluster& discreteCluster, float gridSpacing);

    Cluster(const Cluster& other) = default;
    Cluster& operator=(const Cluster& other) = default;

    float getDistanceSquared(size_t atomIndex1, size_t atomIndex2) const;

    float getAtomEnergyAVX(size_t atomIndex) const;
    float getClusterEnergyAVX() const;
    
    void getClusterGradient(std::vector<float>& gradX, std::vector<float>& gradY, std::vector<float>& gradZ) const;
};

#endif // DATA_STRUCTURES_H