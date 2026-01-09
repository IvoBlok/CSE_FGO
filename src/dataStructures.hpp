#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <stdexcept>
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


struct DiscretePoints {
    std::vector<int32_t> x;
    std::vector<int32_t> y;
    std::vector<int32_t> z;
};

using DiscreteCoord = std::array<int32_t, 3>;

struct alignas(64) CellData {
    std::array<int32_t, 16> cellX;
    std::array<int32_t, 16> cellY;
    std::array<int32_t, 16> cellZ;
    std::array<int32_t, 16> cellID;
};

struct AtomIndex {
    int cell;
    int slot;
};

struct DiscreteCluster {
public:
    alignas(64) std::vector<CellData> cellData;
    std::vector<std::vector<int>> cellNeighbours;
    std::vector<int> cellLengths; // number of non-padding elements in the cell
    std::vector<AtomIndex> atomIndices; // go from atom i [0, n] to an index of that atom into cellX,Y,Z,ID
    size_t n;

    int32_t minX, minY, minZ, maxX, maxY, maxZ;
    int nx, ny, nz;

    float INV_CELL_SIZE;
    int CUTOFF2;

public:
    DiscreteCluster() = default;
    explicit DiscreteCluster(const size_t numberOfPoints, const std::vector<DiscreteCoord>& points, int DISCRETE_RADIUS , int CUTOFF2, int CELL_SIZE);
    explicit DiscreteCluster(const DiscretePoints& points, int DISCRETE_RADIUS , int CUTOFF2, int CELL_SIZE);

    DiscreteCoord getAtom(size_t atom) const;
    void updateAtom(size_t atom, DiscreteCoord point);

    float getAtomEnergy(size_t atom, DiscreteCoord point, const std::vector<float>& lookup) const;
    float getAtomEnergy(size_t atom, const std::vector<float>& lookup) const;
    float getClusterEnergy(const std::vector<float>& lookup) const;

    DiscretePoints exportPoints() const;

private:
    inline int cellIndexFromCoord(int32_t px, int32_t py, int32_t pz) const {
        int cx = static_cast<int>((px - minX) * INV_CELL_SIZE);
        int cy = static_cast<int>((py - minY) * INV_CELL_SIZE);
        int cz = static_cast<int>((pz - minZ) * INV_CELL_SIZE);

        //if (cx < 0 || cx > nx - 1 || cy < 0 || cy > ny - 1 || cz < 0 || cz > nz - 1)
        //    throw std::runtime_error("point is outside domain of cluster; either implement some moving window, recenter the structure, or make the domain larger by default");

        return (cx * ny + cy) * nz + cz;
    }

    inline int cellIndexFromCoord(DiscreteCoord point) const {
        return cellIndexFromCoord(point[0], point[1], point[2]);
    }

    void initializeCellNeighboursList();
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
    explicit Cluster(const DiscretePoints& discretePoints, float gridSpacing);

    Cluster(const Cluster& other) = default;
    Cluster& operator=(const Cluster& other) = default;

    float getDistanceSquared(size_t atomIndex1, size_t atomIndex2) const;

    float getAtomEnergyAVX(size_t atomIndex) const;
    float getClusterEnergyAVX() const;
    
    void getClusterGradient(std::vector<float>& gradX, std::vector<float>& gradY, std::vector<float>& gradZ) const;
};

#endif // DATA_STRUCTURES_H