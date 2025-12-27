This project hosts an attempt at implementing the Fuzzy Global Optimization algorithm 'recently' proposed for finding optimal atomic clusters. It is set up for the gcc compiler suite. Additionally, a parallel version of the algorithm is hosted here. The multi-threaded approach is based on MPI, for easy compatability with super/cluster computers.

Compiling
========
Cmake is used to handle the build process and dependencies. The single Cmake file creates both executables for the single-threaded and multi-threaded scenario's.

    mkdir build
    cd build
    cmake ..
    cmake --build .

PROFILING
========
I now use perf and hotspot. So for example gather data with 'perf record --call-graph dwarf ./main_debug', then run 'hotspot' to get the results.

TODO
=========
 - rewrite gradient calculation using SIMD instructions?
 - Investigate local real optimization. Potentially normalizing the gradient in one way or another might be an improvement. etc..., or something like replacing the quadratic fitting for line-search + armijo rule or smth
 - check spawnRadiusFactor = 0.4
 - parallelization
 - check lookup tables / interpolation methods etc for LJPotential evaluations, since current versions just calculate the potential (very quickly)
 - Write AVX atomEnergy version that uses neighbours
 - Only check energy with neighbours in the localDiscreteFrozenOptimization steps
 - In a similar vein, in many cases we don't need to calculate clusterPotential from scratch, since generally only 1 or 2 atoms got moved; we just need to calculate the change these moves caused
 - Investigate the drastically lower success percentage compared to the paper results