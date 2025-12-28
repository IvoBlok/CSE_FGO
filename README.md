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
Alternatively use the GUI in (sudo) hotspot; Framepoint is probably the nicest mode. 

TODO
=========
 - Investigate the lower success percentage compared to the paper results
 - rewrite gradient calculation using SIMD instructions?
 - Try to implement L-BFGS for localRealOptimization; try with stepsize=1, and only half it if the energy drops for the starting stepsize. Fix center of mass. Look into preconditioning for this problem
 - add results storage to mainMPI; probably only store energies of each candidate etc, and maybe the best cluster for each single run; storing all candidate clusters (x, y, z) makes for massive json files (and memory)
 - In some cases it seemingly gets stuck in some loop somewhere
 - check lookup tables / interpolation methods etc for LJPotential evaluations, since current versions just calculates the potential (very quickly)
 - Write AVX atomEnergy version that uses neighbours
 - Only check energy with neighbours in the localDiscreteFrozenOptimization steps