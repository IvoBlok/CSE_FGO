This project hosts an attempt at implementing the Fuzzy Global Optimization algorithm proposed for finding optimal atomic clusters in the paper 'Unbiased fuzzy global optimization of
Lennard-Jones clusters for N <= 1000'. It is set up for the gcc compiler suite. Additionally, a parallel version of the algorithm is hosted here. The multi-threaded approach is based on MPI, for easy compatability with larger computers. Various changes have been made to the strategy proposed by the paper, though many of them were based on limiting the complexity of implementation. The more broad algorithm has not been altered; compromises have only been made for the actual speed. 

Currently the implementation is limited to the first 3 steps from the paper; SMC is left for later. My priority now is to recreate the success rates shown in the paper when only using DMC1 (+DMC2), and hence up to this point I've only been looking at the behaviour for N < 120. 

Though originally the plan was to develop this as part of a minor program , I now work on it as a hobby besides my ongoing masters program on computational science. 

The figures in src/benchmarking are largely for debugging / profiling purposes. The right plot shows the distribution of the collective candidates across all samples at that N, and the number below is the number of samples that came to the global minimum. 

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
 - instead of in DMC moving the active to a random point around the target, try calculating the average direction