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
 - the spawningRadius seems to matter quite a bit for how many DMC candidates get generated; for larger N (20+), with standard (/small) spawningRadius, the first local optimization often results in a quite shit cluster (E >> 0). DMC then only manages to create maybe 2-5 clusters, with also pretty shit E. local real optimization then suddenly optimizes those from say E=100, to E=-30; an absurd improvement relative to its behaviour for smaller N, where it causes an improvement in the range [0, 4] with in most cases it being closer to 0.
 - Investigate the lower success percentage compared to the paper results
 - localDiscreteFrozenOptimization can probably be sped up further; only the freeAtom moves around; so getAtomEnergy() could keep the same neighbour points data loaded, only updating the broadcasted freeAtom
 - rewrite gradient calculation using SIMD instructions?
 - Try to implement L-BFGS for localRealOptimization; try with stepsize=1, and only half it if the energy drops for the starting stepsize. Fix center of mass. Look into preconditioning for this problem
 - check basic interpolation methods for discrete Energy calculation; the smaller lookup (and potentially higher cache hit rate) might be worth it for some basic interpolation methods
 - instead of in DMC moving the active to a random point around the target, try calculating the average direction of the targets energy, such that we can place the active atom on the opposite side such that we're likelier to get an actual totalEnergy improvement. 