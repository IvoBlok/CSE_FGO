This project hosts an attempt at implementing the Fuzzy Global Optimization algorithm proposed for finding optimal atomic clusters in the paper 'Unbiased fuzzy global optimization of
Lennard-Jones clusters for N <= 1000'. It is set up for the gcc compiler suite. Additionally, a parallel version of the algorithm is hosted here. The multi-threaded approach is based on MPI, for easy compatability with larger computers. Various changes have been made to the strategy proposed by the paper, though many of them were based on limiting the complexity of implementation. The more broad algorithm has not been altered; compromises have only been made for the actual speed. 

Currently the implementation is limited to the first 3 steps from the paper; SMC is left for later. My priority now is to recreate the success rates shown in the paper when only using DMC1 (+DMC2), and hence up to this point I've only been looking at the behaviour for N < 120. 

Though originally the plan was to develop this as part of a minor program , I now work on it as a hobby during my ongoing masters program computational science. 

The figures in src/benchmarking are largely for debugging / profiling purposes. The right plot shows the distribution of the collective candidates across all samples at that N, and the number below is the number of samples that came to the global minimum. 

Compiling
========
Cmake is used to handle the build process and dependencies. The single Cmake file creates both executables for the single-threaded and multi-threaded scenario's.

    mkdir build
    cd build
    cmake ..
    cmake --build .

Running
========
you can do a test by running first running the optimization: 

    ./main

the results can then be rendered with:

    python3 src/benchmarking/visualize_benchmark.py build/benchmark_results.json 

Note that the energy distribution plot gives the distribution of each algorithm step shared across ALL samples for that cluster size. The number below each column is the number of tries (out of the total sample size) that got at least 1 candidate to the global minimum.

Profiling
========
I now use perf and hotspot. So for example gather data with 'perf record --call-graph dwarf ./main_debug', then run 'hotspot' to get the results.
Alternatively use the GUI in (sudo) hotspot; Framepoint is probably the nicest mode. 
I also use Intel VTune here, since it has a pretty nice source -> assembly visualization. To use Intel Vtune run:

    source /opt/intel/oneapi/setvars.sh
    vtune -collect hotspots -knob enable-characterization-insights=false -r vTuneResults -- ./mainDebug
    vtune-gui <results_directory (vTuneResults)>

Profiling the parallel version can also be done with VTune:

    source /opt/intel/oneapi/setvars.sh
    mpirun --use-hwthread-cpus -np 16 vtune -collect hotspots -knob enable-characterization-insights=false -r vTuneResults -- ./mainMPIDebug
    vtune-gui <results_directory>

To Do
=========
 - use new structure to keep track of non-empty neighbour cells
 - use new CellData storage to only store non-empty cells (drastically limits memory usage per DiscreteCluster, and DiscreteCluster copying costs)
 - redo the output (JSON) storage, such that results from a run get written to the file directly, not copied to some global piece of memory
 - Investigate the lower success percentage compared to the paper results? Seems mostly resolved (for the values of N tested; 2-100)
 - the quadratic fit local real optimization, on rare occasions, instead of decreasing the energy, increases it by quite a bit; leading to occasional spikes in the RealCluster violin plots. 
 - General speedups are defintely achievable; stuff like gradient calculation, getNeighbours and others can easily be improved upon
 - (FUTURE CHANGE) Try to implement L-BFGS for localRealOptimization; try with stepsize=1, and only half it if the energy drops for the starting stepsize. Fix center of mass. Look into preconditioning for this problem
 - (FUTURE CHANGE) check basic interpolation methods for discrete Energy calculation; the smaller lookup (and potentially higher cache hit rate) might be worth it for some basic interpolation methods
 - (FUTURE CHANGE) instead of in DMC moving the active to a random point around the target, try calculating the average direction of the targets energy, such that we can place the active atom on the opposite side such that we're likelier to get an actual totalEnergy improvement. 