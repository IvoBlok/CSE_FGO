COMPILING
========
    mkdir build
    cd build
    cmake ..
    cmake --build .

RUNNING PARALLEL VERSION ON LOCAL SYSTEM
========
    mkdir buildMPI
    cd buildMPI
    mpicxx -O3 -march=native -ffast-math -funroll-loops -o mainMPI ../src/mainMPI.cpp ../src/FGO.cpp ../src/dataStructures.cpp ../src/iscreteDistribution.cpp -lm
    mpirun --use-hwthread-cpus -np 8 mainMPI

RUNNING ON DELFTBLUE
========
    module load 2023r1 openmpi
    mpicxx -o mainMPI mainMPI.cpp FGO.cpp dataStructures.cpp discreteDistribution.cpp -lm
    srun --mem-per-cpi=128MB --account=Education-EEMCS-Courses-CSEMinor --partition=compute --cpus-per-task=1 --time=0-01:00:00 --ntasks=8 mainMPI > output.txt

PROFILING
========

For profiling, I currently use vallgrind, and I visualize the result with gprof2dot

    valgrind --tool=calgrind ./main_debug
    gprof2dot -f callgrind callgrind.out.1 | dot -Tsvg -o output.svg

or just run the program in msvc, and use the internal profilers there

RESULTS
=========
N=2 finds / attempts: 100 / 100 <br />
N=3 finds / attempts: 100 / 100 <br />
N=4 finds / attempts: 100 / 100 <br />
N=5 finds / attempts: 100 / 100 <br />
N=6 finds / attempts: 18 / 100 <br />
N=7 finds / attempts: 93 / 100 <br />
N=8 finds / attempts: 95 / 100 <br />
N=9 finds / attempts: 97 / 100 <br />
N=10 finds / attempts: 99 / 100 <br />
N=11 finds / attempts: 99 / 100 <br />
N=12 finds / attempts: 100 / 100 <br />
N=13 finds / attempts: 100 / 100 <br />
N=14 finds / attempts: 96 / 100 <br />
N=15 finds / attempts: 82 / 100 <br />
N=16 finds / attempts: 80 / 100 <br />
N=17 finds / attempts: 74 / 100 <br />
N=18 finds / attempts: 75 / 100 <br />
N=19 finds / attempts: 88 / 100 <br />
N=20 finds / attempts: 61 / 100 <br />
N=21 finds / attempts: 61 / 100 <br />
N=22 finds / attempts: 75 / 100 <br />
N=23 finds / attempts: 78 / 100 <br />
N=24 finds / attempts: 53 / 100 <br />
N=25 finds / attempts: 60 / 100 <br />
N=26 finds / attempts: 62 / 100 <br />
N=27 finds / attempts: 42 / 100 <br />
N=28 finds / attempts: 55 / 100 <br />
N=29 finds / attempts: 49 / 100 <br />
N=30 finds / attempts: 34 / 100 <br />
N=31 finds / attempts: 16 / 100 <br />
N=32 finds / attempts: 11 / 100 <br />
N=33 finds / attempts: 11 / 100 <br />
N=34 finds / attempts: 13 / 100 <br />
N=35 finds / attempts: 17 / 100 <br />
N=36 finds / attempts: 12 / 100 <br />
N=37 finds / attempts: 8 / 100 <br />
N=38 finds / attempts: 1 / 100 <br />
N=39 finds / attempts: 14 / 100 <br />
N=40 finds / attempts: 15 / 100 <br />
N=41 finds / attempts: 11 / 100 <br />

TODO
=========
 - Improve performance by adjusting the 'DiscreteCluster::getAtomEnergy' and 'DiscreteCluster::getAtomNeighbours' implementations / number of calls
 - Improve memory handling by using std::vector instead of the 'new' allocators in the Discrete (and probably also Continuous) cluster structs. 
 - Investigate the drastically lower success percentage compared to the paper results
 - Investigate local real optimization. Potentially normalizing the gradient in one way or another might be an improvement. etc...