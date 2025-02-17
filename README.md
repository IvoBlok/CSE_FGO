COMPILING ON WINDOWS
========
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .  or  mingw32-make

RUNNING ON DELFTBLUE
========

module load 2023r1 openmpi
mpicxx -o mainMPI mainMPI.cpp FGO.cpp dataStructures.cpp discreteDistribution.cpp -lm
srun --mem-per-cpi=128MB --account=Education-EEMCS-Courses-CSEMinor --partition=compute --cpus-per-task=1 --time=0-01:00:00 --ntasks=8 mainMPI > output.txt

PROFILING
========

For profiling, I currently use vallgrind, and I visualize the result with gprof2dot

'valgrind --tool=calgrind ./main_debug'
'gprof2dot -f callgrind callgrind.out.1 | dot -Tsvg -o output.svg'

or just run the program in msvc, and use the internal profilers there

RESULTS
=========
N=2 finds / attempts: 100 / 100
N=3 finds / attempts: 100 / 100
N=4 finds / attempts: 100 / 100
N=5 finds / attempts: 100 / 100
N=6 finds / attempts: 22 / 100
N=7 finds / attempts: 97 / 100
N=8 finds / attempts: 96 / 100
N=9 finds / attempts: 99 / 100
N=10 finds / attempts: 96 / 100
N=11 finds / attempts: 93 / 100
N=12 finds / attempts: 99 / 100
N=13 finds / attempts: 100 / 100
N=14 finds / attempts: 98 / 100
N=15 finds / attempts: 85 / 100
N=16 finds / attempts: 77 / 100
N=17 finds / attempts: 66 / 100
N=18 finds / attempts: 78 / 100
N=19 finds / attempts: 91 / 100
N=20 finds / attempts: 67 / 100
N=21 finds / attempts: 59 / 100
N=22 finds / attempts: 60 / 100
N=23 finds / attempts: 80 / 100
N=24 finds / attempts: 58 / 100
N=25 finds / attempts: 20 / 100
N=26 finds / attempts: 53 / 100
N=27 finds / attempts: 11 / 100
N=28 finds / attempts: 1 / 100
N=29 finds / attempts: 5 / 100
N=30 finds / attempts: 0 / 100
N=31 finds / attempts: 11 / 100
N=32 finds / attempts: 4 / 100
N=33 finds / attempts: 9 / 100
N=34 finds / attempts: 7 / 100
N=35 finds / attempts: 8 / 100
N=36 finds / attempts: 11 / 100
N=37 finds / attempts: 2 / 100
N=38 finds / attempts: 0 / 100
N=39 finds / attempts: 7 / 100
N=40 finds / attempts: 10 / 100
N=41 finds / attempts: 11 / 100
N=42 finds / attempts: 6 / 100
N=43 finds / attempts: 4 / 100
N=44 finds / attempts: 2 / 100
N=45 finds / attempts: 4 / 100
N=46 finds / attempts: 1 / 100
N=47 finds / attempts: 1 / 100
N=48 finds / attempts: 0 / 100
N=49 finds / attempts: 3 / 100

TODO
=========
 - Improve performance by adjusting the 'DiscreteCluster::getAtomEnergy' and 'DiscreteCluster::getAtomNeighbours' implementations / number of calls
 - Improve memory handling by using std::vector instead of the 'new' allocators in the Discrete (and probably also Continuous) cluster structs. 
 - Investigate the drastically lower success percentage compared to the paper results