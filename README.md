COMPILING ON WINDOWS
========
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .  or  mingw32-make

DEBUGGING
========

For debugging, I currently use vallgrind, and I visualize the result with gprof2dot

'valgrind --tool=calgrind ./main_debug'
'gprof2dot -f callgrind callgrind.out.1 | dot -Tsvg -o output.svg'

RESULTS
=========
N=2 finds / attempts: 1000 / 1000
N=3 finds / attempts: 1000 / 1000
N=4 finds / attempts: 1000 / 1000
N=5 finds / attempts: 1000 / 1000
N=6 finds / attempts: 187 / 1000
N=7 finds / attempts: 979 / 1000
N=8 finds / attempts: 958 / 1000
N=9 finds / attempts: 970 / 1000
N=10 finds / attempts: 974 / 1000
N=11 finds / attempts: 968 / 1000
N=12 finds / attempts: 996 / 1000
N=13 finds / attempts: 999 / 1000
N=14 finds / attempts: 944 / 1000
N=15 finds / attempts: 851 / 1000
N=16 finds / attempts: 781 / 1000
N=17 finds / attempts: 753 / 1000
N=18 finds / attempts: 783 / 1000

TODO
=========
 - Improve memory handling by using std::vector instead of the 'new' allocators in the Discrete (and probably also Continuous) cluster structs. 
 - Investigate the drastically lower success percentage compared to the paper results