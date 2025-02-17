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
N=2 finds / attempt: 1000 / 1000
N=3 finds / attempt: 1000 / 1000
N=4 finds / attempt: 998 / 1000
N=5 finds / attempt: 1000 / 1000
N=6 finds / attempt: 138 / 1000
N=7 finds / attempt: 780 / 1000
N=8 finds / attempt: 795 / 1000
N=9 finds / attempt: 671 / 1000
N=10 finds / attempt: 475 / 1000
N=11 finds / attempt: 398 / 1000
N=12 finds / attempt: 368 / 1000
N=13 finds / attempt: 454 / 1000
N=14 finds / attempt: 617 / 1000
N=15 finds / attempt: 587 / 1000

TODO
=========
 - Improve random generation performance further, by potentially swithing out the std::discrete_distribution with something like the 'Alias method' for better performance in the our conditions where we construct a new distribution for each evaluation
 - Investigate the drastically lower success percentage compared to the paper results