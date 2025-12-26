cmake_minimum_required(VERSION 3.10)
project(FuzzyOptimizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_SOURCES
    src/FGO.cpp
    src/dataStructures.cpp
    src/discreteDistribution.cpp
)

# main (performance) build
add_executable(main src/main.cpp ${COMMON_SOURCES})
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -mavx2
        -mfma
        -ffast-math
        -funroll-loops
        -flto
        -fno-math-errno
        -fassociative-math
    >
)
target_link_options(main PRIVATE -flto -s)
target_link_libraries(main PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# debug build
add_executable(main_debug src/main.cpp ${COMMON_SOURCES})
target_compile_options(main_debug PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Og
        -g3
        -march=native
        -mavx2
        -mfma
        -fno-optimize-sibling-calls
    >
)
target_link_options(main_debug PRIVATE -g)
target_link_libraries(main_debug PRIVATE ${CMAKE_THREAD_LIBS_INIT})