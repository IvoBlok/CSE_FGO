cmake_minimum_required(VERSION 3.10)
project(FuzzyOptimizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_SOURCES
    src/FGO.cpp
    src/dataStructures.cpp
)

# main (performance) build
add_executable(main src/main.cpp ${COMMON_SOURCES})
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -mavx512f
        -mavx512vl
        -mavx512bw
        -mavx512dq
        -mavx2
        -mfma
        -funroll-loops
        -fno-math-errno
    >
)
target_link_options(main PRIVATE -s)
target_link_libraries(main PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# debug build
add_executable(mainDebug src/main.cpp ${COMMON_SOURCES})
target_compile_options(mainDebug PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Og
        -g3
        -march=native
        -mavx512f
        -mavx512vl
        -mavx512bw
        -mavx512dq
        -mavx2
        -mfma
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
    >
)
target_link_options(mainDebug PRIVATE -g)
target_link_libraries(mainDebug PRIVATE ${CMAKE_THREAD_LIBS_INIT})