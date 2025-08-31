cmake_minimum_required(VERSION 3.10)
project(FuzzyOptimizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

set(COMMON_SOURCES
    src/FGO.cpp
    src/dataStructures.cpp
    src/discreteDistribution.cpp
)

add_executable(main_debug
    src/main.cpp
    ${COMMON_SOURCES}
)

add_executable(main
    src/main.cpp
    ${COMMON_SOURCES}
)

add_executable(main_mpi
    src/mainMPI.cpp
    ${COMMON_SOURCES}
)

target_include_directories(main_mpi PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_compile_options(main_debug PRIVATE -g -pg)
target_link_options(main_debug PRIVATE -g -pg)

target_compile_options(main PRIVATE -O3 -march=native -ffast-math -funroll-loops)
target_link_options(main PRIVATE -O3)

target_compile_options(main_mpi PRIVATE -O3 -march=native -ffast-math -funroll-loops)
target_link_options(main_mpi PRIVATE -O3)

target_link_libraries(main_debug PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(main PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(main_mpi PRIVATE MPI::MPI_CXX ${CMAKE_THREAD_LIBS_INIT})