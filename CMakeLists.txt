cmake_minimum_required(VERSION 3.10)
project(FuzzyOptimizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add debug executable
add_executable(main_debug
    src/main.cpp
    src/FGO.cpp
    src/dataStructures.cpp
    src/discreteDistribution.cpp
)

# Add optimized executable
add_executable(main
    src/main.cpp
    src/FGO.cpp
    src/dataStructures.cpp
    src/discreteDistribution.cpp
)

# Set properties for debug executable
set_target_properties(main_debug PROPERTIES
    COMPILE_FLAGS "-g -pg"        # Debug symbols
    LINK_FLAGS "-g -pg"           # Link with debug symbols
)

# Set properties for optimized executable
set_target_properties(main PROPERTIES
    COMPILE_FLAGS "-O3"       # Optimization flags
    LINK_FLAGS "-O3"          # Link with optimization flags
)

# Link libraries (if needed, you can replace this with specific libraries)
target_link_libraries(main_debug PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(main PRIVATE ${CMAKE_THREAD_LIBS_INIT})