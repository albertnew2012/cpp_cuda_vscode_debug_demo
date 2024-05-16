cmake_minimum_required(VERSION 3.10)
project(pointpillars_detection CUDA CXX)

# Set build type to Debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find CUDA package
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Add the source files
add_executable(test_pointpillars_detection src/main.cpp src/scatter.cu)

# Set CUDA properties for the target
set_target_properties(test_pointpillars_detection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Ensure linking with CUDA and C++ standard libraries
target_link_libraries(test_pointpillars_detection ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
