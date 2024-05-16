cmake_minimum_required(VERSION 3.5)
project(pointpillars_detection CUDA CXX)

# Set build type to Debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find CUDA and GTest packages
find_package(CUDA REQUIRED QUIET)
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS}
                    ${GTEST_INCLUDE_DIRS})

# Add library
add_library(${PROJECT_NAME} STATIC src/scatter.cu)

# Add test executable if(BUILD_TESTING)
enable_testing()
add_executable(test_${PROJECT_NAME} test/test_pointpillars_detection.cpp)
set_target_properties(test_${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                      ON)
target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} ${GTEST_LIBRARIES}
                      pthread)
add_test(NAME test_${PROJECT_NAME} COMMAND test_${PROJECT_NAME})
# endif()
