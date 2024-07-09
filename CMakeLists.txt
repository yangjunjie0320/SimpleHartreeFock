cmake_minimum_required(VERSION 3.20)
project(cxx-test-project)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include CMake's ExternalProject module
include(ExternalProject)
include(cmake/hdf5.cmake)
include(cmake/armadillo.cmake)

# add the executable
add_executable(cxx-test-project ${PROJECT_SOURCE_DIR}/src/hartreefock.cxx)