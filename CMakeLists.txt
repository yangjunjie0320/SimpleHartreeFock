cmake_minimum_required(VERSION 3.22)
project(cxx-test-project)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include CMake's ExternalProject module
include(ExternalProject)
include(cmake/armadillo.cmake)

# add the executable
add_executable(hf.x ${PROJECT_SOURCE_DIR}/src/mol.cxx ${PROJECT_SOURCE_DIR}/src/rhf.cxx)
target_include_directories(hf.x PRIVATE ${PROJECT_SOURCE_DIR}/include)
