# find the hdf5 package
find_package(HDF5 REQUIRED COMPONENTS C CXX)
if (HDF5_FOUND)
    message(STATUS "HDF5 include dir: ${HDF5_INCLUDE_DIRS}")
    message(STATUS "HDF5 libs: ${HDF5_LIBRARIES}")

    include_directories(${HDF5_INCLUDE_DIRS})
    link_libraries(${HDF5_LIBRARIES})
endif()

# Set the Armadillo Git repository
set(ARMADILLO_GIT https://gitlab.com/conradsnicta/armadillo-code.git)

# Set the Armadillo version
set(ARMADILLO_VERSION 14.0.x) # Adjust the version as necessary

# Add Armadillo as an external project
ExternalProject_Add(armadillo
    GIT_REPOSITORY ${ARMADILLO_GIT}
    GIT_TAG        ${ARMADILLO_VERSION} # or use 'master' for the latest commit
    PREFIX         ${PROJECT_BINARY_DIR}/armadillo
    DOWNLOAD_EXTRACT_TIMESTAMP true
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DALLOW_OPENBLAS_MACOS=ON -DBUILD_SMOKE_TEST=OFF
)

# Include directories for the Armadillo headers after building
ExternalProject_Get_Property(armadillo INSTALL_DIR)
include_directories(${INSTALL_DIR}/include)
link_directories(${INSTALL_DIR}/lib)
link_libraries(armadillo)

# TO USE ARMA_USE_HDF5
add_compile_options(-DARMA_USE_HDF5)