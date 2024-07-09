# Set the Armadillo Git repository
set(ARMADILLO_GIT https://gitlab.com/conradsnicta/armadillo-code.git)

# Set the Armadillo version
set(ARMADILLO_VERSION 14.0.x) # Adjust the version as necessary

# Add Armadillo as an external project
ExternalProject_Add(armadillo
    GIT_REPOSITORY ${ARMADILLO_GIT}
    GIT_TAG        ${ARMADILLO_VERSION} # or use 'master' for the latest commit
    PREFIX         ${PROJECT_BINARY_DIR}/armadillo
    CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DARMA_USE_HDF5=ON
)

# Include directories for the Armadillo headers after building
ExternalProject_Get_Property(armadillo INSTALL_DIR)
include_directories(${INSTALL_DIR}/include)
link_directories(${INSTALL_DIR}/lib)
