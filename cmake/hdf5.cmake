# Specify URL for the HDF5 source distribution
set(HDF5_URL "https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.4.3/hdf5.tar.gz")
set(HDF5_VERSION "hdf5_1.14.4.3")  # Adjust the version as necessary

# Define the install directory
set(HDF5_INSTALL_DIR ${PROJECT_BINARY_DIR}/hdf5)

# Add HDF5 as an external project
ExternalProject_Add(hdf5
    URL ${HDF5_URL}
    PREFIX ${PROJECT_BINARY_DIR}/hdf5
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ./configure --prefix=<INSTALL_DIR> --enable-cxx
    BUILD_COMMAND make -j8
    INSTALL_COMMAND make install
)

# Include directories for the HDF5 headers after building
ExternalProject_Get_Property(hdf5 INSTALL_DIR)
include_directories(${HDF5_INSTALL_DIR}/include)
link_directories(${HDF5_INSTALL_DIR}/lib)
