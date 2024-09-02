#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "mol.cxx"
#include <armadillo>
#include <string>

// Helper function to create a temporary HDF5 file for testing
void create_test_hdf5_file(const std::string& filename) {
    arma::Col<int> nelec = {5, 5};
    arma::Mat<double> ovlp = arma::eye(10, 10);
    arma::Mat<double> hcore = arma::randu(10, 10);
    arma::Cube<double> cderi = arma::randu(10, 10, 5);
    arma::Col<double> ene_nuc = {-100.0};
    arma::Col<double> ene_rhf = {-200.0};
    arma::Col<double> ene_uhf = {-199.0};

    nelec.save(arma::hdf5_name(filename, "nelec"));
    ovlp.save(arma::hdf5_name(filename, "ovlp"));
    hcore.save(arma::hdf5_name(filename, "hcore"));
    cderi.save(arma::hdf5_name(filename, "cderi"));
    ene_nuc.save(arma::hdf5_name(filename, "ene_nuc"));
    ene_rhf.save(arma::hdf5_name(filename, "ene_rhf"));
    ene_uhf.save(arma::hdf5_name(filename, "ene_uhf"));
}

TEST_CASE("MoleculeInformation class tests", "[MoleculeInformation]") {
    std::string test_filename = "test_molecule.h5";
    create_test_hdf5_file(test_filename);

    SECTION("Constructor and basic properties") {
        MoleculeInformation<double> mol(test_filename);

        REQUIRE(mol.nelec_alph == 5);
        REQUIRE(mol.nelec_beta == 5);
        REQUIRE(mol.nao == 10);
        REQUIRE(mol.naux == 5);
        REQUIRE(mol.ene_nuc == Approx(-100.0));
        REQUIRE(mol.ene_rhf_ref == Approx(-200.0));
        REQUIRE(mol.ene_uhf_ref == Approx(-199.0));
    }

    SECTION("Getter functions") {
        MoleculeInformation<double> mol(test_filename);

        REQUIRE(mol.get_ovlp().n_rows == 10);
        REQUIRE(mol.get_ovlp().n_cols == 10);
        REQUIRE(mol.get_hcore().n_rows == 10);
        REQUIRE(mol.get_hcore().n_cols == 10);
        REQUIRE(mol.get_cderi().n_rows == 10);
        REQUIRE(mol.get_cderi().n_cols == 10);
        REQUIRE(mol.get_cderi().n_slices == 5);
    }

    // Clean up the temporary file
    std::remove(test_filename.c_str());
}