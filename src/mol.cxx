#include <iostream>
#include <cassert>
#include <armadillo>
#include <string>
#include <cstdio>
#include <fstream> // Add this include for file checking

// molecule information, T is the type of the data class
class MoleculeInformation {
public:
    int nelec_alph;
    int nelec_beta;
    int nao, naux;

    double ene_nuc, ene_rhf_ref, ene_uhf_ref;

    arma::imat atm;
    arma::imat env;
    arma::mat bas;

    arma::mat ovlap, hcore;
    arma::cube cderi;

    MoleculeInformation(std::string filename) {
        // check if the hdf5 file exists
        std::ifstream file(filename); // Check if the file can be opened
        if (!file) {
            std::cerr << "Error: HDF5 file " << filename << " does not exist." << std::endl;
            throw std::runtime_error("HDF5 file does not exist");
        }
        file.close(); // Close the file after checking

        // load the molecule information from the hdf5 file
        arma::ivec nelec; nelec.load(arma::hdf5_name(filename, "nelec"));
        nelec_alph = nelec(0); nelec_beta = nelec(1);

        ovlap.load(arma::hdf5_name(filename, "ovlp"));
        hcore.load(arma::hdf5_name(filename, "hcore"));
        cderi.load(arma::hdf5_name(filename, "cderi"));

        nao = hcore.n_rows; // Corrected from s1e to hcore
        naux = cderi.n_slices;

        arma::vec ene_tmp; 
        ene_tmp.load(arma::hdf5_name(filename, "ene_nuc")); ene_nuc = ene_tmp(0);
        ene_tmp.load(arma::hdf5_name(filename, "ene_rhf")); ene_rhf_ref = ene_tmp(0);
        ene_tmp.load(arma::hdf5_name(filename, "ene_uhf")); ene_uhf_ref = ene_tmp(0);
    }

    ~MoleculeInformation() {
        // The destructor
    }
};

arma::mat& get_ovlap(const MoleculeInformation& mol_obj) {
    return mol_obj.ovlap;
}

arma::mat& get_hcore(const MoleculeInformation& mol_obj) {
    return mol_obj.hcore;
}

arma::cube& get_cderi(const MoleculeInformation& mol_obj) {
    return mol_obj.cderi;
}