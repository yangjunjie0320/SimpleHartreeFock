#include <iostream>
#include <cassert>
#include <armadillo>
#include <string>
#include <cstdio>

class MoleculeInformation {
public:
    int nelec_alph;
    int nelec_beta;
    int nao, naux;

    double ene_nuc, ene_rhf_ref, ene_uhf_ref;

    arma::mat  s1e;
    arma::mat  h1e;
    arma::cube cderi;

    arma::imat atm;
    arma::mat  bas;
    arma::imat env;

    // arma::imat
    MoleculeInformation(std::string filename) {
        arma::ivec nelec; nelec.load(arma::hdf5_name(filename, "nelec"));
        nelec_alph = nelec(0);
        nelec_beta = nelec(1);

        s1e.load(arma::hdf5_name(filename, "ovlp"));
        h1e.load(arma::hdf5_name(filename, "hcore"));
        cderi.load(arma::hdf5_name(filename, "cderi"));

        nao = s1e.n_rows;
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

arma::mat MoleculeInformation::get_hcore() {
    return this->h1e;
}

arma::mat MoleculeInformation::get_ovlp() {
    return this->s1e;
}

arma::mat MoleculeInformation::get_cderi() {
    return this->cderi;
}