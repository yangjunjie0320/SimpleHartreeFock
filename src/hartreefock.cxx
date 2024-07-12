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
};

void eigh(const arma::mat& h, const arma::mat& s, arma::mat& c, arma::vec& e) {
    arma::vec x; arma::mat y;
    arma::eig_sym(x, y, s);

    arma::mat s_half = y * arma::diagmat(arma::sqrt(x)) * y.t();
    arma::mat s_half_inv = y * arma::diagmat(1 / arma::sqrt(x)) * y.t();

    arma::mat h_prime = s_half_inv * h * s_half_inv;
    arma::eig_sym(e, c, h_prime);

    c = s_half_inv * c;
}


arma::mat get_vjk(const MoleculeInformation& mol_obj, const arma::mat& rdm1) {
    auto nao = mol_obj.nao;
    auto naux = mol_obj.naux;
    const arma::cube& cderi = mol_obj.cderi;

    arma::mat vj = arma::zeros(nao, nao);
    arma::mat vk = arma::zeros(nao, nao);

    for (int x = 0; x < naux; x++) {
        for (int mu = 0; mu < nao; mu++) {
            for (int nu = 0; nu < nao; nu++) {
                for (int lm = 0; lm < nao; lm++) {
                    for (int sg = 0; sg < nao; sg++) {
                        vj(mu, nu) += cderi(mu, nu, x) * cderi(lm, sg, x) * rdm1(lm, sg);
                        vk(mu, nu) += cderi(mu, lm, x) * cderi(nu, sg, x) * rdm1(lm, sg);
                    }
                }
            }
        }
    }

    return 2 * vj - vk;
}

int hartree_fock(MoleculeInformation mol_obj) {
    auto s1e = mol_obj.s1e;
    auto h1e = mol_obj.h1e;
    auto cderi = mol_obj.cderi;

    auto nao = mol_obj.nao;
    auto naux = mol_obj.naux;
    
    // restricted closed-shell
    assert(mol_obj.nelec_alph == mol_obj.nelec_beta);
    auto nocc = mol_obj.nelec_alph;
    auto nvir = nao - nocc;
    auto norb = nao, nmo = nao;

    arma::mat c = arma::zeros(nao, nmo);
    arma::vec e = arma::zeros(nmo);
    eigh(h1e, s1e, c, e);

    arma::mat c_occ = c.cols(0, nocc - 1);
    arma::mat c_vir = c.cols(nocc, norb - 1);
    double e_cur = 0.0, e_pre = 0.0;

    bool is_converged = false;
    bool is_max_iter = false;
    double e_tot = 0.0, de = 1.0, conv_tol = 1e-8;

    int iter = 0, max_iter = 100;

    std::cout << std::string(44, '-') << std::endl;
    std::cout << std::setw(4) << "Iter"
              << std::setw(20) << "Total Energy"
              << std::setw(20) << "Energy Change"
              << std::endl;
    std::cout << std::string(44, '-') << std::endl;

    while (not is_converged and not is_max_iter) {
        arma::mat rdm1 = c_occ * c_occ.t();
        arma::mat fock = h1e + get_vjk(mol_obj, rdm1);

        e_cur = 0.0;
        for (auto mu = 0; mu < nao; mu++) {
            for (auto nu = 0; nu < nao; nu++) {
                e_cur += rdm1(mu, nu) * (h1e(mu, nu) + fock(mu, nu));
            }
        }

        eigh(fock, s1e, c, e);
        c_occ = c.cols(0, nocc - 1);

        e_tot = e_cur + mol_obj.ene_nuc;
        de = (iter > 0) ? std::abs(e_cur - e_pre) : 1.0;
        is_converged = (de < conv_tol and iter > 0);
        is_max_iter  = (iter > max_iter);

        // format output
        if (iter > 0) {
            // Output the iteration number with padding
            std::cout << std::setw(4) << iter << " "
                      // Output the energy sum with fixed-point notation and precise format
                      << std::setw(19) << std::right << std::fixed << std::setprecision(8) << e_tot
                      // Output the energy change with scientific notation and precise format
                      << std::setw(20) << std::right << std::scientific << std::setprecision(4) << de
                      << std::endl;
        }

        e_pre = e_cur; iter++;
    }
    std::cout << std::string(44, '-') << std::endl;

    assert(is_converged);
    std::cout << "Total energy     :" << std::setw(20) << std::right << std::fixed << std::setprecision(8) << e_tot << std::endl;
    std::cout << "Reference energy :" << std::setw(20) << std::right << std::fixed << std::setprecision(8) << mol_obj.ene_rhf_ref << std::endl;
    assert(std::abs(e_tot - mol_obj.ene_rhf_ref) < 1e-6);
    return 0;  
}

int main(int argc, char** argv) {
    std::string filename = "/Users/yangjunjie/work/SimpleHartreeFock/data/h2o/0.5000/data.h5";
    hartree_fock(filename);
    return 0;
}