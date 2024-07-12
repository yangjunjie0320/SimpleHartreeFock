#include <iostream>
#include <cassert>
#include <armadillo>
#include <string>
#include <cstdio>

void eigh(const arma::mat& h, const arma::mat& s, arma::mat& c, arma::vec& e) {
    arma::vec x; arma::mat y;
    arma::eig_sym(x, y, s);

    arma::mat s_half = y * arma::diagmat(arma::sqrt(x)) * y.t();
    arma::mat s_half_inv = y * arma::diagmat(1 / arma::sqrt(x)) * y.t();

    arma::mat h_prime = s_half_inv * h * s_half_inv;
    arma::eig_sym(e, c, h_prime);

    c = s_half_inv * c;
}


arma::mat get_coul(const MoleculeInformation& mol_obj, const arma::mat& rdm1) {
    auto nao = mol_obj.nao;
    auto naux = mol_obj.naux;
    const arma::cube& cderi = mol_obj.cderi;

    arma::mat vj = arma::zeros(nao, nao);

    for (int x = 0; x < naux; x++) {
        for (int mu = 0; mu < nao; mu++) {
            for (int nu = 0; nu < nao; nu++) {
                for (int lm = 0; lm < nao; lm++) {
                    for (int sg = 0; sg < nao; sg++) {
                        vj(mu, nu) += cderi(mu, nu, x) * cderi(lm, sg, x) * rdm1(lm, sg);
                    }
                }
            }
        }
    }

    return vj;
}

arma::mat get_exch(const MoleculeInformation& mol_obj, const arma::mat& rdm1) {
    auto nao = mol_obj.nao;
    auto naux = mol_obj.naux;
    const arma::cube& cderi = mol_obj.cderi;
    arma::mat vk = arma::zeros(nao, nao);

    for (int x = 0; x < naux; x++) {
        for (int mu = 0; mu < nao; mu++) {
            for (int nu = 0; nu < nao; nu++) {
                for (int lm = 0; lm < nao; lm++) {
                    for (int sg = 0; sg < nao; sg++) {
                        vk(mu, nu) += cderi(mu, lm, x) * cderi(nu, sg, x) * rdm1(lm, sg);
                    }
                }
            }
        }
    }

    return vk;
}

double solve_rhf(MoleculeInformation mol_obj, int max_iter = 100, double conv_tol = 1e-8) {
    auto s1e = mol_obj.get_ovlp();
    auto h1e = mol_obj.get_hcore();
    auto cderi = mol_obj.get_cderi();

    auto nao = mol_obj.nao;
    auto naux = mol_obj.naux;
    
    // restricted closed-shell
    assert(mol_obj.nelec_alph == mol_obj.nelec_beta);
    auto nocc = mol_obj.nelec_alph;
    auto nvir = nao - nocc;
    auto norb = nao, nmo = nao;

    arma::mat coeff_mo = arma::zeros(nao, nmo);
    arma::vec energ_mo = arma::zeros(nmo);
    eigh(h1e, s1e, coeff_mo, energ_mo);

    arma::mat c_occ = coeff_mo.cols(0,    nocc - 1);
    arma::mat c_vir = coeff_mo.cols(nocc, norb - 1);
    double e_cur = 0.0, e_pre = 0.0;
    double e_tot = 0.0, err = 1.0;
    bool is_converged = false, is_max_iter = false;

    std::cout << std::string(44, '-') << std::endl;
    std::cout << std::setw(4) << "Iter"
              << std::setw(20) << "Total Energy"
              << std::setw(20) << "Energy Change"
              << std::endl;
    std::cout << std::string(44, '-') << std::endl;

    int iter = 0;
    while (not is_converged and not is_max_iter) {
        arma::mat rdm1 = c_occ * c_occ.t() * 2.0;
        arma::mat fock = h1e + get_coul(mol_obj, rdm1) - 0.5 * get_exch(mol_obj, rdm1);
        e_cur = arma::accu(rdm1 % (h1e + fock)) * 0.5;

        eigh(fock, s1e, coeff_mo, energ_mo);
        c_occ = coeff_mo.cols(0, nocc - 1);

        e_tot = e_cur + mol_obj.ene_nuc;
        err = (iter > 0) ? std::abs(e_cur - e_pre) : 1.0;
        is_converged = (err < conv_tol and iter > 0);
        is_max_iter  = (iter > max_iter);

        // format output
        if (iter > 0) {
            // Output the iteration number with padding
            std::cout << std::setw(4) << iter << " "
                      // Output the energy sum with fixed-point notation and precise format
                      << std::setw(19) << std::right << std::fixed << std::setprecision(8) << e_tot
                      // Output the energy change with scientific notation and precise format
                      << std::setw(20) << std::right << std::scientific << std::setprecision(4) << err
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
    auto ene_rhf = solve_rhf(filename);
    return 0;
}