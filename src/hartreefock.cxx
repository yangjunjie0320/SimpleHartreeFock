#include <iostream>
#include <cassert>
#include <armadillo>

class MoleculeInformation {
public:
    int nelec_alph;
    int nelec_beta;
    int nao, naux;

    arma::mat  s1e;
    arma::mat  h1e;
    arma::cube cderi;

    MoleculeInformation() = default;
    MoleculeInformation(std::string filename) {
        arma::ivec nelec; nelec.load(arma::hdf5_name(filename, "nelec"));
        nelec_alph = nelec(0);
        nelec_beta = nelec(1);

        s1e.load(arma::hdf5_name(filename, "ovlp"));
        h1e.load(arma::hdf5_name(filename, "hcore"));
        cderi.load(arma::hdf5_name(filename, "cderi"));

        nao = s1e.n_rows;
        naux = cderi.n_slices;
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
    auto nao  = mol_obj.nao;
    auto naux = mol_obj.naux;

    arma::mat vj = arma::zeros(nao, nao);
    arma::mat vk = arma::zeros(nao, nao);

    for (auto mu = 0; mu < nao; mu++) {
        for (auto nu = 0; nu < nao; nu++) {
            for (auto lm = 0; lm < nao; lm++) {
                for (auto sg = 0; sg < nao; sg++) {
                    for (auto x = 0; x < naux; x++) {
                        vj(mu, nu) += mol_obj.cderi(mu, nu, x) * mol_obj.cderi(lm, sg, x) * rdm1(lm, sg);
                        vk(mu, nu) += mol_obj.cderi(mu, sg, x) * mol_obj.cderi(lm, nu, x) * rdm1(lm, sg);
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

    int iter = 0, max_iter = 100;

    while (not is_converged and not is_max_iter) {
        arma::mat rdm1 = c_occ * c_occ.t();
        arma::mat fock = h1e + get_vjk(mol_obj, rdm1);

        e_cur = arma::accu(rdm1 % (h1e + fock));

        eigh(fock, s1e, c, e);
        c_occ = c.cols(0, nocc - 1);

        if (std::abs(e_cur - e_pre) < 1e-6) {
            is_converged = true;
        }

        if (iter > max_iter) {
            is_max_iter = true;
        }

        e_pre = e_cur; iter++;

        std::cout << "Iter: " << iter << " Energy: " << e_cur << std::endl;
    }

}

int main(int argc, char** argv) {
    std::string filename = "/Users/yangjunjie/work/SimpleHartreeFock/data/h2o/0.5000/data.h5";
    hartree_fock(filename);
    return 0;
}