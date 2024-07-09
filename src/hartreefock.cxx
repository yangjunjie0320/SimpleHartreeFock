#include <iostream>

#define ARMA_USE_HDF5
#include <armadillo>

arma::mat get_vjk(const arma::cube& cderi, const arma::mat& rdm1) {
    auto nao  = rdm1.n_rows;
    auto naux = cderi.n_slices;

    arma::mat vj = arma::zeros(nao, nao);
    arma::mat vk = arma::zeros(nao, nao);

    for (auto mu = 0; mu < nao; mu++) {
        for (auto nu = 0; nu < nao; nu++) {
            for (auto lm = 0; lm < nao; lm++) {
                for (auto sg = 0; sg < nao; sg++) {
                    for (auto x = 0; x < naux; x++) {
                        vj(mu, nu) += cderi(mu, nu, x) * cderi(lm, sg, x) * rdm1(lm, sg);
                        vk(mu, nu) += cderi(mu, sg, x) * cderi(lm, nu, x) * rdm1(lm, sg);
                    }
                }
            }
        }
    }

    return vj - 0.5 * vk;
}

int hartree_fock(std::string filename) {
  std::cout << "filename: " << filename << std::endl;

  arma::mat atm;
  atm.load(arma::hdf5_name(filename, "atm"));
  atm.print("atm");

  arma::mat bas;
  bas.load(arma::hdf5_name(filename, "bas"));
  bas.print("bas");

  arma::mat env;
  env.load(arma::hdf5_name(filename, "env"));
  env.print("env");
  
  arma::mat s1e;
  s1e.load(arma::hdf5_name(filename, "ovlp"));

  arma::mat h1e;
  h1e.load(arma::hdf5_name(filename, "hcore"));

  arma::cube cderi;
  cderi.load(arma::hdf5_name(filename, "cderi"));

  arma::mat fock = h1e;

  return 1;
}

int main(int argc, char** argv) {
  hartree_fock("/Users/yangjunjie/work/cxx-test-project/data/h2o/0.5000/data.h5");
  return 0;
}

