#include <iostream>
#include <armadillo>

int hartree_fock(std::string filename) {
  std::cout << "filename: " << filename << std::endl;

  arma::mat ovlp;
  ovlp.load(arma::hdf5_name(filename, "ovlp"));

  std::cout << "ovlp.n_rows: " << ovlp.n_rows << std::endl;
  std::cout << "ovlp.n_cols: " << ovlp.n_cols << std::endl;
  ovlp.print("ovlp:");

  return 1;
}

int main(int argc, char** argv) {
  hartree_fock("/Users/yangjunjie/work/cxx-test-project/data/h2/0.5000/data.h5");
  return 0;
}

