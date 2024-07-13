import numpy, scipy, h5py

# Use eigh to diagonalize matrices
from scipy.linalg import eigh

class PrimitiveGaussianTypeOrbital(object):
    center : numpy.ndarray
    exponent : float
    coefficient : float
    angular_momentum : numpy.ndarray

class ContractedGaussianTypeOrbital(object):
    primitives : list
    coefficient : float

class BasisSet(object):
    contracted_gto : list

def get_ovlp_integral(h5file : str = None, method : str = "read") -> numpy.ndarray:
    if method == "read":
        with h5py.File(h5file, "r") as f:
            ovlp = f["ovlp"][()]
        return ovlp
    elif method == "compute":
        raise NotImplementedError("This function is not implemented yet.")
    else:
        raise ValueError(f"Unknown method {method}.")

def solve_rhf(h5file : str = None, max_iter :int = 100, tol: float = 1e-8) -> float:
    # Load the data
    with h5py.File(h5file, "r") as f:
        ovlp    = f["ovlp"][()]
        hcore   = f["hcore"][()]
        cderi   = f["cderi"][()]
        eri     = f["eri"][()]
        nelecs  = f["nelec"][()][0]

        ene_nuc = f["ene_nuc"][()][0][0]
        ene_rhf_ref = f["ene_rhf"][()][0][0]

    nelec_alph, nelec_beta = nelecs
    assert nelec_alph == nelec_beta, "This code only supports closed-shell systems."

    nao = hcore.shape[0]
    naux = cderi.shape[0]
    err = abs(numpy.einsum("Qmn,Qkl->mnkl", cderi, cderi) - eri).max()
    assert err < 1e-8, f"ERI tensor is not consistent with the contracted ERI tensor. Error = {err}"

    assert hcore.shape == (nao, nao)
    assert ovlp.shape  == (nao, nao)
    assert cderi.shape == (naux, nao, nao)

    iter_scf     = 0
    is_converged = False
    is_max_iter  = False

    ene_err = 1.0
    dm_err  = 1.0

    ene_rhf = None
    ene_old = None
    ene_cur = None

    nmo  = nao
    nocc = (nelec_alph + nelec_beta) // 2
    mo_occ = numpy.zeros(nmo, dtype=int)
    mo_occ[:nocc] = 2
    occ_list = numpy.where(mo_occ > 0)[0]

    # Diagonalize the core Hamiltonian to get the initial guess for density matrix
    energy_mo, coeff_mo = eigh(hcore, ovlp)
    coeff_occ = coeff_mo[:, occ_list]
    dm_old    = numpy.dot(coeff_occ, coeff_occ.T) * 2.0
    dm_cur    = None
    fock      = None

    while not is_converged and not is_max_iter:
        # Compute the Fock matrix
        # coul =   numpy.einsum("Qpq,Qrs,rs->pq", cderi, cderi, dm_old)
        # exch = - numpy.einsum("Qpr,Qqs,rs->pq", cderi, cderi, dm_old) * 0.5
        coul = numpy.einsum("pqrs,rs->pq", eri, dm_old)
        exch = numpy.einsum("prqs,rs->pq", eri, dm_old) * 0.5
        fock = hcore + coul - exch

        # Diagonalize the Fock matrix
        energy_mo, coeff_mo = eigh(fock, ovlp)

        # Compute the new density matrix
        coeff_occ = coeff_mo[:, occ_list]
        dm_cur    = numpy.dot(coeff_occ, coeff_occ.T) * 2.0

        # Compute the energy
        ene_cur = 0.5 * numpy.einsum("pq,pq->", hcore + fock, dm_cur)
        ene_rhf = ene_cur + ene_nuc

        # Compute the errors
        if ene_old is not None:
            dm_err  = numpy.linalg.norm(dm_cur - dm_old)
            ene_err = abs(ene_cur - ene_old)
            print(f"SCF iteration {iter_scf:3d}, energy = {ene_rhf: 12.8f}, error = {ene_err: 6.4e}, {dm_err: 6.4e}")

        dm_old  = dm_cur
        ene_old = ene_cur

        # Check convergence
        iter_scf += 1
        is_max_iter  = iter_scf >= max_iter
        is_converged = ene_err < tol and dm_err < tol

    if is_converged:
        print(f"SCF converged in {iter_scf} iterations.")
    else:
        if ene_rhf is not None:
            print(f"SCF did not converge in {max_iter} iterations.")
        else:
            ene_rhf = 0.0
            print("SCF is not running.")

    assert abs(ene_rhf - ene_rhf_ref) < 1e-8, f"RHF energy {ene_rhf} is not consistent with the reference {ene_rhf_ref}."
    return ene_rhf

if __name__ == "__main__":
    h5file = "/Users/yangjunjie/work/SimpleHartreeFock/data/h2o/0.5000/data.h5"
    ene_rhf = solve_rhf(h5file, max_iter=100, tol=1e-8)
    print(f"RHF energy = {ene_rhf: 12.8f}")
