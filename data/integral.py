import numpy, scipy, h5py

# Use eigh to diagonalize matrices
from scipy.linalg import eigh
from typing import List, Dict, Tuple

class PrimitiveGaussianTypeOrbital(object):
    center : numpy.ndarray
    exponent : float
    angular_momentum : Tuple[int, int, int]

class ContractedGaussianTypeOrbital(object):
    center : numpy.ndarray
    primitives : list
    coefficients : list
    angular_momentum : Tuple[int, int, int]

    def get_pgto(self):
        npgto = len(self.primitives)

        assert len(self.primitives) == npgto
        assert len(self.coefficients) == npgto

        pgtos = []
        for ipgto in range(npgto):
            pgto = PrimitiveGaussianTypeOrbital()
            pgto.center = self.center
            pgto.exponent = self.primitives[ipgto]
            pgto.angular_momentum = self.angular_momentum
            pgtos.append(pgto)
        return pgtos

class Shell(object):
    atom_index : int
    center : numpy.ndarray
    angular_momentum : int
    number_of_pgto : int
    number_of_cgto : int
    
    exponents : numpy.ndarray
    coefficients : numpy.ndarray

    def get_cgto(self):
        ang = self.angular_momentum
        cgtos = []
        for xx in range(ang + 1):
            for yy in range(ang + 1 - xx):
                zz = ang - xx - yy
                
                cgto = ContractedGaussianTypeOrbital()
                cgto.primitives = numpy.array(self.exponents)
                cgto.coefficients = numpy.array(self.coefficients)
                cgto.angular_momentum = (xx, yy, zz)
                cgto.center = self.center
                cgtos.append(cgto)
        return cgtos

def read_basis_set(h5file : str = None):
    from pyscf import gto, lib
    atm = lib.chkfile.load(h5file, "atm")
    bas = lib.chkfile.load(h5file, "bas")
    env = lib.chkfile.load(h5file, "env")

    # I guess the basis is in fact shell in other context
    nbas = len(bas)
    natm = len(atm)

    ATOM_OF         = 0
    ANG_OF          = 1
    NPRIM_OF        = 2
    NCTR_OF         = 3
    KAPPA_OF        = 4
    PTR_EXP         = 5
    PTR_COEFF       = 6
    RESERVE_BASLOT  = 7
    BAS_SLOTS       = 8

    CHARGE_OF       = 0
    PTR_COORD       = 1
    NUC_MOD_OF      = 2
    PTR_ZETA        = 3
    PTR_FRAC_CHARGE = 4
    RESERVE_ATMSLOT = 5
    ATM_SLOTS       = 6

    shells = []

    for ibas in range(nbas):
        shell = Shell()
        shell.atom_index = bas[ibas][ATOM_OF]
        shell.center = env[atm[shell.atom_index][PTR_COORD]:atm[shell.atom_index][PTR_COORD]+3]
        shell.angular_momentum = bas[ibas][ANG_OF]
        shell.number_of_pgto = bas[ibas][NPRIM_OF]
        shell.number_of_cgto = bas[ibas][NCTR_OF]

        npgto = shell.number_of_pgto
        ptr_exp = bas[ibas][PTR_EXP]
        shell.exponents = env[ptr_exp:ptr_exp+npgto]

        ptr_coeff = bas[ibas][PTR_COEFF]
        shell.coefficients = env[ptr_coeff:ptr_coeff+npgto]

        shells.append(shell)

    cgtos = []
    for ishl, shl in enumerate(shells):
        cgtos += shl.get_cgto()

    pgtos = []
    for cgto in cgtos:
        pgtos += cgto.get_pgto()

    for ipgto in range(len(pgtos)):
        print(f"\nPGTO {ipgto}")
        print(f"Center = {pgtos[ipgto].center}")
        print(f"Exponent = {pgtos[ipgto].exponent}")
        print(f"Angular momentum = {pgtos[ipgto].angular_momentum}")
        print()

    return pgtos

def _ovlp_for_pgto_ref(pgto1 : PrimitiveGaussianTypeOrbital, pgto2 : PrimitiveGaussianTypeOrbital) -> float:
    from pyscf import gto
    from pyscf.gto import mole
    from pyscf.gto import moleintor

    # bas_info = []
    # bas_info.append([(, ((pgto1.exponent, 1.0), ))])
    # bas_info.append([(numpy.sum(pgto2.angular_momentum), ((pgto2.exponent, 1.0), ))])\
    print()

    mol = gto.Mole()
    mol.atom = [["H1", pgto1.center], ["H2", pgto2.center]]
    mol.basis = {
        "H1": [[0, [0.183192, 1.0]]],
        "H2": [[0, [0.183192, 1.0]]]
    }
    print(mol.basis)
    mol.build()\
    
    mol = gto.Mole()
    mol.atom = [["H1", pgto1.center], ["H2", pgto2.center]]
    mol.basis = {
        "H1": [[0, [pgto1.exponent, 1.0]]],
        "H2": [[0, [pgto2.exponent, 1.0]]]  
    }
    print(mol.basis)
    mol.build()

    ovlp = mol.intor("int1e_ovlp")
    print(ovlp)
    assert ovlp.shape == (2, 2)
    return ovlp[0, 1]

def _ovlp_for_pgto_sol(pgto1 : PrimitiveGaussianTypeOrbital, pgto2 : PrimitiveGaussianTypeOrbital) -> float:
    s = _ovlp(
        pgto1.center, pgto1.exponent, pgto1.angular_momentum,
        pgto2.center, pgto2.exponent, pgto2.angular_momentum
    )
    return s

def _ovlp(cen1 : numpy.ndarray, exp1 : float, ang1 : Tuple[int, int, int], 
          cen2 : numpy.ndarray, exp2 : float, ang2 : Tuple[int, int, int]) -> float:
    cx1, cy1, cz1 = cen1
    cx2, cy2, cz2 = cen2
    ax1, ay1, az1 = ang1
    ax2, ay2, az2 = ang2

    is_zero = ax1 < 0 or ay1 < 0 or az1 < 0
    is_zero = is_zero or ax2 < 0 or ay2 < 0 or az2 < 0
    if is_zero:
        return 0.0

    if ax1 > 0:
        tmp1 = _ovlp(cen1, exp1, (ax1-1, ay1, az1), cen2, exp2, ang2)
        tmp1 *= (exp1 * cx1 + exp2 * cx2) / (exp1 + exp2) - cx1

        tmp2 = _ovlp(cen1, exp1, (ax1-2, ay1, az1), cen2, exp2, ang2)
        tmp2 *= (ax1 - 1) / 2.0 / (exp1 + exp2)

        tmp3 = _ovlp(cen1, exp1, (ax1-1, ay1, az1), cen2, exp2, (ax2-1, ay2, az2))
        tmp3 *= ax2 / 2.0 / (exp1 + exp2)

        return tmp1 + tmp2 + tmp3
    
    elif ay1 > 0:
        tmp1 = _ovlp(cen1, exp1, (ax1, ay1-1, az1), cen2, exp2, ang2)
        tmp1 *= (exp1 * cy1 + exp2 * cy2) / (exp1 + exp2) - cy1

        tmp2 = _ovlp(cen1, exp1, (ax1, ay1-2, az1), cen2, exp2, ang2)
        tmp2 *= (ay1 - 1) / 2.0 / (exp1 + exp2)

        tmp3 = _ovlp(cen1, exp1, (ax1, ay1-1, az1), cen2, exp2, (ax2, ay2-1, az2))
        tmp3 *= ay2 / 2.0 / (exp1 + exp2)

        return tmp1 + tmp2 + tmp3
    
    elif az1 > 0:
        tmp1 = _ovlp(cen1, exp1, (ax1, ay1, az1-1), cen2, exp2, ang2)
        tmp1 *= (exp1 * cz1 + exp2 * cz2) / (exp1 + exp2) - cz1

        tmp2 = _ovlp(cen1, exp1, (ax1, ay1, az1-2), cen2, exp2, ang2)
        tmp2 *= (az1 - 1) / 2.0 / (exp1 + exp2)

        tmp3 = _ovlp(cen1, exp1, (ax1, ay1, az1-1), cen2, exp2, (ax2, ay2, az2-1))
        tmp3 *= az2 / 2.0 / (exp1 + exp2)

        return tmp1 + tmp2 + tmp3
    
    elif ax2 > 0:
        tmp1 = _ovlp(cen1, exp1, ang1, cen2, exp2, (ax2-1, ay2, az2))
        tmp1 *= (exp1 * cx1 + exp2 * cx2) / (exp1 + exp2) - cx2

        tmp2 = _ovlp(cen1, exp1, ang1, cen2, exp2, (ax2-2, ay2, az2))
        tmp2 *= (ax2 - 1) / 2.0 / (exp1 + exp2)

        tmp3 = _ovlp(cen1, exp1, (ax1-1, ay1, az1), cen2, exp2, (ax2-1, ay2, az2))
        tmp3 *= ax1 / 2.0 / (exp1 + exp2)

        return tmp1 + tmp2 + tmp3
    
    elif ay2 > 0:
        tmp1 = _ovlp(cen1, exp1, ang1, cen2, exp2, (ax2, ay2-1, az2))
        tmp1 *= (exp1 * cy1 + exp2 * cy2) / (exp1 + exp2) - cy2

        tmp2 = _ovlp(cen1, exp1, ang1, cen2, exp2, (ax2, ay2-2, az2))
        tmp2 *= (ay2 - 1) / 2.0 / (exp1 + exp2)

        tmp3 = _ovlp(cen1, exp1, (ax1, ay1-1, az1), cen2, exp2, (ax2, ay2-1, az2))
        tmp3 *= ay1 / 2.0 / (exp1 + exp2)

        return tmp1 + tmp2 + tmp3
    
    elif az2 > 0:
        tmp1 = _ovlp(cen1, exp1, ang1, cen2, exp2, (ax2, ay2, az2-1))
        tmp1 *= (exp1 * cz1 + exp2 * cz2) / (exp1 + exp2) - cz2

        tmp2 = _ovlp(cen1, exp1, ang1, cen2, exp2, (ax2, ay2, az2-2))
        tmp2 *= (az2 - 1) / 2.0 / (exp1 + exp2)

        tmp3 = _ovlp(cen1, exp1, (ax1, ay1, az1-1), cen2, exp2, (ax2, ay2, az2-1))
        tmp3 *= az1 / 2.0 / (exp1 + exp2)

        return tmp1 + tmp2 + tmp3
    
    else:
        assert ax1 == 0 and ay1 == 0 and az1 == 0
        assert ax2 == 0 and ay2 == 0 and az2 == 0

        tmp = numpy.sqrt(numpy.pi / (exp1 + exp2)) * numpy.pi / (exp1 + exp2)
        tmp *= numpy.exp(-exp1 * exp2 / (exp1 + exp2) * numpy.linalg.norm(cen1 - cen2) ** 2)
        return tmp

def get_ovlp_integral(h5file : str = None, method : str = "read") -> numpy.ndarray:
    if method == "read":
        with h5py.File(h5file, "r") as f:
            ovlp = f["ovlp"][()]
        return ovlp
    elif method == "compute":
        pgtos = read_basis_set(h5file)
        npgtop = len(pgtos)
        ovlp_pgto = numpy.zeros((npgtop, npgtop))
        for ii in range(npgtop):
            for jj in range(npgtop):
                s1 = _ovlp_for_pgto_sol(pgtos[ii], pgtos[jj])
                s2 = _ovlp_for_pgto_ref(pgtos[ii], pgtos[jj])

                print(f"ii = {ii}, jj = {jj}, s1 = {s1}, s2 = {s2}")
                err = abs(s1 - s2)
                print(f"Error = {err}")
    else:
        raise ValueError(f"Unknown method {method}.")

if __name__ == "__main__":
    h5file = "/Users/yangjunjie/work/SimpleHartreeFock/data/h2o/0.5000/data.h5"
    ovlp_ref = get_ovlp_integral(h5file, method="read")
    ovlp_sol = get_ovlp_integral(h5file, method="compute")

    err = abs(ovlp_ref - ovlp_sol).max()
    print(f"Error = {err}")
