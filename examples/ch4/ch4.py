from pyscf import gto,dft
from pyscf.ita.promolecule import ProMolecule
from pyscf.ita.kernel import ITA

# constructe molecule
mol = gto.Mole()
mol.atom = """
C  -0.000000000000   0.000000000000   0.000000000000
H   1.183771681898  -1.183771681898  -1.183771681898
H   1.183771681898   1.183771681898   1.183771681898
H  -1.183771681898   1.183771681898  -1.183771681898
H  -1.183771681898  -1.183771681898   1.183771681898
"""
mol.basis = "6-31G"
mol.charge = 0
mol.multiplicity = 1
mol.unit = 'A'
mol.build()

# run mean field method
dft_mf = dft.KS(mol) # likewise for RKS, UKS, ROKS and GKS
dft_mf.xc = 'm062x'
dft_mf.run()

# build grids
grids = dft.Grids(mol)
grids.atom_grid = (75, 302)
grids.becke_scheme
grids.prune = True
grids.build()

# build promolecule
promol = ProMolecule(dft_mf)
promol.pro_charge = {'H':0, 'C':0}
promol.pro_multiplicity = {'H':2, 'C':3}
promol.build()

# build ita
ita = ITA(dft_mf, grids)
ita.promolecule = promol
ita.build() 

# batch compute
ita.batch_compute(ita_code=[11,12,13,14,15,16,17], representation='electron density', partition = 'hirshfeld', filename='./ita_ed.log')
