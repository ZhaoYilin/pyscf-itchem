from pyscf import gto,scf,dft
from pyscf.ita.promolecule import ProMolecule
from pyscf.ita.ita import ITA

# constructe molecule
mol = gto.Mole()
mol.atom = """
H  0.000000000000   0.000000000000   0.000000000000
F  0.000000000000   0.000000000000   1.100000000000
"""
mol.basis = "ccpvdz"
mol.charge = 0
mol.multiplicity = 1
mol.unit = 'A'
mol.build()

# run mean field method
dft_mf = scf.HF(mol) # likewise for RKS, UKS, ROKS and GKS
dft_mf.run()

# build grids
grids = dft.Grids(mol)
grids.atom_grid = (75, 302)
grids.becke_scheme
grids.prune = True
grids.build()

# build promolecule
promol = ProMolecule(dft_mf)
promol.pro_charge = {'H':0, 'F':0}
promol.pro_multiplicity = {'H':2, 'F':2}
promol.build()

# build ita
ita = ITA(dft_mf, grids)
ita.promolecule = promol
ita.build() 

# batch compute
ita.batch_compute(ita_code=[11,12,13,14,15,16,17], representation='electron density', partition = 'hirshfeld', filename='./ita_ed.log')
