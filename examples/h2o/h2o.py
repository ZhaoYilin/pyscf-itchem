from pyscf import gto,dft
from pyscf.ita.promolecule import ProMolecule
from pyscf.ita.kernel import ITA

# constructe molecule
mol = gto.Mole()
mol.atom = """
O   0.000000000000  -0.143225816552   0.000000000000
H   1.638036840407   1.136548822547  -0.000000000000
H  -1.638036840407   1.136548822547  -0.000000000000
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
promol.pro_charge = {'H':0, 'O':0}
promol.pro_multiplicity = {'H':2, 'O':3}
promol.build()

# build ita
ita = ITA(dft_mf, grids)
ita.category = 'regular'
ita.representation='electron density'
ita.partition = 'hirshfeld'
ita.promolecule = promol
ita.build() 

# batch compute
ita.batch_compute(ita_code=[1,2,3,4,5,6,7],  filename='./ita_ed.log')