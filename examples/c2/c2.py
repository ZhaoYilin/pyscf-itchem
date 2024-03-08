from pyscf import gto,scf,dft,fci
from pyscf.ita.promolecule import ProMolecule
from pyscf.ita.ita import ITA

# constructe molecule
mol = gto.Mole()
mol.atom = """
C   0.000000000000  0.000000000000   0.000000000000
C   0.000000000000  0.000000000000   1.100000000000
"""
mol.basis = "sto-3g"
mol.charge = 0
mol.multiplicity = 1
mol.unit = 'A'
mol.build()

# run mean field method
hf_mf = scf.HF(mol)
hf_mf.run()

# run post hf method
#
# create an FCI solver based on the SCF object
#
fcisolver = fci.FCI(hf_mf)
print('E(FCI) = %.12f' % fcisolver.kernel()[0])


# build grids
grids = dft.Grids(mol)
grids.atom_grid = (75, 302)
grids.becke_scheme
grids.prune = True
grids.build()

# build promolecule
promol = ProMolecule(fcisolver)
promol.pro_charge = {'C':0}
promol.pro_multiplicity = {'C':3}
promol.build()

# build ita
ita = ITA(fcisolver, grids)
ita.promolecule = promol
ita.build() 

# batch compute
ita.batch_compute(ita_code=[11,12,13,14,15,16,17], representation='electron density', partition = 'hirshfeld', filename='./ita_ed.log')
