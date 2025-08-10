from pyscf import gto, scf, dft
from pyscf.itchem.promolecule import ProMolecule
from pyscf.itchem.kernel import ITA

filename = 'h2o2.xyz'

# Mole section
mol = gto.Mole()
mol.verbose = 4
mol.output = filename.replace('xyz','out')
mol.atom = filename
mol.basis = '6-31G(d)'
mol.symmetry = 1
mol.build()

# Mean field section
mymf = scf.KS(mol)
mymf.xc = 'B3LYP'
mymf.level_shift = .4
mymf.max_cycle = 100
mymf.conv_tol = 1e-9
mymf.run()

# build grids
grids = dft.Grids(mol)
grids.atom_grid = (99, 590)
grids.becke_scheme
grids.prune = True
grids.build()

# build promolecule
promol = ProMolecule(mymf)
promol.pro_charge = {'H':0,'O':0}
promol.pro_multiplicity = {'H':2,'O':3}
promol.build()

# build ita
ita = ITA(mymf, grids)
ita.rung = [3,3]
ita.promolecule = promol
ita.build() 

# batch compute
ita.batch_compute(ita_code=[11,12,16,17,21,22,26], representation='electron density', partition = 'hirshfeld', filename='ed'+filename.replace('xyz','ita'))
ita.batch_compute(ita_code=[11,12,28,29,30], representation='atoms in molecules', partition = 'hirshfeld', filename='aim'+filename.replace('xyz','ita'))