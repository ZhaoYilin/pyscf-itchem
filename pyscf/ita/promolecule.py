import numpy as np
from pyscf import scf, dft

class ProMolecule:
    def __init__(self, promolmethods):
        """_summary_

        Parameters
        ----------
        promolmethods : _type_
            _description_
        """        
        self.promolmethods = promolmethods

    @classmethod
    def build(cls, method, charge, multiplicity):
        """_summary_

        Parameters
        ----------
        charges : _type_
            _description_
        mults : _type_
            _description_
        """
        from pyscf.dft.rks import KohnShamDFT
        from pyscf.scf.hf import SCF

        if isinstance(method, SCF):
            if isinstance(method, KohnShamDFT):
                promethod = dft.KS
                promethod.xc = method.xc
            else:
                promethod = scf.HF
        else:
            if 'MP'.lower() in method.__class__.__name__.lower():
                from pyscf import mp
                promethod = getattr(mp, method.__class__.__name__)              
            elif 'CC'.lower() in method.__class__.__name__.lower():
                from pyscf import cc
                promethod = getattr(cc, method.__class__.__name__)              
            elif 'CI'.lower() in method.__class__.__name__.lower():
                from pyscf import ci  
                promethod = getattr(ci, method.__class__.__name__)              
            else:
                raise TypeError("Type not supported.")     


        mol = method.mol

        promethods = {}
        for element in set(mol.elements):
            proatom = mol.__class__()
            proatom.atom = [[element, 0.0, 0.0, 0.0]]
            proatom.basis = mol.basis
            proatom.charge = charge[element]
            proatom.multiplicity = multiplicity[element]
            proatom.unit = mol.unit
            proatom.verbose = 0
            proatom.build()       

            if isinstance(method, SCF):
                atom_method = promethod(proatom)
                atom_method.kernel()
            else:
                proscf = scf.HF(proatom) 
                proscf.kernel()
                atom_method = promethod(proscf)
                atom_method.kernel()
            promethods[element] = atom_method

        return promethods

    @property
    def mol(self):
        pass

    @property
    def _scf(self):
        pass