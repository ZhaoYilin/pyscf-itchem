import numpy as np

from pyscf import scf, dft
from pyscf.scf import atom_hf, atom_ks
from pyscf.dft.rks import KohnShamDFT
from pyscf.scf.hf import SCF        
from pyscf.data.elements import NRSRHF_CONFIGURATION

__all__ = ["ProMolecule"]

MULTIPLICITIES = {
    "H": 2, "He":1, 
    "Li":2, "Be":1, "B": 2, "C": 3, "N": 4, "O": 3, "F": 2, "Ne":1, 
    "Na":2, "Mg":1, "Al":2, "Si":3, "P": 4, "S": 3, "Cl":2, "Ar":1, 
    "K": 2, "Ca":1, "Sc":2, "Ti":3, "V": 4, "Cr":7, "Mn":6, "Fe":5, 
    "Co":4, "Ni":3, "Cu":2, "Zn":1, "Ga":2, "Ge":3, "As":4, "Se":3, 
    "Br":2, "Kr":1, "Rb":2, "Sr":1, "Y": 2, "Zr":3, "Nb":6, "Mo":7, 
    "Tc":6, "Ru":5, "Rh":4, "Pd":1, "Ag":2, "Cd":1, "In":2, "Sn":3, 
    "Sb":4, "Te":3, "I" :2, "Xe":1, "Cs":2, "Ba":1, "La":2, "Ce":1, 
    "Pr":4, "Nd":5, "Pm":6, "Sm":7, "Eu":8, "Gd":9, "Tb":6, "Dy":5,
    "Ho":4, "Er":3, "Tm":2, "Yb":1, "Lu":2, "Hf":3, "Ta":4, "W" :5,
    "Re":6, "Os":5, "Ir":4, "Pt":3, "Au":2, "Hg":1, "Tl":2, "Pb":3, 
    "Bi":4, "Po":3, "At":2, "Rn":1, "Fr":2, "Ra":1, "Ac":2, "Th":3, 
    "Pa":4, "U" :5, "Np":6, "Pu":7, "Am":8, "Cm":9, "Bk":6, "Cf":5, 
    "Es":4, "Fm":3
}
r"""Dictionary of the multiplicities for each isoelectronic series (up to 100 electrons).
"""

class ProMolecule:
    r""" Promolecule class.

    A promolecule is an approximation of a molecule constructed from a linear combination of atomic
    and/or ionic species. Properties of this promolecule can be computed from those of the atomic
    and/or ionic species, depending on whether the property is an extensive one or an intensive one.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> mf = scf.HF(mol)
    >>> mf.kernel()
    >>> pro_charge, pro_multiplicity = {'H':0, 'O':0}, {'H':2, 'O':3}
    >>> promol = ProMolecule.build(mf, pro_charge, pro_multiplicity)    
    """    
    def __init__(
        self, 
        method, 
        element_methods=None, 
        pro_charge=None, 
        pro_multiplicity=None
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf method.
        element_methods : Dict{element:PyscfMethod}, optional
            Dictionary of PyscfMethod for each element, by default None.
        pro_charge : Dict{str:int}, optional
            Dictionary of charge for each element, by default None.
        pro_multiplicity : Dict{str:int}, optional
            Dictionary of multiplicity for each element, by default None.
        """
        self.method = method        
        self.element_methods = element_methods
        self.pro_charge = pro_charge
        self.pro_multiplicity = pro_multiplicity

    def build(
        self, 
        method=None, 
        pro_charge=None, 
        pro_multiplicity=None
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf method.
        pro_charge : Dict{str:int}, optional
            Dictionary of charge for each element, by default None.
        pro_multiplicity : Dict{str:int}, optional
            Dictionary of multiplicity for each element, by default None.

        Returns
        -------
        ProMolecule 
            Instance of ProMolecule class.
        """
        if method is None:
            method = self.method
        if pro_charge is None:
            pro_charge = self.pro_charge
        if pro_multiplicity is None:
            pro_multiplicity = self.pro_multiplicity

        # By default use neutral atoms.
        if (pro_charge is None) and (pro_multiplicity is None):
            pro_charge, pro_multiplicity = {}, {}
            for element in set(method.mol.elements):
                pro_charge[element] = 0
                pro_multiplicity[element] = MULTIPLICITIES[element]

        # Parse the method
        atom_method = self.identical_method(method)  

        # Build atom in promolecule
        mol = method.mol
        element_methods = {}
        for element in set(mol.elements):
            atom = mol.__class__()
            atom.atom = [[element, 0.0, 0.0, 0.0]]
            atom.basis = mol.basis
            atom.charge = pro_charge[element]
            atom.multiplicity = pro_multiplicity[element]
            atom.unit = mol.unit
            atom.cart = mol.cart
            atom.verbose = 0
            atom.build()       

            if isinstance(method, SCF):
                element_method = atom_method(atom)
                element_method.kernel()
            else:
                element_mf = scf.HF(atom) 
                element_mf.kernel()
                element_method = atom_method(element_mf)
                element_method.kernel()
            element_methods[element] = element_method

        self.method = method
        self.element_methods = element_methods
        self.pro_charge = pro_charge
        self.pro_multiplicity = pro_multiplicity
        return self

    def identical_method(self, method):
        """_summary_

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.

        Returns
        -------
        _type_
            _description_
        """        
        # If Mean field method 
        if isinstance(method, SCF):
            if isinstance(method, KohnShamDFT):
                method_func = dft.KS
                method_func.xc = method.xc
            else:
                method_func = scf.HF

        # If post-scf method
        else:
            if 'MP'.lower() in method.__class__.__name__.lower():
                from pyscf import mp
                method_func = getattr(mp, method.__class__.__name__)              
            elif 'CC'.lower() in method.__class__.__name__.lower():
                from pyscf import cc
                method_func = getattr(cc, method.__class__.__name__)              
            elif 'CI'.lower() in method.__class__.__name__.lower():
                from pyscf import ci  
                method_func = getattr(ci, method.__class__.__name__)              
            else:
                raise TypeError("Method type not supported.")  
            
        return method_func   

    def spheric_average_method(self, method):
        """_summary_

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.

        Returns
        -------
        _type_
            _description_
        """        
        # If Mean field method 
        if isinstance(method, SCF):
            if isinstance(method, KohnShamDFT):
                method_func = atom_ks.AtomSphAverageRKS
                method_func.xc = method.xc
            else:
                if method.mol.nelectron == 1:
                    method_func = atom_hf.AtomHF1e
                else:
                    method_func = atom_hf.AtomSphAverageRHF
            method_func.atomic_configuration = NRSRHF_CONFIGURATION

        # If post-scf method
        else:
            if 'MP'.lower() in method.__class__.__name__.lower():
                from pyscf import mp
                method_func = getattr(mp, method.__class__.__name__)              
            elif 'CC'.lower() in method.__class__.__name__.lower():
                from pyscf import cc
                method_func = getattr(cc, method.__class__.__name__)              
            elif 'CI'.lower() in method.__class__.__name__.lower():
                from pyscf import ci  
                method_func = getattr(ci, method.__class__.__name__)              
            else:
                raise TypeError("Method type not supported.")  
            
        return method_func   

    def electron_density(
        self, 
        grids_coords, 
        spin='ab', 
        deriv=2
    ):
        """ Compute the electron density of the promolecule at the desired points.

        Parameters
        ----------
        grid_coords: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm')
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density,
            by default 'ab'.            
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.

        Returns
        -------
        PartitionDensity
            Instance of PartitionDensity class. 
        """
        from pyscf.ita.dens import ElectronDensity, PartitionDensity
        element_methods = self.element_methods

        proatom_dens_list = []
        for atom_geom in self.method.mol._atom:
            symbol = atom_geom[0]
            element_methods[symbol].mol.set_geom_([atom_geom],unit='B')  
            free_atom_dens = ElectronDensity.build(element_methods[symbol], grids_coords=grids_coords, spin=spin, deriv=deriv)
            proatom_dens_list.append(free_atom_dens)

        promoldens = PartitionDensity(proatom_dens_list)
        return promoldens

    def get_method(self, element):
        """Get the Pyscf method for given element.

        Parameters
        ----------
        element : str
            Symbol of element.

        Returns
        -------
        PyscfMethod
            Pyscf method.
        """        
        return self.element_methods[element]
    
    def get_atom(self, element):
        """Get the Pyscf Mole object for given element.

        Parameters
        ----------
        element : str
            Symbol of element.

        Returns
        -------
        Mole
            Pyscf Mole object for free atom.
        """        
        return self.element_methods[element].mol
    
    def get_xc(self, element):
        """Get the exchange correlation functional for given element.

        Parameters
        ----------
        element : str
            Symbol of element.

        Returns
        -------
        str
            Name of exchange correlation functional.
        """        
        return self.element_methods[element].xc   

    def get_scf(self, element):
        """Get the mean field Pyscf method for given element.

        Parameters
        ----------
        element : str
            Symbol of element.

        Returns
        -------
        PyscfMethod
            Mean field pyscf method.
        """        
        return self.element_methods[element]._scf  