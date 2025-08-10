import numpy as np

from pyscf import scf, dft
from pyscf.scf import atom_hf, atom_ks
from pyscf.dft.rks import KohnShamDFT
from pyscf.scf.hf import SCF        
from pyscf.data.elements import NRSRHF_CONFIGURATION

from pyscf.itchem.utils.constants import MULTIPLICITIES

__all__ = ["ProMolecule"]

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
        """Build proatoms by identical method with given molecule.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
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
                if 'fci' in str(method):
                    from pyscf import fci
                    method_func = getattr(fci, 'FCI')    
                else:          
                    from pyscf import ci  
                    method_func = getattr(ci, method.__class__.__name__)              
            else:
                raise TypeError("Method type not supported.")  
            
        return method_func   

    def spheric_average_method(self, method):
        """Build proatoms by spheric average method.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
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

    def one_electron_density(
        self, 
        grids_coords, 
        deriv=2
    ):
        """ Compute the electron density of the promolecule at the desired points.

        Parameters
        ----------
        grids_coords : np.ndarray((Ngrids,3), dtype=float)
            Grids coordinates on N points.         
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
        from pyscf.itchem.dens import OneElectronDensity, PartitionDensity
        element_methods = self.element_methods

        proatom_dens_list = []
        for atom_geom in self.method.mol._atom:
            symbol = atom_geom[0]
            element_methods[symbol].mol.set_geom_([atom_geom],unit='B')  
            free_atom_dens = OneElectronDensity.build(element_methods[symbol], grids_coords=grids_coords, deriv=deriv)
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