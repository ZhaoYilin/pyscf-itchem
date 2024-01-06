import numpy as np

from pyscf.ita.promolecule import ProMolecule

__all__ = ["Becke", "Hirshfeld"]


class AIM:
    r"""Atoms-in-Molecules (AIM) representation of the information-theoretic approach (ITA) 
    from the perspective of atoms in molecules. To consider atomic contributions of an 
    information-theoretic quantity in a molecular system, three approaches are available 
    to perform atom partitionsin molecules. They are Becke's fuzzy atom approach, Bader's 
    zero-flux AIM approach, and Hirshfeld's stockholder approach. 
    """  
    def __init__(self):
        raise NotImplemented  
    
    def sharing_function(self):
        raise NotImplemented
    

class Becke(AIM):
    r"""Becke's fuzzy atom approach.
    """
    def __init__(
        self, 
        mol=None, 
        grids=None,
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        mol : Mole, optional
            Pyscf Mole instance, by default None.
        grids : Grid, optional
            Pyscf Grids instance, by default None.
        """
        self.mol = mol
        self.grids = grids
    
    def sharing_function(
        self, 
        mol=None, 
        grids=None,
    ):
        r"""Becke sharing funcition :math:`\omega_A(\mathbf{r})` defined as:

        .. math::
            \omega_A(\mathbf{r}) = \frac{V_A (\mathbf{r}) }{\sum_A V_A (\mathbf{r})} 

        Parameters
        ----------
        mol : Mole, optional
            Pyscf Mole instance, by default None.
        grids : Grid, optional
            Pyscf Grids instance, by default None.

        Returns
        -------
        omega : List[np.ndarray((N,), dtype=float)]
            Sharing functions for a list of atoms.            
        """        
        if mol==None:
            mol = self.mol
        if grids==None:
            grids = self.grids

        n_nuc = len(mol.elements)
        omega = []
        _, grids_weights = grids.get_partition(mol)
        grids_weights = np.ma.masked_less(grids_weights, 1e-30)
        grids_weights.filled(1e-30)
        for i, grids_weights_i in enumerate(np.split(grids_weights,n_nuc)):
            omega_i = [np.zeros_like(grids_weights_i)]*n_nuc
            omega_i[i] = grids_weights_i
            omega_i = np.concatenate(omega_i,axis=0)  
            omega_i = omega_i/grids_weights         
            omega.append(omega_i)

        return omega

class Hirshfeld(AIM):
    r"""Hirshfeld's stockholder approach.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> mf = scf.HF(mol)
    >>> mf.kernel()
    >>> grids = dft.Grids(mol)
    >>> grids.build()
    >>> pro_charge, pro_multiplicity = {'H':0, 'O':0}, {'H':2, 'O':3}
    >>> aim = Hirshfeld.build(mf, grids.coords, pro_charge, pro_multiplicity)   
    >>> omega = aim.sharing_function()  
    """
    def __init__(self, promoldens=None):
        r""" Initialize a instance.

        Parameters
        ----------        
        promoldens : ElectronDensity, optional
            ElectronDensity instance for molecule, by default None.
        """
        self.promoldens = promoldens
    
    @classmethod
    def build(
        cls, 
        method, 
        grids_coords, 
        pro_charge=None, 
        pro_multiplicity=None, 
        spin='ab', 
        deriv=0
        ):    
        """Class method to build the class.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
        grids_coords : np.ndarray((N,), dtype=float)
            Grids coords of N points.
        pro_charge : Dict{str:int}, optional
            Dictionary of charge for each element, by default None.
        pro_multiplicity : Dict{str:int}, optional
            Dictionary of multiplicity for each element, by default None.            
        spin: ('ab' | 'a' | 'b' | 'm')
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density,
            by default 'ab'.            
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 0.

        Returns
        -------
        obj
            Instance of Hirshfeld class. 
        """       
        promol = ProMolecule.build(method, charge=pro_charge, multiplicity=pro_multiplicity)
        promoldens = promol.electron_density(grids_coords, spin=spin, deriv=deriv)
        obj = cls(promoldens)
        return obj
    
    def sharing_function(self, promoldens=None):
        r"""Hirshfeld sharing funcition :math:`\omega_A(\mathbf{r})` defined as:

        .. math::
            \omega_A(\mathbf{r}) = \frac{\rho^0_A (\mathbf{r}) }{\sum_A \rho^0_A (\mathbf{r})} 

        Parameters
        ----------
        promoldens : PartitionDensity
            PartitionDensity instance for promolecule, by default None.        

        Returns
        -------
        omega : List[np.ndarray((N,), dtype=float)]
            Sharing functions for a list of atoms.            
        """        
        if promoldens is None:
            promoldens = self.promoldens
        
        omega = [free_atom_dens.density()/promoldens.density(mask=True) for free_atom_dens in promoldens]
        return omega