import numpy as np
from functools import partial
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
        NotImplemented 
 

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
        grids, 
        pro_charge=None, 
        pro_multiplicity=None, 
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
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 0.

        Returns
        -------
        obj
            Instance of Hirshfeld class. 
        """       
        promol = ProMolecule.build(method, charge=pro_charge, multiplicity=pro_multiplicity)
        promoldens = promol.one_electron_density(grids, deriv=deriv)
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

    def partition(self, ita_name, **kwargs):
        """_summary_

        Parameters
        ----------
        ita_name : _type_
            _description_
        """        
        def decorator(ita):
            omega = self.sharing_function()
            ita_func = getattr(ita, ita_name)
            ita_func = partial(ita_func, **kwargs)
            if ita_name in ['renyi_entropy','tsallis_entropy','onicescu_information']:
                itad_func = getattr(ita.itad, "rho_power")
            else:
                itad_func = getattr(ita.itad, ita_name)
                itad_func = partial(itad_func, **kwargs) 
            def wrapper(**kwargs):
                atomic_ita = []
                if ita.representation == 'electron density':
                    for omega_i in omega:
                        itad_i = itad_func()*omega_i
                        atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                        atomic_ita.append(atomic_partition) 
                elif ita.representation == 'shape function':
                    for omega_i in omega:
                        nelec_i = (ita.grids.weights * ita.moldens.density() * omega_i).sum()
                        itad_i = itad_func(omega=1./nelec_i)
                        atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                        atomic_ita.append(atomic_partition) 
                elif ita.representation=='atoms in molecules':
                    for atom_id, omega_i in enumerate(omega):
                        if ita_name in ['fisher_information','G3']:
                            prorho_i = ita.prodens[atom_id].density(mask=True)
                            prorho_grad_i = ita.prodens[atom_id].gradient()
                            itad_i = itad_func(omega=omega_i, prorho=prorho_i, prorho_grad=prorho_grad_i)                          
                        else:
                            itad_i = itad_func(omega=omega_i)
                        atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                        atomic_ita.append(atomic_partition) 
                else:
                    raise ValueError("Not valid representation.")                                       
                return atomic_ita
            return wrapper
        return decorator