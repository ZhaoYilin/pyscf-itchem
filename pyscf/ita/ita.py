"""
Information-theoretic approach (ITA) module. 

References
----------
The following references are for the atoms-in-molecules module.
"""

import numpy as np
from pyscf import dft

from pyscf.ita.dens import ElectronDensity, PartitionDensity
from pyscf.ita.ked import KineticEnergyDensity
from pyscf.ita.itad import ItaDensity
from pyscf.ita.promolecule import ProMolecule

__all__ = ["ITA"]

class ITA:
    r"""Information-Theoretic Approch (ITA) class.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> mf = scf.HF(mol)
    >>> mf.kernel()
    >>> grids = dft.Grids(mol)
    >>> grids.build()    
    >>> ita = ITA()
    >>> ita.method = mf
    >>> ita.grids = grids
    >>> ita.build()        
    """
    def __init__(
        self, 
        method, 
        grids, 
        rung=[3,3], 
        spin='ab',
        promolecule=None,
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
        grids : Grids
            Pyscf Grids instance.
        rung : List[int, int]
            List of molecule and promolecule derivate+1 level, by default [3,3].           
        spin: ('ab' | 'a' | 'b' | 'm')
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density,
            by default='ab'    
        promolecule : ProMolecule, optional
            ProMolecule instance, by default None.
        """
        self.method = method
        self.grids = grids  
        self.rung = rung
        self.spin = spin
        self.promolecule = promolecule
        
    def build(
        self, 
        method=None, 
        grids=None,
        rung=None, 
        spin=None,
        promolecule=None,
    ):
        r"""Method to build the class.

        Parameters
        ----------
        method : PyscfMethod, optional
            Pyscf scf method or post-scf method instance, by deafult None.
        grids : Grids, optional
            Pyscf Grids instance, by deafult None.
        rung : List[int, int], optional
            List of molecule and promolecule derivate+1 level, by default [3,0].           
        spin: ('ab' | 'a' | 'b' | 'm'), optional
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density,
            by default None.    
        promolecule : ProMolecule, optional
            ProMolecule instance, by default None.
        """
        if method is None:
            method = self.method
        if grids is None:
            grids = self.grids
        if rung is None:
            rung = self.rung
        if spin is None:
            spin = self.spin
        if promolecule is None:
            promolecule = self.promolecule

        # Build molecule electron density.
        moldens = ElectronDensity.build(method, grids.coords, spin=spin, deriv=rung[0]-1)
        self.moldens = moldens

        # Build kinetic energy electron density.
        if rung[0]==3:
            orbdens = PartitionDensity.orbital(method, grids.coords, spin=spin, deriv=1)
            self.orbdens = orbdens
            molkeds = KineticEnergyDensity(moldens,orbdens)
            self.molkeds = molkeds

        # Build promolecule electron density
        if promolecule is not None:
            promoldens = promolecule.electron_density(grids.coords, spin=spin, deriv=rung[1]-1)
        else:
            promoldens = None
        self.promoldens = promoldens

        # Build ITA density
        itad = ItaDensity(moldens, promoldens, molkeds)
        self.itad = itad

        return self

    def batch_compute(
        self, 
        ita_code=[], 
        representation='electron density',
        partition='hirshfeld',    
        filename = 'pyita.log',
    ):
        r"""ITA batch calcuation.
 
        Parameters
        ----------
        ita : ITA
            Instance of ITA class.
        ita_code : List[int]
            List of ITA code to calculate.
        representation : ('electron density' | 'shape function' | 'atoms in molecules')
            Type of representation, by default 'electron density'.
        partition : ('hirshfeld' | 'bader' | 'becke'), optional
            Atoms in molecule partition method, by default 'hirshfeld'.        
        filename : str, optional
            File path and name of output, by default 'pyita.log'
        """
        from pyscf.ita.script import batch_compute
        return batch_compute(self, ita_code, representation, partition, filename)

    def rho_power(
        self, 
        n = 2,
        grids_weights=None, 
        ita_density=None
    ):
        r"""Electron density of power n defined as :math:`\int \rho(\mathbf{r})^n d\mathbf{r}`.

        Parameters
        ----------
        n : int, optional
            Order of rho power, by default 2.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ita_density is None:
            ita_density = self.itad.rho_power(n=n)

        result = (grids_weights * ita_density).sum()
        return result

    def shannon_entropy(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Shannon entropy :math:`S_S` defined as:

        .. math::
            S_S = -\int \rho(\mathbf{r}) \ln \rho(\mathbf{r}) dr   

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ita_density is None:
            ita_density = self.itad.shannon_entropy()

        result = (grids_weights * ita_density).sum()
        return result

    def fisher_information(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Fisher information :math:`I_F` defined as:

        .. math::
            I_F = \int \frac{|\nabla \rho(\mathbf{r})|^2}{\rho(\mathbf{r})} d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.fisher_information()

        result = (grids_weights * ita_density).sum()
        return result

    def alternative_fisher_information(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Alternative Fisher information :math:`I^{\prime}_F` defined as:

        .. math::
            I^{\prime}_F = -\int \nabla^2 \rho(\mathbf{r}) \ln \rho(\mathbf{r}) d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """            
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.alternative_fisher_information()

        result = (grids_weights * ita_density).sum()
        return result
    
    def GBP_entropy(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Ghosh-Berkowitz-Parr (GBP) entropy :math:`S_{GBP}` defined as:

        .. math::
            S_{GBP} = \int \frac{3}{2}k\rho(\mathbf{r}) 
                \left[ c+\ln \frac{t(\mathbf{r};\rho)}{t_{TF}(\mathbf{r};\rho)} \right] d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.GBP_entropy()

        result = (grids_weights * ita_density).sum()
        return result 

    def renyi_entropy(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Renyi entropy :math:`R_n` defined as:

        .. math::
            R_n = \frac{1}{1-n} \ln \left[ \int \rho(\mathbf{r})^n d\mathbf{r} \right]

        Parameters
        ----------
        n : int, optional
            Order of the Renyi entropy, by default 2.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """ 
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.rho_power(n)
            
        result = (1/(1-n))*np.log10((grids_weights * ita_density).sum())
        return result
    
    def tsallis_entropy(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Tsallis entropy :math:`T_n` defined as:

        .. math::
            T_n = \frac{1}{n-1} \left[ 1- \int \rho(\mathbf{r})^n d\mathbf{r} \right]

        Parameters
        ----------
        n : int, optional
            Order of the Tsallis entropy, by default 2.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.rho_power(n)

        result = (1/(n-1))*(1-(grids_weights * ita_density).sum())
        return result 
    
    def onicescu_information(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Onicescu information :math:`E_n` defined as:

        .. math::
            E_n = \frac{1}{n-1} \int \rho(\mathbf{r})^n d\mathbf{r}
         
        Parameters
        ----------
        n : int, optional
            Order of the Onicescu information, by default 2.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.
        
        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.rho_power(n)

        result = (1/(n-1))*(grids_weights * ita_density).sum()        
        return result

    def relative_shannon_entropy(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Relative Shannon entropy :math:`S^r_S` defined as:

        .. math::
            S^r_S = \int \rho(\mathbf{r})\ln \frac{\rho(\mathbf{r})}{\rho_0(\mathbf{r})} d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.relative_shannon_entropy()

        result = (grids_weights * ita_density).sum()
        return result

    def relative_fisher_information(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Relative Fisher information defined as:

        .. math::
            {}^r_F I(\mathbf{r})
            = \int \rho(\mathbf{r}) 
            \left\vert 
            \frac{\nabla \rho(\mathbf{r})}{\rho(\mathbf{r})} 
            -\frac{\nabla \rho_0(\mathbf{r})}{\rho_0(\mathbf{r})} 
            \right\vert^2 dr

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.relative_fisher_information()

        result = (grids_weights * ita_density).sum()
        return result

    def relative_alternative_fisher_information(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Relative alternative Fisher information defined as:

        .. math::
            {}^r_F I^{\prime}(\mathbf{r}) 
            = \int \nabla^2 \rho(\mathbf{r}) 
            \ln \frac{\rho(\mathbf{r})}{\rho_0(\mathbf{r})}  dr

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.relative_alternative_fisher_information()

        result = (grids_weights * ita_density).sum()
        return result

    def relative_renyi_entropy(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Relative Renyi entropy :math:`R^r_n` defined as:

        .. math::
            R^r_n = \frac{1}{1-n} \ln 
                \left[ \int \frac{\rho^n(\mathbf{r})}{\rho^{n-1}_0(\mathbf{r})} d\mathbf{r} \right]

        Parameters
        ----------
        n : int, optional
            Order of the relative Renyi entropy, by default 2.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.relative_rho_power(n)

        result = (1/(1-n))*np.log10((grids_weights * ita_density).sum())
        return result

    def relative_tsallis_entropy(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Relative Tsallis entropy :math:`T^r_n` defined as:

        .. math::
            T^r_n = \frac{1}{n-1}  
                \left[ 1- \int \frac{\rho^n(\mathbf{r})}{\rho^{n-1}_0(\mathbf{r})} d\mathbf{r} \right]

        Parameters
        ----------
        n : int, optional
            Order of the relative Renyi entropy, by default 2.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.relative_rho_power(n)

        result = (1/(n-1))*(1-(grids_weights * ita_density).sum())
        return result    

    def relative_onicescu_information(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Relative Tsallis entropy :math:`E^r_n` defined as:

        .. math::
            E^r_n = \frac{1}{n-1} 
                \int \frac{\rho^n(\mathbf{r})}{\rho^{n-1}_0(\mathbf{r})} d\mathbf{r} 

        Parameters
        ----------
        n : int, optional
            Order of the relative Renyi entropy, by default 2.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.relative_rho_power(n)

        result = (1/(n-1))*(grids_weights * ita_density).sum()
        return result     
     
    def G1(
        self, 
        grids_weights=None, 
        ita_density=None            
    ):
        r"""G1 density defined as:

        .. math::
            G_1(\mathbf{r}) = \sum_A \int
            \nabla^2 \rho_A(\mathbf{r}) 
            \ln \frac{\rho_A(\mathbf{r})}{\rho^0_A(\mathbf{r})} 
            d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.G1()

        result = (grids_weights * ita_density).sum()
        return result        

    def G2(
        self, 
        grids_weights=None, 
        ita_density=None            
    ):
        r"""G2 density defined as:

        .. math::
            G_2(\mathbf{r}) = \sum_A \int 
            \rho_A(\mathbf{r}) \left[ 
            \frac{\nabla^2 \rho_A(\mathbf{r})}{\rho_A(\mathbf{r})} 
            -\frac{\nabla^2 \rho^0_A(\mathbf{r})}{\rho^0_A(\mathbf{r})}
            \right] d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.G2()

        result = (grids_weights * ita_density).sum()
        return result    

    def G3(
        self, 
        grids_weights=None, 
        ita_density=None            
    ):
        r"""G3 density defined as:

        .. math::
            G_3(\mathbf{r})
            = \sum_A \int 
            \rho_A(\mathbf{r}) \left\vert 
            \frac{\nabla \rho_A(\mathbf{r})}{\rho_A(\mathbf{r})} 
            -\frac{\nabla \rho^0_A(\mathbf{r})}{\rho^0_A(\mathbf{r})} 
            \right\vert^2 d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.G3()

        result = (grids_weights * ita_density).sum()
        return result            
