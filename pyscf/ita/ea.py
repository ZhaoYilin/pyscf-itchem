from pyscf.ita.ked import KineticEnergyDensity
from pyscf.ita.dens import OneElectronDensity, TwoElectronDensity

import numpy as np

class EnergyAnalysis(object):
    def __init__(
        self, 
        method, 
        grids,
        category='regular',
        rung=3 
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
        grids : Grids
            Pyscf Grids instance.
        rung : int
            Density derivate+1 level, by default 3,.           
        """
        self.method = method
        self.grids = grids  
        self.categrory = category
        self.rung = rung

    def build(
        self, 
        method=None, 
        grids=None,
        category=None,
        rung=None
    ):
        r"""Method to build the class.

        Parameters
        ----------
        method : PyscfMethod, optional
            Pyscf scf method or post-scf method instance, by deafult None.
        grids : Grids, optional
            Pyscf Grids instance, by deafult None.
        rung : int
            Density derivate+1 level, by default 3,.           
        """
        if method is None:
            method = self.method
        if grids is None:
            grids = self.grids
        if category is None:
            category = self.category
        if rung is None:
            rung = self.rung

        # Build molecule electron density.
        if category=='regular':
            dens = OneElectronDensity.build(method, grids, deriv=2)
            orbdens = OneElectronDensity.orbital_partition(method, grids, deriv=1)
            keds = KineticEnergyDensity(dens,orbdens)
        elif category=='joint':
            dens = TwoElectronDensity.build(method, grids, deriv=2)
            orbdens = TwoElectronDensity.orbital_partition(method, grids, deriv=1)
            keds = KineticEnergyDensity(dens,orbdens)
        else:
            raise ValueError("Not a valid category.")
        
        # Build ITA density
        self.keds = keds
        return self
    
    def general(
        self, 
        a = 0.0,
        grids_weights=None, 
        ked=None
    ):
        r"""Compute general(ish) kinetic energy defined as:

        .. math::
            T_\text{G} (\alpha) = \int
               \tau_\text{PD} (\mathbf{r}) +
               \frac{1}{4} (a - 1) \nabla^2 \rho (\mathbf{r})
                d\mathbf{r} 

        Parameters
        ----------
        a : float
            Value of parameter :math:`a`.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.general(a=a)

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result
    
    def thomas_fermi(
        self, 
        grids_weights=None, 
        ked=None
    ):
        r"""Thomas-Fermi kinetic energy defined as:

        .. math::
            T_\text{TF} \left(\mathbf{r}\right) = \int
                \tfrac{3}{10} \left(6 \pi^2 \right)^{2/3}
                \left(\frac{\rho\left(\mathbf{r}\right)}{2}\right)^{5/3}
                d\mathbf{r} 

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """    
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.thomas_fermi()

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result          

    def dirac(
        self, 
        grids_weights=None, 
        ked=None
    ):
        r"""Dirac kinetic energy defined as:

        .. math::
            T_\text{D} = \int
                \tfrac{3}{10} \left(6 \pi^2 \right)^{2/3}
                \left(\frac{\rho\left(\mathbf{r}\right)}{2}\right)^{5/3}
                d\mathbf{r} 

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """   
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.dirac()

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result              
    
    def weizsacker(
        self, 
        grids_weights=None, 
        ked=None
    ):    
        r"""Weizsacker kinetic energy defined as:

        .. math::
            T_\text{W} (\mathbf{r}) = \int \frac{1}{8}
                \frac{\lvert \nabla\rho (\mathbf{r}) \rvert^2}{\rho (\mathbf{r})}
                d\mathbf{r} 

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """   
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.weizsacker()

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result  
    
    def gradient_expansion(
        self,
        a = None,
        b = None, 
        grids_weights=None, 
        ked=None
    ):
        r"""Gradient expansion approximation of kinetic energy defined as:

        .. math::
            T_\text{GEA}(a,b) = \int
                \tau_\text{TF} \left(\mathbf{r}\right) +
                a* \tau_\text{W} \left(\mathbf{r}\right) +
                b* \nabla^2 \rho\left(\mathbf{r}\right)
                d\mathbf{r}

        There is a special case of :func:`gradient_expansion` with
        :math:`a=\tfrac{1}{9}` and :math:`b=\tfrac{1}{6}`.

        Parameters
        ----------
        a : float, optional
            Value of parameter :math:`a`, by default None.
        b : float, optional
            Value of parameter :math:`b`, by default None.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """   
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.gradient_expansion(a=a,b=b)

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result          
    
    def single_particle(
        self, 
        grids_weights=None, 
        ked=None
    ):
        r"""Single-particle kinetic energy defined as:

        .. math::
            T_\text{s} = \int \sum_i \frac{1}{8} 
                \frac{\nabla \rho_i \nabla \rho_i}{\rho_i}
                -\frac{1}{8} \nabla^2 \rho    
                d\mathbf{r} 

        where :math:`\rho_i` are the Kohn-Sham orbital densities.

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """   
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.single_particle()

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result          