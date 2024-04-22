import numpy as np

__all__=["KineticEnergyDensity"]

class KineticEnergyDensity:
    r"""The kinetic-energy density class.
    """    
    def __init__(
        self, 
        dens=None, 
        orbdens=None
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        dens : ElectronDensity, optional
            ElectronDensity instance for molecule, by default None.
        orbdens : List[ElectronDensity], optional
            List of ElectronDensity instance for orbitals, by default None.
        """
        self.dens = dens 
        self.orbdens = orbdens 

    def general(
        self, 
        a=0.0,
        rho=None, 
        rho_lapl=None,
        omega=None 

    ):
        r"""Compute general(ish) kinetic energy density defined as:

        .. math::
           \tau_\text{G} (\mathbf{r}, \alpha) =
               \tau_\text{PD} (\mathbf{r}) +
               \frac{1}{4} (a - 1) \nabla^2 \rho (\mathbf{r})

        Parameters
        ----------
        a : float
            Value of parameter :math:`a`.
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        rho_lapl : np.ndarray((N,), dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((N,), dtype=float)
            Kinetic energy density on grid of N points.
        """  
        if rho is None: 
            rho = self.dens.density()        
        if rho_lapl is None: 
            rho_lapl = self.dens.laplacian()
        if omega is not None:
            rho = rho*omega
            rho_lapl = rho_lapl*omega

        ked = rho + 1.0/4.0*(a-1)*rho_lapl
        return ked

    def thomas_fermi(
        self, 
        rho=None,
        omega=None
    ):
        r"""Thomas-Fermi kinetic energy density defined as:

        .. math::
           \tau_\text{TF} \left(\mathbf{r}\right) = \tfrac{3}{10} \left(6 \pi^2 \right)^{2/3}
                  \left(\frac{\rho\left(\mathbf{r}\right)}{2}\right)^{5/3}

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((N,), dtype=float)
            Kinetic energy density on grid of N points.
        """        
        if rho is None: 
            rho = self.dens.density()        
        if omega is not None:
            rho = rho*omega

        c_tf = 0.3 * (3.0 * np.pi**2.0)**(2.0 / 3.0)
        ked = c_tf * rho ** (5.0 / 3.0)
        return ked

    def dirac(
        self, 
        rho=None,
        omega=None
    ):
        r"""Dirac kinetic energy density defined as:

        .. math::
           \tau_\text{D} = \tfrac{3}{10} \left(6 \pi^2 \right)^{2/3}
                  \left(\frac{\rho\left(\mathbf{r}\right)}{2}\right)^{5/3}

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((N,), dtype=float)
            Kinetic energy density on grid of N points.
        """      
        if rho is None: 
            rho = self.dens.density()        
        if omega is not None:
            rho = rho*omega

        c_x = 3.0/4.0*(3.0/np.pi)**(1.0/3.0) 
        ked = -c_x * rho ** (4.0 / 3.0)
        return ked

    def weizsacker(
        self, 
        rho=None, 
        rho_grad_norm=None,
        omega=None
    ):    
        r"""Weizsacker kinetic energy density defined as:

        .. math::
           \tau_\text{W} (\mathbf{r}) = \frac{1}{8}
           \frac{\lvert \nabla\rho (\mathbf{r}) \rvert^2}{\rho (\mathbf{r})}

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        rho_grad_norm : np.ndarray((N,), dtype=float), optional
            Electron density graident norm on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((N,), dtype=float)
            Kinetic energy density on grid of N points.
        """
        if rho is None: 
            rho = self.dens.density(mask=True)        
        if rho_grad_norm is None: 
            rho_grad_norm = self.dens.gradient_norm()
        if omega is not None:
            rho = rho*omega
            rho_grad_norm = rho_grad_norm*omega

        ked = 0.125*rho_grad_norm**2/(rho)    
        return ked
    
    def gradient_expansion(
        self,
        a = None,
        b = None, 
        rho=None, 
        rho_grad_norm=None, 
        rho_lapl=None,
        omega=None
    ):
        r"""Gradient expansion approximation of kinetic energy density defined as:

        .. math::
           \tau_\text{GEA} \left(\mathbf{r}\right) =
           \tau_\text{TF} \left(\mathbf{r}\right) +
           a* \tau_\text{W} \left(\mathbf{r}\right) +
           b* \nabla^2 \rho\left(\mathbf{r}\right)

        There is a special case of :func:`gradient_expansion` with
        :math:`a=\tfrac{1}{9}` and :math:`b=\tfrac{1}{6}`.

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        rho_grad_norm : np.ndarray((N,), dtype=float), optional
            Electron density graident norm on grid of N points, by default None.
        rho_lapl : np.ndarray((N,), dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        a : float, optional
            Value of parameter :math:`a`, by default None.
        b : float, optional
            Value of parameter :math:`b`, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((N,), dtype=float)
            Kinetic energy density on grid of N points.
        """
        if a is None:
            a = 1.0/9.0
        if b is None:
            b = 1.0/6.0
        if rho is None: 
            rho = self.dens.density()        
        if rho_grad_norm is None: 
            rho_grad_norm = self.dens.gradient_norm()
        if rho_lapl is None: 
            rho_lapl = self.dens.laplacian()
        if omega is not None:
            rho = rho*omega
            rho_grad_norm = rho_grad_norm*omega
            rho_lapl = rho_lapl*omega

        tau_tf =  self.thomas_fermi(rho=rho)
        tau_w = self.weizsacker(rho=rho,rho_grad_norm=rho_grad_norm)
        ked = tau_tf + a*tau_w + b * rho_lapl
        return ked

    def single_particle(
        self, 
        rho_lapl=None,
        orbrho=None,
        orbrho_grad_norm=None,
        omega=None
    ):
        r"""Single-particle kinetic energy density defined as:

        .. math::
           \tau_\text{s} = \sum_i \frac{1}{8} \frac{\nabla \rho_i \nabla \rho_i}{\rho_i}
                -\frac{1}{8} \nabla^2 \rho    

        where :math:`\rho_i` are the Kohn-Sham orbital densities.

        Parameters
        ----------
        rho_lapl : np.ndarray((N,), dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        orbrho : np.ndarray((N,), dtype=float), optional
            Electron density of orbitals on grid of N points, by default None.
        orbrho_grad_norm : np.ndarray((N,), dtype=float), optional
            Electron density graident norm of orbitals on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((N,), dtype=float)
            Kinetic energy density on grid of N points.
        """
        if rho_lapl is None: 
            rho_lapl = self.dens.laplacian()
        if orbrho is None:
            orbrho = [] 
            for orbdens_i in self.orbdens:
                orbrho.append(orbdens_i.density(mask=True))
        if orbrho_grad_norm is None: 
            orbrho_grad_norm = [] 
            for orbdens_i in self.orbdens:
                orbrho_grad_norm.append(orbdens_i.gradient_norm())
        if omega is not None:
            rho_lapl = rho_lapl*omega
            orbrho = [orbrho_i*omega for orbrho_i in orbrho]
            orbrho_grad_norm = [orbrho_grad_norm_i*omega for orbrho_grad_norm_i in orbrho_grad_norm]

        term1 = orbrho_grad_norm[0]**2/orbrho[0]
        for orbrho_i,orbrho_grad_norm_i in zip(orbrho[1:],orbrho_grad_norm[1:]):
            term1 += orbrho_grad_norm_i**2/orbrho_i

        term1 *= 0.125
        ked = term1 - 0.125*rho_lapl
        return ked

class JointKineticEnergyDensity(KineticEnergyDensity):
    r"""The Joint kinetic-energy density class.
    """    
    def __init__(
        self, 
        dens=None, 
        orbdens=None
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        dens : TwoElectronDensity, optional
            TwoElectronDensity instance for molecule, by default None.
        orbdens : List[TwoElectronDensity], optional
            List of TwoElectronDensity instance for orbitals, by default None.
        """
        self.dens = dens 
        self.orbdens = orbdens 

class ConditionalKineticEnergyDensity(KineticEnergyDensity):
    r"""The conditional kinetic-energy density class.
    """
    def __init__(
        self,
        dens=None,
        orbdens=None 
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        dens : TwoElectronDensity, optional
            TwoElectronDensity instance for molecule, by default None.
        orbdens : List[TwoElectronDensity], optional
            List of TwoElectronDensity instance for orbitals, by default None.
        """        
        from pyscf.ita.dens import PartitionDensity

        marginal_dens = np.sum(dens, axis=-1)
        marginal_orbdens = PartitionDensity([np.sum(orbden, axis=-1) for orbden in orbdens])

        self.dens = marginal_dens
        self.orbdens = marginal_orbdens