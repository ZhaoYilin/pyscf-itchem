import numpy as np

from pyscf.itchem.scope import Local
from pyscf.itchem.dens import OneElectronDensity

__all__=["KineticEnergyDensity"]

class KineticEnergyDensity(Local):
    r"""The kinetic-energy density class.
    
    Attributes
    ----------
    dens : ElectronDensity, optional
        Electron density and it's derivative.
    orbdens : OrbitalDensity, optional
        Orbitall density.
    """  
    dens=None, 
    orbdens=None
    def __init__(
        self, 
        method=None, 
        grids_coords=None,
        rung=3, 
        representation='electron density',
        partition=None,
        promolecule=None
    ):
        r"""Initialize a instance.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
        grids_coords : np.ndarray((Ngrids,3), dtype=float), optional
            Grids coordinates on N points, by default None.            
        rung : int, optional
            Density derivate+1 level, by default 3.
        representation : ('electron density' | 'shape function' | 'atoms in molecules')
            Type of representation, by default 'electron density'.            
        partition : (None | 'hirshfeld' | 'bader' | 'becke'), optional
            Atoms in molecule partition method, by default None.                      
        promolecule : ProMolecule, optional
            ProMolecule instance, by default None.
        """
        self.method = method
        self.grids_coords = grids_coords 
        self.rung = rung
        self.representation = representation
        self.partition = partition
        self.promolecule = promolecule 

    def build(
        self, 
        method=None, 
        grids_coords=None,
        rung=None, 
        representation=None,
        partition=None,
        promolecule=None,
    ):
        r"""Setup ITA and initialize some control parameters. Whenever you
        change the value of the attributes of :class:`ITA`, you need call
        this function to refresh the internal data of ITA.

        Parameters
        ----------
        method : PyscfMethod, optional
            Pyscf scf method or post-scf method instance, by deafult None.
        grids_coords : np.ndarray((Ngrids,3), dtype=float), optional
            Grids coordinates on N points, by default None.            
        rung : int
            Density derivate+1 level, by default 3,.           
        representation : ('electron density' | 'shape function' | 'atoms in molecules')
            Type of representation, by default 'electron density'.            
        partition : (None | 'hirshfeld' | 'bader' | 'becke'), optional
            Atoms in molecule partition method, by default None.                      
        promolecule : ProMolecule, optional
            ProMolecule instance, by default None.
        """
        if method is None: method = self.method 
        else: self.method = method
        if grids_coords is None: grids_coords = self.grids_coords
        else: self.grids_coords = grids_coords
        if rung is None: rung = self.rung
        else: self.rung = rung
        if representation is None: representation = self.representation
        else: self.representation = representation
        if partition is None: partition = self.partition
        else: self.partition = partition
        if promolecule is None: promolecule = self.promolecule
        else: self.promolecule = promolecule

        if partition=='hirshfeld' and promolecule is None:
            from pyscf.itchem.promolecule import ProMolecule
            promolecule = ProMolecule(method).build()
            self.promolecule = promolecule

        nelec = self.method.mol.nelectron
        dens = OneElectronDensity.build(method, grids_coords, deriv=self.rung-1)
        if rung==3:
            if representation=='shape function': 
                dens = dens/nelec
                orbdens = orbdens/nelec
            orbdens = OneElectronDensity.orbital_partition(method, grids_coords, deriv=1)  
            self.orbdens = orbdens
        else:
            if representation=='shape function': 
                dens = dens/nelec
        
        self.dens = dens
        return self
    
    def positive_definite(
        self, 
        orbrho=None,
        orbrho_grad_norm=None,
        omega=None
    ):
        r"""The positive definite (nonnegative) kinetic energy density defined as:

        .. math::
           \tau = \frac{1}{2} \sum_i |n_i \nabla \phi_i|^2

        where :math:`\phi_i` and :math:`n_i` ni are the molecule (spin) orbitals and 
        their occupation numbers. 

        Parameters
        ----------
        orbrho : np.ndarray((Ngrids,), dtype=float), optional
            Electron density of orbitals on grid of N points, by default None.
        orbrho_grad_norm : np.ndarray((Ngrids,), dtype=float), optional
            Electron density graident norm of orbitals on grid of N points, by default None.
        omega : np.ndarray((Ngrids,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((Ngrids,), dtype=float)
            Kinetic energy density on grid of N points.
        """       
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

        ked = term1*0.125
        return ked

    def general(
        self, 
        a=0.0,
        orbrho=None,
        orbrho_grad_norm=None,
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
        a : float, optional
            Value of parameter :math:`a`, by default 0.
        orbrho : np.ndarray((Ngrids,), dtype=float), optional
            Electron density of orbitals on grid of N points, by default None.
        orbrho_grad_norm : np.ndarray((Ngrids,), dtype=float), optional
            Electron density graident norm of orbitals on grid of N points, by default None.
        rho_lapl : np.ndarray((Ngrids,), dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        omega : np.ndarray((Ngrids,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((Ngrids,), dtype=float)
            Kinetic energy density on grid of N points.
        """  
        if orbrho is None:
            orbrho = [] 
            for orbdens_i in self.orbdens:
                orbrho.append(orbdens_i.density(mask=True))
        if orbrho_grad_norm is None: 
            orbrho_grad_norm = [] 
            for orbdens_i in self.orbdens:
                orbrho_grad_norm.append(orbdens_i.gradient_norm())
        if rho_lapl is None: 
            rho_lapl = self.dens.laplacian()
        if omega is not None:
            pd = pd*omega
            rho = rho*omega
            rho_lapl = rho_lapl*omega

        tau_pd =  self.positive_definite(orbrho=orbrho,orbrho_grad_norm=orbrho_grad_norm)
        ked = tau_pd + 1.0/4.0*(a-1)*rho_lapl
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
        rho : np.ndarray((Ngrids,), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((Ngrids,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((Ngrids,), dtype=float)
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
        rho : np.ndarray((Ngrids,), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((Ngrids,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((Ngrids,), dtype=float)
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
        rho : np.ndarray((Ngrids,), dtype=float), optional
            Electron density on grid of N points, by default None.
        rho_grad_norm : np.ndarray((Ngrids,), dtype=float), optional
            Electron density graident norm on grid of N points, by default None.
        omega : np.ndarray((Ngrids,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((Ngrids,), dtype=float)
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
        rho : np.ndarray((Ngrids,), dtype=float), optional
            Electron density on grid of N points, by default None.
        rho_grad_norm : np.ndarray((Ngrids,), dtype=float), optional
            Electron density graident norm on grid of N points, by default None.
        rho_lapl : np.ndarray((Ngrids,), dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        a : float, optional
            Value of parameter :math:`a`, by default None.
        b : float, optional
            Value of parameter :math:`b`, by default None.
        omega : np.ndarray((Ngrids,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((Ngrids,), dtype=float)
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
        rho_lapl : np.ndarray((Ngrids,), dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        orbrho : np.ndarray((Ngrids,), dtype=float), optional
            Electron density of orbitals on grid of N points, by default None.
        orbrho_grad_norm : np.ndarray((Ngrids,), dtype=float), optional
            Electron density graident norm of orbitals on grid of N points, by default None.
        omega : np.ndarray((Ngrids,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ked : np.ndarray((Ngrids,), dtype=float)
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