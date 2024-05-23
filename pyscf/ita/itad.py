import numpy as np

class ItaDensity(object):
    r"""Information-Theoretic Approch (ITA) Density class.
    """
    def __init__(
        self, 
        dens=None, 
        keds=None
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        dens : ElectronDensity, optional
            ElectronEensity instance for molecule, by default None.
        keds : KineticEnergyDensity, optional
            KineticEnergyDensity instance for molecule, by default None.
        """        
        self.dens = dens
        self.keds = keds

    def rho_power(
        self, 
        n=2, 
        rho=None, 
        omega=None
    ):
        r"""Electron density of power n defined as :math:`\rho(\mathbf{r})^n`.

        Parameters
        ----------
        n : int, optional
            Order of rho power, by default 2.
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None:           
            rho = self.dens.density()
        if omega is not None:
            rho = rho*omega

        ita_density = rho**n
        return ita_density
    
    def shannon_entropy(
        self, 
        rho=None, 
        omega=None
    ):
        r"""Shannon entropy density defined as:

        .. math::
            s_S = -\rho(\mathbf{r}) \ln \rho(\mathbf{r})       

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.
            
        Returns
        -------
        ita_density : np.ndarray((N,), dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None: 
            rho = self.dens.density() 
        if omega is not None:
            rho = rho*omega

        ita_density = -rho*np.ma.log(rho)
        return ita_density
    
    def fisher_information(
        self, 
        rho=None, 
        rho_grad_norm=None, 
        omega=None
    ):
        r"""Fisher information density defined as:

        .. math::
            i_F = \frac{|\nabla \rho(\mathbf{r})|^2}{\rho(\mathbf{r})} 

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
        ita_density : np.ndarray((N,), dtype=float)
            Information theory density on grid of N points.
        """      
        if rho is None: 
            rho = self.dens.density(mask=True)
        if rho_grad_norm is None:        
            rho_grad_norm = self.dens.gradient_norm()
        if omega is not None:
            rho = rho*omega
            rho_grad_norm = rho_grad_norm*omega

        ita_density = rho_grad_norm**2/(rho)
        return ita_density

    def alternative_fisher_information(
        self, 
        rho=None, 
        rho_lapl=None, 
        omega=None
    ):
        r"""Alternative Fisher information density defined as:

        .. math::
            I^{\prime}_F = \nabla^2 \rho(\mathbf{r}) \ln \rho(\mathbf{r})

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        rho_lapl : np.ndarray((N,), dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray((N,), dtype=float)
            Information theory density on grid of N points.
        """            
        if rho is None: 
            rho = self.dens.density()
        if rho_lapl is None: 
            rho_lapl = self.dens.laplacian()
        if omega is not None:
            rho = rho*omega
            rho_lapl = rho_lapl*omega

        ita_density = -rho_lapl*np.ma.log(rho)
        return ita_density

    def GBP_entropy(
        self, 
        rho=None, 
        ts=None, 
        tTF=None, 
        k=1.0,
        omega=None
    ):
        r"""Ghosh-Berkowitz-Parr (GBP) entropy density :math:`s_{GBP}` defined as:

        .. math::
            s_{GBP} = \frac{3}{2}k\rho(\mathbf{r}) 
                \left[ c+\ln \frac{t(\mathbf{r};\rho)}{t_{TF}(\mathbf{r};\rho)} \right]

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        ts : np.ndarray((N,), dtype=float), optional
            Single particle kenetic Electron density on grid of N points, by default None.
        tTF : np.ndarray((N,), dtype=float), optional
            Thomas-Fermi kinetic energy density on grid of N points, by default None.
        k : float, optional
            Boltzmann constant, by default 1.0 for convenience.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray((N,), dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None: 
            rho = self.dens.density()
        if ts is None:
            ts = self.keds.single_particle()            
        if tTF is None:
            tTF = self.keds.thomas_fermi()
        if omega is not None:
            rho = rho*omega
            ts = self.keds.single_particle(omega=omega)            
            tTF = self.keds.thomas_fermi(omega=omega) 

        cK = 0.3 * (3.0 * np.pi**2.0)**(2.0 / 3.0)
        c = (5/3) + np.log(4*np.pi*cK/3)

        ita_density = 1.5*k*rho*(c+np.ma.log(ts/tTF))   
        return ita_density  

class RelativeItaDensity(ItaDensity):
    r"""Relative Information-Theoretic Approch (ITA) Density class.
    """
    def __init__(
        self, 
        dens=None, 
        prodens=None
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        dens : ElectronDensity, optional
            ElectronEensity instance for molecule, by default None.
        prodens : ElectronDensity, optional
            ElectronEensity instance for promolecule, by default None.
        """        
        self.dens = dens
        self.prodens = prodens

    def shannon_entropy(
        self, 
        rho=None, 
        prorho=None, 
        omega=None
    ):
        r"""Relative Shannon entropy density defined as:

        .. math::
            s^r_S = \rho(\mathbf{r})\ln \frac{\rho(\mathbf{r})}{\rho_0(\mathbf{r})}

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None:           
            rho = self.dens.density(mask=True) 
        if prorho is None:
            prorho = self.prodens.density(mask=True)
        if omega is not None:
            rho = rho*omega
            prorho = prorho*omega

        ita_density = rho*np.ma.log(rho/prorho)  
        return ita_density

    def fisher_information(
        self, 
        rho=None, 
        prorho=None, 
        rho_grad=None, 
        prorho_grad=None, 
        omega=None
    ):
        r"""Relative Fisher information density defined as:

        .. math::
            {}^r_F i(\mathbf{r})
            = \rho(\mathbf{r}) 
            \left\vert 
            \frac{\nabla \rho(\mathbf{r})}{\rho(\mathbf{r})} 
            -\frac{\nabla \rho_0(\mathbf{r})}{\rho_0(\mathbf{r})} 
            \right\vert^2

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_grad : np.ndarray(dtype=float), optional
            Electron density graident on grid of N points, by default None.
        prorho_grad : np.ndarray(dtype=float), optional
            Electron density graident of promolecule on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None:           
            rho = self.dens.density(mask=True)
        if prorho is None:
            prorho = self.prodens.density(mask=True)
        if rho_grad is None:        
            rho_grad = self.dens.gradient()
        if prorho_grad is None:
            prorho_grad = self.prodens.gradient()
        if omega is not None:
            rho = rho*omega
            rho_grad = rho_grad*omega
        ita_density = rho*np.linalg.norm(rho_grad/rho[np.newaxis,:] - prorho_grad/prorho[np.newaxis,:],axis=0)**2  
        return ita_density   

    def alternative_fisher_information(
        self, 
        rho=None, 
        prorho=None, 
        rho_lapl=None,
        omega=None
    ):
        r"""Relative alternative Fisher information density defined as:

        .. math::
            {}^r_F i^{\prime}(\mathbf{r}) 
            = \nabla^2 \rho(\mathbf{r}) 
            \ln \frac{\rho(\mathbf{r})}{\rho_0(\mathbf{r})}        

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_lapl : np.ndarray(dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None:           
            rho = self.dens.density()
        if prorho is None:
            prorho = self.prodens.density(mask=True)
        if rho_lapl is None:        
            rho_lapl = self.dens.laplacian()
        if omega is not None:
            rho_lapl = rho_lapl*omega

        ita_density = rho_lapl*np.ma.log(rho/prorho)
        return ita_density    

    def rho_power(
        self, 
        n=2, 
        rho=None, 
        prorho=None, 
        omega=None
    ):
        r"""Relative Renyi entropy density defined as:

        .. math::
            r^r_n = \frac{\rho^n(\mathbf{r})}{\rho^{n-1}_0(\mathbf{r})}

        Parameters
        ----------
        n : int, optional
            Order of relative Renyi entropy, by default 2.
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """           
        if rho is None:           
            rho = self.dens.density() 
        if prorho is None:
            prorho = self.prodens.density()
        if omega is not None:
            rho = rho*omega
            prorho = prorho*omega

        prorho_power = np.ma.masked_less(prorho**(n-1), 1.0e-30)
        prorho_power.filled(1.0e-30)
        ita_density = rho**n/prorho_power
        return ita_density
    
    def G1(
        self, 
        rho=None, 
        prorho=None, 
        rho_lapl=None,
        omega=None
    ):
        r"""G1 density defined as:

        .. math::
            g_1(\mathbf{r}) \equiv {}^r_F i^{\prime}(\mathbf{r}) 
            = \nabla^2 \rho(\mathbf{r}) 
            \ln \frac{\rho(\mathbf{r})}{\rho_0(\mathbf{r})}

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_lapl : np.ndarray(dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        return self.alternative_fisher_information(rho, prorho, rho_lapl, omega) 

    def G2(
        self, 
        rho=None, 
        prorho=None, 
        rho_lapl=None, 
        prorho_lapl=None, 
        omega=None
    ):
        r"""G2 density defined as:

        .. math::
            g_2(\mathbf{r}) = \rho(\mathbf{r}) 
            \left[ 
            \frac{\nabla^2 \rho(\mathbf{r})}{\rho(\mathbf{r})} 
            -\frac{\nabla^2 \rho_0(\mathbf{r})}{\rho_0(\mathbf{r})}
            \right]

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_lapl : np.ndarray(dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        prorho_lapl : np.ndarray(dtype=float), optional
            Electron density laplacian of promolecule on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None:           
            rho = self.dens.density(mask=True) 
        if prorho is None:
            prorho = self.prodens.density(mask=True)
        if rho_lapl is None:           
            rho_lapl = self.dens.laplacian() 
        if prorho_lapl is None:
            prorho_lapl = self.prodens.laplacian()     

        if omega is not None:
            ita_density = rho*omega*(rho_lapl/rho-prorho_lapl/prorho)
        else:
            ita_density = rho*(rho_lapl/rho-prorho_lapl/prorho)

        return ita_density

    def G3(
        self, 
        rho=None, 
        prorho=None, 
        rho_grad=None, 
        prorho_grad=None, 
        omega=None
    ):
        r"""G3 density defined as:

        .. math::
            g_3(\mathbf{r}) \equiv {}^r_F i(\mathbf{r})
            = \rho(\mathbf{r}) 
            \left\vert 
            \frac{\nabla \rho(\mathbf{r})}{\rho(\mathbf{r})} 
            -\frac{\nabla \rho_0(\mathbf{r})}{\rho_0(\mathbf{r})} 
            \right\vert^2

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_grad : np.ndarray(dtype=float), optional
            Electron density graident on grid of N points, by default None.
        prorho_grad : np.ndarray(dtype=float), optional
            Electron density graident of promolecule on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        return self.fisher_information(rho, prorho, rho_grad, prorho_grad, omega)

class JointItaDensity(ItaDensity):
    r"""Information-Theoretic Approch (ITA) Density class.
    """
    def __init__(
        self, 
        dens=None,
        keds=None 
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        dens : ElectronDensity, optional
            ElectronEensity instance for molecule, by default None.
        keds : KineticEnergyDensity, optional
            KineticEnergyDensity instance for molecule, by default None.
        """       
        self.dens = dens
        self.keds = keds

    def rho_power(
        self, 
        n=2, 
        gamma=None, 
        omega=None
    ):
        r"""Electron density of power n defined as :math:`\gamma(\mathbf{r})^n`.

        Parameters
        ----------
        n : int, optional
            Order of rho power, by default 2.
        gamma : np.ndarray((N,N), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((N,N), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray((N,N), dtype=float)
            Information theory density on grid of N points.
        """
        ita_density = super(JointItaDensity, self).rho_power(n,gamma,omega)
        return ita_density
    
    def shannon_entropy(
        self, 
        gamma=None, 
        omega=None
    ):
        r"""Shannon entropy density defined as:

        .. math::
            s_S = -\gamma(\mathbf{r}) \ln \gamma(\mathbf{r})       

        Parameters
        ----------
        gamma : np.ndarray((N,N), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((N,N), dtype=float), optional
            Sharing function of single atom, by default None.
            
        Returns
        -------
        ita_density : np.ndarray((N,N), dtype=float)
            Information theory density on grid of N points.
        """
        ita_density = super(JointItaDensity, self).shannon_entropy(gamma,omega)
        return ita_density
    
    def fisher_information(
        self, 
        gamma=None, 
        gamma_grad_norm=None, 
        omega=None
    ):
        r"""Fisher information density defined as:

        .. math::
            i_F = \frac{|\nabla \gamma(\mathbf{r})|^2}{\gamma(\mathbf{r})} 

        Parameters
        ----------
        gamma : np.ndarray((N,N), dtype=float), optional
            Electron density on grid of N points, by default None.
        gamma_grad_norm : np.ndarray((N,N), dtype=float), optional
            Electron density graident norm on grid of N points, by default None.
        omega : np.ndarray((N,N), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray((N,N), dtype=float)
            Information theory density on grid of N points.
        """      
        ita_density = super(JointItaDensity, self).fisher_information(gamma,gamma_grad_norm,omega)
        return ita_density

    def alternative_fisher_information(
        self, 
        gamma=None, 
        gamma_lapl=None, 
        omega=None
    ):
        r"""Alternative Fisher information density defined as:

        .. math::
            I^{\prime}_F = \nabla^2 \gamma(\mathbf{r}) \ln \gamma(\mathbf{r})

        Parameters
        ----------
        gamma : np.ndarray((N,N), dtype=float), optional
            Electron density on grid of N points, by default None.
        gamma_lapl : np.ndarray((N,N), dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        omega : np.ndarray((N,N), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray((N,N), dtype=float)
            Information theory density on grid of N points.
        """  
        ita_density = super(JointItaDensity, self).alternative_fisher_information(gamma,gamma_lapl,omega)
        return ita_density                  

    def GBP_entropy(
        self, 
        gamma=None, 
        ts=None, 
        tTF=None, 
        k=1.0,
        omega=None
    ):
        r"""Ghosh-Berkowitz-Parr (GBP) entropy density :math:`s_{GBP}` defined as:

        .. math::
            s_{GBP} = \frac{3}{2}k\gamma(\mathbf{r}) 
                \left[ c+\ln \frac{t(\mathbf{r};\gamma)}{t_{TF}(\mathbf{r};\gamma)} \right]

        Parameters
        ----------
        gamma : np.ndarray((N,N), dtype=float), optional
            Two electron density on grid of N points, by default None.
        ts : np.ndarray((N,N), dtype=float), optional
            Single particle kenetic Electron density on grid of N points, by default None.
        tTF : np.ndarray((N,N), dtype=float), optional
            Thomas-Fermi kinetic energy density on grid of N points, by default None.
        k : float, optional
            Boltzmann constant, by default 1.0 for convenience.
        omega : np.ndarray((N,N), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray((N,N), dtype=float)
            Information theory density on grid of N points.
        """
        ita_density = super(JointItaDensity, self).GBP_entropy(gamma,ts,tTF,k,omega)
        return ita_density                  
    
class ConditionalItaDensity(ItaDensity):
    r"""Conditional Information-Theoretic Approch (ITA) Density class.
    """
    def __init__(
        self, 
        onedens=None,
        twodens=None 
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        onedens : ElectronDensity, optional
            One particle ElectronEensity instance for molecule, by default None.
        twodens : ElectronDensity, optional
            Two particle ElectronEensity instance for reference, by default None.
        """        
        self.onedens = onedens
        self.twodens = twodens

    def rho_power(
        self, 
        n=2, 
        rho=None, 
        gamma=None,
        omega=None
    ):
        r"""Electron density of power n defined as :math:`\rho(\mathbf{r})^n`.

        Parameters
        ----------
        n : int, optional
            Order of rho power, by default 2.
        rho : np.ndarray((N,), dtype=float), optional
            One electron density on grid of N points, by default None.
        gamma : np.ndarray((N,N), dtype=float), optional
            Two electron density on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None: 
            rho = self.onedens.density(mask=True) 
        if gamma is None: 
            gamma = self.twodens.density() 
        if omega is not None:
            rho = rho*omega
            gamma = gamma*omega

        rho = np.ma.masked_less(rho**(n-1), 1.0e-30)
        rho.filled(1.0e-30)
        ita_density = gamma**n/rho
        return ita_density
    
    def shannon_entropy(
        self, 
        rho=None, 
        gamma = None,
        omega=None
    ):
        r"""Shannon entropy density defined as:

        .. math::
            s_S = -\rho(\mathbf{r}) \ln \rho(\mathbf{r})       

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.
            
        Returns
        -------
        ita_density : np.ndarray((N,), dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None: 
            rho = self.onedens.density(mask=True) 
        if gamma is None: 
            gamma = self.twodens.density() 
        if omega is not None:
            rho = rho*omega
            gamma = gamma*omega

        rho = np.ma.masked_less(rho, 1.0e-30)
        rho.filled(1.0e-30)
        ita_density = gamma*np.ma.log(gamma/rho)
        return ita_density


    def fisher_information(
        self, 
        gamma=None, 
        rho=None, 
        gamma_grad=None, 
        rho_grad=None, 
        omega=None
    ):
        r"""Relative Fisher information density defined as:

        .. math::
            {}^r_F i(\mathbf{r})
            = \rho(\mathbf{r}) 
            \left\vert 
            \frac{\nabla \rho(\mathbf{r})}{\rho(\mathbf{r})} 
            -\frac{\nabla \rho_0(\mathbf{r})}{\rho_0(\mathbf{r})} 
            \right\vert^2

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_grad : np.ndarray(dtype=float), optional
            Electron density graident on grid of N points, by default None.
        prorho_grad : np.ndarray(dtype=float), optional
            Electron density graident of promolecule on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if gamma is None:           
            gamma = self.twodens.density(mask=True)
        if gamma_grad is None:        
            gamma_grad = self.twodens.gradient()
        if rho is None:
            rho = self.onedens.density(mask=True)            
        if rho_grad is None:
            rho_grad = self.onedens.gradient()
        if omega is not None:
            gamma = gamma*omega
            gamma_grad = gamma_grad*omega

        gamma_newaxis = np.ma.masked_less(gamma[np.newaxis,:], 1.0e-30)
        gamma_newaxis.filled(1.0e-30)
        rho_newaxis = np.ma.masked_less(rho[np.newaxis,:], 1.0e-30)
        rho_newaxis.filled(1.0e-30)

        ita_density = []
        ita_density.append(gamma*np.linalg.norm(gamma_grad/gamma_newaxis, axis=0)**2)
        ita_density.append(rho*np.linalg.norm(rho_grad/rho_newaxis, axis=0)**2) 

        return ita_density   

    def alternative_fisher_information(
        self, 
        gamma=None, 
        rho=None, 
        gamma_lapl=None,
        omega=None
    ):
        r"""Relative alternative Fisher information density defined as:

        .. math::
            {}^r_F i^{\prime}(\mathbf{r}) 
            = \nabla^2 \rho(\mathbf{r}) 
            \ln \frac{\rho(\mathbf{r})}{\rho_0(\mathbf{r})}        

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_lapl : np.ndarray(dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if gamma is None:           
            gamma = self.twodens.density()
        if rho is None:
            rho = self.onedens.density(mask=True)
        if gamma_lapl is None:        
            gamma_lapl = self.twodens.laplacian()
        if omega is not None:
            gamma_lapl = gamma_lapl*omega

        ita_density = gamma_lapl*np.ma.log(gamma/rho)
        return ita_density 

class MutualItaDensity(ItaDensity):
    r"""Mutual Information-Theoretic Approch (ITA) Density class.
    """
    def __init__(
        self, 
        onedens=None,
        twodens=None 
    ):
        r""" Initialize a instance.

        Parameters
        ----------
        onedens : ElectronDensity, optional
            One particle ElectronEensity instance for molecule, by default None.
        twodens : ElectronDensity, optional
            Two particle ElectronEensity instance for reference, by default None.
        """        
        self.onedens = onedens
        self.twodens = twodens

    def rho_power(
        self, 
        n=2, 
        rho=None, 
        gamma=None,
        omega=None
    ):
        r"""Electron density of power n defined as :math:`\rho(\mathbf{r})^n`.

        Parameters
        ----------
        n : int, optional
            Order of rho power, by default 2.
        rho : np.ndarray((N,), dtype=float), optional
            One electron density on grid of N points, by default None.
        gamma : np.ndarray((N,N), dtype=float), optional
            Two electron density on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None: 
            rho = self.onedens.density() 
        if gamma is None: 
            gamma = self.twodens.density() 
        if omega is not None:
            rho = rho*omega
            gamma = gamma*omega

        rho_outer = np.ma.masked_less(np.outer(rho,rho)**(n-1), 1.0e-30)
        rho_outer.filled(1.0e-30)
        ita_density = gamma**n/rho_outer
        return ita_density
    
    def shannon_entropy(
        self, 
        rho=None, 
        gamma = None,
        omega=None
    ):
        r"""Shannon entropy density defined as:

        .. math::
            s_S = -\rho(\mathbf{r}) \ln \rho(\mathbf{r})       

        Parameters
        ----------
        rho : np.ndarray((N,), dtype=float), optional
            Electron density on grid of N points, by default None.
        omega : np.ndarray((N,), dtype=float), optional
            Sharing function of single atom, by default None.
            
        Returns
        -------
        ita_density : np.ndarray((N,), dtype=float)
            Information theory density on grid of N points.
        """
        if rho is None: 
            rho = self.onedens.density() 
        if gamma is None: 
            gamma = self.twodens.density() 
        if omega is not None:
            rho = rho*omega
            gamma = gamma*omega

        rho_outer = np.ma.masked_less(np.outer(rho,rho), 1.0e-30)
        rho_outer.filled(1.0e-30)
        ita_density = gamma*np.ma.log(gamma/rho_outer)
        return ita_density


    def fisher_information(
        self, 
        gamma=None, 
        rho=None, 
        gamma_grad=None, 
        rho_grad=None, 
        omega=None
    ):
        r"""Relative Fisher information density defined as:

        .. math::
            {}^r_F i(\mathbf{r})
            = \rho(\mathbf{r}) 
            \left\vert 
            \frac{\nabla \rho(\mathbf{r})}{\rho(\mathbf{r})} 
            -\frac{\nabla \rho_0(\mathbf{r})}{\rho_0(\mathbf{r})} 
            \right\vert^2

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_grad : np.ndarray(dtype=float), optional
            Electron density graident on grid of N points, by default None.
        prorho_grad : np.ndarray(dtype=float), optional
            Electron density graident of promolecule on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if gamma is None:           
            gamma = self.twodens.density(mask=True)
        if gamma_grad is None:        
            gamma_grad = self.twodens.gradient()
        if rho is None:
            rho = self.onedens.density(mask=True)            
        if rho_grad is None:
            rho_grad = self.onedens.gradient()
        if omega is not None:
            gamma = gamma*omega
            gamma_grad = gamma_grad*omega

        gamma = np.ma.masked_less(gamma, 1.0e-30)
        gamma.filled(1.0e-30)
        rho_outer = np.ma.masked_less(np.outer(rho,rho), 1.0e-30)
        rho_outer.filled(1.0e-30)

        # r in (x,y,z) 
        term1 = np.array([gamma_r/gamma for gamma_r in gamma_grad]) 
        term2 = np.array([(np.outer(rho,rho)+np.outer(rho_r,rho))/rho_outer for rho_r in rho_grad])
        ita_density = gamma*np.linalg.norm(term1-term2, axis=0)**2 

        return ita_density   

    def alternative_fisher_information(
        self, 
        gamma=None, 
        rho=None, 
        gamma_lapl=None,
        omega=None
    ):
        r"""Relative alternative Fisher information density defined as:

        .. math::
            {}^r_F i^{\prime}(\mathbf{r}) 
            = \nabla^2 \rho(\mathbf{r}) 
            \ln \frac{\rho(\mathbf{r})}{\rho_0(\mathbf{r})}        

        Parameters
        ----------
        rho : np.ndarray(dtype=float), optional
            Electron density on grid of N points, by default None.
        prorho : np.ndarray(dtype=float), optional
            Electron density of promolecule on grid of N points, by default None.
        rho_lapl : np.ndarray(dtype=float), optional
            Electron density laplacian on grid of N points, by default None.
        omega : np.ndarray(dtype=float), optional
            Sharing function of single atom, by default None.

        Returns
        -------
        ita_density : np.ndarray(dtype=float)
            Information theory density on grid of N points.
        """
        if gamma is None:           
            gamma = self.twodens.density()
        if rho is None:
            rho = self.onedens.density(mask=True)
        if gamma_lapl is None:        
            gamma_lapl = self.twodens.laplacian()
        if omega is not None:
            gamma_lapl = gamma_lapl*omega

        rho_outer = np.ma.masked_less(np.outer(rho,rho), 1.0e-30)
        rho_outer.filled(1.0e-30)
        ita_density = gamma_lapl*np.ma.log(gamma/np.outer(rho,rho))
        return ita_density 