import numpy as np

class InformationTheoryDensity:
    def __init__(self, moldens=None, promoldens=None, molkeds=None, promolkeds=None):
        """_summary_

        Parameters
        ----------
        rho : _type_
            _description_
        """
        self.moldens = moldens 
        self.promoldens = promoldens
        self.molkeds = molkeds
        self.promolkeds = promolkeds

    @property
    def shannon_entropy(self):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """        
        rho = self.moldens.density
        kernel = rho*np.ma.log(rho)
        return kernel
    
    @property
    def relative_shannon_entropy(self):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """           
        rho = self.moldens.density
        prorho = self.promoldens.density

        rho = np.ma.masked_less(rho, 1.0e-30)
        rho.filled(1.0e-30)
        prorho = np.ma.masked_less(prorho, 1.0e-30)
        prorho.filled(1.0e-30)
        kernel = rho*np.ma.log(rho/prorho)        
        return kernel
    
    @property
    def fisher_information(self):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """           
        rho = self.moldens.density
        rho_grad = self.moldens.gradient

        rho = np.ma.masked_less(rho, 1.0e-30)
        rho.filled(1.0e-30)

        rho_grad_square = np.linalg.norm(rho_grad, axis=0)**2
        kernel = rho_grad_square/(rho)
        return kernel

    def alternative_fisher_information(self):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """            
        rho = self.moldens.density
        rho_lapl = self.moldens.lapl

        rho = np.ma.masked_less(rho, 1.0e-30)
        rho.filled(1.0e-30)

        kernel = rho_lapl*np.ma.log(rho)
        return kernel

    def rho_power(self, n):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """           
        rho = self.moldens.density

        kernel = rho**n
        return kernel

    def relative_renyi_entropy(self, n):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """           
        rho = self.moldens.density
        prorho = self.promoldens.density    

        prorho = np.ma.masked_less(prorho, 1.0e-30)
        prorho.filled(1.0e-30)

        kernel = rho**n/prorho**(n-1)
        return kernel

    def GBP_entropy(self, aos, mo_coeff, mo_occ):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """
        rho = self.moldens.density         
        #k = 3.166811563*1e-6
        k = 1
        cK = 0.3 * (3.0 * np.pi**2.0)**(2.0 / 3.0)
        c = (5/3) + np.log(4*np.pi*cK/3)

        ked = self.molkeds
        ts = ked.single_particle(aos, mo_coeff, mo_occ)        
        tTF = ked.thomas_fermi

        return 1.5*k*rho*(c+np.ma.log(ts/tTF))    