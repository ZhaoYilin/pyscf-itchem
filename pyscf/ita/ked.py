import numpy as np

class KineticEnergyDensity:
    def __init__(self, moldens=None):
        """_summary_

        Parameters
        ----------
        rhos : _type_, optional
            _description_, by default None
        """
        self.moldens = moldens  

    @property    
    def positive_definite(self):
        rho = self.moldens.density
        tau = rho
        return tau

    def general(self, a):
        rho = self.moldens.density
        rho_lapl = self.moldens.laplacian

        tau = rho + 1.0/4.0*(a-1)*rho_lapl
        return tau

    @property
    def thomas_fermi(self):
        """Thomas-Fermi kinetic energy density.

        .. math::
            \tau_\text{TF} \left(\mathbf{r}\right) = \tfrac{3}{10} \left(6 \pi^2 \right)^{2/3}
                \left(\frac{\rho\left(\mathbf{r}\right)}{2}\right)^{5/3}
        """
        rho = self.moldens.density

        c_tf = 0.3 * (3.0 * np.pi**2.0)**(2.0 / 3.0)
        ked = c_tf * rho ** (5.0 / 3.0)
        return ked

    @property
    def dirac(self):
        """_summary_

        Parameters
        ----------
        rho : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """      
        rho = self.moldens.density

        c_x = 3.0/4.0*(3.0/np.pi)**(1.0/3.0) 
        ked = -c_x * rho ** (4.0 / 3.0)
        return ked

    @property
    def weizsacker(self, rho, rho_grad):    
        """Weizsacker kinetic energy density
        """
        rho = self.moldens.density
        rho_grad = self.moldens.gradient

        rho = np.ma.masked_less(rho, 1.0e-30)
        rho.filled(1.0e-30)

        rho_grad_square = np.linalg.norm(rho_grad, axis=0)**2
        ked = 0.125*rho_grad_square/(rho)    
        return ked
    
    @property
    def gradient_expansion(self, rho, rho_grad, rho_lap):
        """_summary_

        Parameters
        ----------
        rho : _type_
            _description_
        rho_grad : _type_
            _description_
        rho_lap : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        rho_lapl = self.moldens.laplacian             
        tau_tf =  self.thomas_fermi
        tau_w = self.weizsacker
        ked = tau_tf + 1.0/9.0*tau_w + 1.0/6.0 * rho_lapl
        return ked

    def single_particle(self, aos, mo_coeff, mo_occ, spin='ab'):
        """ This is a rewrite of Pyscf eval_rho function that calculate the electron density and density derivatives.
 
        Parameters
        ----------
        ao : 2D array of shape (N,1) for deriv=0, 3D array of shape (4,N,1) for deriv=1.
            N is the number of grids. 
        """
        rho_lapl = self.moldens.laplacian             

        ao = aos[0]
        ao_grad = aos[1:4]
        term1 = np.zeros(ao.shape[:-1])

        if len(mo_coeff.shape)==3:
            if spin=='ab':
                mo_coeff = mo_coeff[0] + mo_coeff[1]
                mo_occ = mo_occ[0] + mo_occ[1]
            elif spin=='a':
                mo_coeff = mo_coeff[0]
                mo_occ = mo_occ[0]
            elif spin=='b':
                mo_coeff = mo_coeff[1]
                mo_occ = mo_occ[1]
            elif spin=='m':
                mo_coeff = mo_coeff[0] - mo_coeff[1]
                mo_occ = mo_occ[0] - mo_occ[1]
            else:
                raise ValueError("Value of spin not valid.")
        elif len(mo_coeff.shape)==2:
            if spin=='ab':
                pass
            elif spin=='a' or spin=='b':
                mo_coeff = mo_coeff/2.0
                mo_occ = mo_occ/2.0
            elif spin=='m':
                mo_coeff = np.zeros_like(mo_coeff)
                mo_occ = np.zeros_list(mo_occ)
            else:
                raise ValueError("Value of spin not valid.")

        for i, occ in enumerate(mo_occ):
            # Make One-Particle Reduced Density Matrix, 1-RDM
            Corb = mo_coeff[:,i].reshape((-1,1))
            rdm1_orb = occ*np.einsum('ui,vi->uv', Corb, Corb)

            rho_orb = np.einsum("gu, gv, uv -> g", ao, ao, rdm1_orb)
            rho_orb = np.ma.masked_less(rho_orb, 1.0e-30)
            rho_orb.filled(1.0e-30)   

            rho_grad_orb = 2 * np.einsum("rgu, gv, uv -> rg", ao_grad, ao, rdm1_orb)
            rho_grad_orb2 = np.linalg.norm(rho_grad_orb, axis=0)**2

            term1 += rho_grad_orb2/rho_orb

        term1 *= 0.125

        ked = term1 - 0.125*rho_lapl
        return ked
