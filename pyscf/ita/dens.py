import numpy as np
from pyscf import dft

__all__ = ['Density']

class Density:
    """ Class to generate the electron density of the molecule at the desired points.
    """    
    def __init__(self, rhos=None):
        """ Class to generate the electron density of the molecule at the desired points.

        Parameters
        ----------
        rhos : List
            1D array of size N to store electron density if deriv=0; 2D array of (4,N) to 
            store density and “density derivatives” for x,y,z components if deriv=1; 
            For deriv=2, returns can be a (6,N) (with_lapl=True) array where last two rows 
            are nabla^2 rho and tau = 1/2(nabla f)^2 or (5,N) (with_lapl=False) where the last 
            row is tau = 1/2(nabla f)^2
        """     
        self.rhos = rhos

    @classmethod
    def molecule(cls, method, grids_coords, spin='ab', deriv=2):
        """ Compute the electron density of the molecule at the desired points.

        Parameters
        ----------
        grid_coords: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density.

        Returns
        -------
        promol_rho
            1D array of size N to store electron density if xctype = LDA;  2D array
            of (4,N) to store density and "density derivatives" for x,y,z components
            if xctype = GGA; For meta-GGA, returns can be a (6,N) (with_lapl=True)
            array where last two rows are \nabla^2 rho and tau = 1/2(\nabla f)^2
        """
        mol = method.mol
        rdm1 = method.make_rdm1()
        rdm1 = np.array(rdm1)

        if len(rdm1.shape)==3:
            if spin=='ab':
                rdm1 = rdm1[0] + rdm1[1]
            elif spin=='a':
                rdm1 = rdm1[0]
            elif spin=='b':
                rdm1 = rdm1[1]
            elif spin=='m':
                rdm1 = rdm1[0] - rdm1[1]
            else:
                raise ValueError("Value of spin not valid.")
            
        elif len(rdm1.shape)==2:
            if spin=='ab':
                rdm1 = rdm1
            elif spin=='a':
                rdm1 = rdm1/2.0
            elif spin=='b':
                rdm1 = rdm1/2.0
            elif spin=='m':
                rdm1 = np.zeros_like(rdm1)
            else:
                raise ValueError("Value of spin not valid.")

        # Build instance of ITA class
        aos = dft.numint.eval_ao(mol, grids_coords, deriv=deriv)
        if deriv==0:
            rhos = [dft.numint.eval_rho(mol, aos, rdm1, xctype='LDA')]
        elif deriv==1:
            rhos = dft.numint.eval_rho(mol, aos, rdm1, xctype='GGA')
        elif deriv==2:
            rhos = dft.numint.eval_rho(mol, aos, rdm1, xctype='mGGA')
        else:
            raise ValueError('Level value not valid.')

        obj = cls(rhos) 
        return obj

    @classmethod
    def promolecule(cls, method, grids_coords, charge, multiplicity, spin='ab', deriv=2):
        """ Compute the electron density of the promolecule at the desired points.

        Parameters
        ----------
        grid_coords: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density.

        Returns
        -------
        promol_rho
            1D array of size N to store electron density if xctype = LDA;  2D array
            of (4,N) to store density and "density derivatives" for x,y,z components
            if xctype = GGA; For meta-GGA, returns can be a (6,N) (with_lapl=True)
            array where last two rows are \nabla^2 rho and tau = 1/2(\nabla f)^2
        """
        from pyscf.ita.promolecule import ProMolecule
        mol = method.mol
        promethods = ProMolecule.build(method, charge, multiplicity)

        if deriv ==0:
            promol_rhos = []
        elif deriv ==1:
            promol_rhos = []
        elif deriv ==2:
            promol_rhos = []
        else:
            raise ValueError('Level value not valid.')

        for atom_geom in mol._atom:
            symbol = atom_geom[0]
            promethods[symbol].mol.set_geom_([atom_geom],unit='B')
            rdm1 = promethods[symbol].make_rdm1()
            rdm1 = np.array(rdm1)
            if len(rdm1.shape)==3:
                if spin=='ab':
                    rdm1 = rdm1[0] + rdm1[1]
                elif spin=='a':
                    rdm1 = rdm1[0]
                elif spin=='b':
                    rdm1 = rdm1[1]
                elif spin=='m':
                    rdm1 = rdm1[0] - rdm1[1]
                else:
                    raise ValueError("Value of spin not valid.")
            elif len(rdm1.shape)==2:
                if spin=='ab':
                    rdm1 = rdm1
                elif spin=='a':
                    rdm1 = rdm1/2.0
                elif spin=='b':
                    rdm1 = rdm1/2.0
                elif spin=='m':
                    rdm1 = np.zeros_like(rdm1)
                else:
                    raise ValueError("Value of spin not valid.")

            # Build instance of ITA class
            aos = dft.numint.eval_ao(promethods[symbol].mol, grids_coords, deriv=deriv)
            if deriv==0:
                promol_rhos.append(dft.numint.eval_rho(promethods[symbol].mol, aos, rdm1, xctype='LDA'))
            elif deriv==1:
                promol_rhos.append(dft.numint.eval_rho(promethods[symbol].mol, aos, rdm1, xctype='LDA'))
            elif deriv==2:
                promol_rhos.append(dft.numint.eval_rho(promethods[symbol].mol, aos, rdm1, xctype='LDA'))
            else:
                raise ValueError('Level value not valid.')
                
        obj = cls(promol_rhos)
        return obj

    @property
    def density(self):
        r"""Electron density :math:`\rho\left(\mathbf{r}\right)`."""
        
        return self.rhos[0]

    @property
    def gradient(self):
        r"""Gradient of electron density :math:`\nabla \rho\left(\mathbf{r}\right)`.

        This is the first-order partial derivatives of electron density w.r.t. coordinate
        :math:`\mathbf{r} = \left(x\mathbf{i}, y\mathbf{j}, z\mathbf{k}\right)`,

         .. math::
            \nabla\rho\left(\mathbf{r}\right) =
            \left(\frac{\partial}{\partial x}\mathbf{i}, \frac{\partial}{\partial y}\mathbf{j},
                  \frac{\partial}{\partial z}\mathbf{k}\right) \rho\left(\mathbf{r}\right)
        """
        return self.rhos[1:4]

    @property
    def gradient_norm(self):
        r"""Norm of the gradient of electron density.

        .. math::
           \lvert \nabla \rho\left(\mathbf{r}\right) \rvert = \sqrt{
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial x}\right)^2 +
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial y}\right)^2 +
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial z}\right)^2 }
        """
        norm = np.linalg.norm(self.gradient, axis=1)
        return norm

    @property
    def laplacian(self):
        r"""Laplacian of electron density :math:`\nabla ^2 \rho\left(\mathbf{r}\right)`.

        This is defined as the trace of Hessian matrix of electron density which is equal to
        the sum of its :math:`\left(\lambda_1, \lambda_2, \lambda_3\right)` eigen-values:

        .. math::
           \nabla^2 \rho\left(\mathbf{r}\right) = \nabla\cdot\nabla\rho\left(\mathbf{r}\right) =
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial x^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial y^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial z^2} =
                     \lambda_1 + \lambda_2 + \lambda_3
        """        
        return self.rhos[5]
    
    @property
    def tau(self):
        r"""Laplacian of electron density :math:`\nabla ^2 \rho\left(\mathbf{r}\right)`.

        This is defined as the trace of Hessian matrix of electron density which is equal to
        the sum of its :math:`\left(\lambda_1, \lambda_2, \lambda_3\right)` eigen-values:

        .. math::
           \nabla^2 \rho\left(\mathbf{r}\right) = \nabla\cdot\nabla\rho\left(\mathbf{r}\right) =
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial x^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial y^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial z^2} =
                     \lambda_1 + \lambda_2 + \lambda_3
        """        
        return self.rhos[6]    

def eval_rho(aos, rdm1, deriv, with_lapl=True):
    """ This is a rewrite of Pyscf eval_rho function that calculate the electron density and density derivatives.

    Parameters
    ----------
    aos : 2D array of shape (N,nao) for deriv=0, 3D array of shape (4,N,nao) for deriv=1.
        N is the number of grids, nao is the number of AO functions. If deriv=2, 
        ao[0] is AO value and ao[1:3] are the AO gradients. ao[4:10] are second 
        derivatives of ao values if applicable.    
    rdm1 : 2D array
        Density matrix
    deriv : int
        Order of density derivatives, it affects the shape of the return density.

    Returns
    -------
    List
        1D array of size N to store electron density if deriv=0; 2D array of (4,N) to 
        store density and “density derivatives” for x,y,z components if deriv=1; 
        For deriv=2, returns can be a (6,N) (with_lapl=True) array where last two rows 
        are nabla^2 rho and tau = 1/2(nabla f)^2 or (5,N) (with_lapl=False) where the last 
        row is tau = 1/2(nabla f)^2
    """
    if deriv==0:
        ao = aos
        rho = np.einsum("gu, gv, uv -> g", ao, ao, rdm1)
        return rho

    elif deriv==1:
        ao = aos[0]
        ao_grad = aos[1:4]

        rho = np.einsum("gu, gv, uv -> g", ao, ao, rdm1)
        rho_grad = 2 * np.einsum("rgu, gv, uv -> rg", ao_grad, ao, rdm1)
        return [rho, *rho_grad]

    elif deriv==2:
        ao = aos[0]
        ao_grad = aos[1:4]        
        ao_hess = np.array([
            [aos[4], aos[5], aos[6]],
            [aos[5], aos[7], aos[8]],
            [aos[6], aos[8], aos[9]],
        ])  

        rho = np.einsum("gu, gv, uv -> g", ao, ao, rdm1)
        rho_grad = 2 * np.einsum("rgu, gv, uv -> rg", ao_grad, ao, rdm1)
        rho_hess = (
            + 2 * np.einsum("rwgu, gv, uv -> rwg", ao_hess, ao, rdm1)
            + 2 * np.einsum("rgu, wgv, uv -> rwg", ao_grad, ao_grad, rdm1)
        )
        rho_lapl = rho_hess.trace()
        tau = 2 * np.einsum("rgu, wgv, uv -> rwg", ao_grad, ao_grad, rdm1)

        if with_lapl==True:
            return [rho, *rho_grad, rho_lapl, tau]   
        else:   
            return [rho, *rho_grad, tau]   