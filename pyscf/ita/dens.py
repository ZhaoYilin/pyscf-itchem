import numpy as np
from pyscf import dft

__all__ = ["ElectronDensity", "PartitionDensity"]

class ElectronDensity(np.ndarray):    
    """Class to generate the electron density of the molecule at the desired points.
  
        Examples:

        >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
        >>> grids = dft.gen_grid.Grids(mol)
        >>> grids.build()
        >>> mf = scf.HF(mol)
        >>> mf.kernel()
        >>> ed = ita.dens.ElectronDenstiy.build(mf, grids)
        """
    def __new__(cls, rhos=None):
        """ Class to generate the electron density of the molecule at the desired points.

        Parameters
        ----------
        rhos : np.ndarray((M, N), dtype=float)
            2D array of shape (M, N) to store electron density. 
            For deriv=0, 2D array of (1,N) to store density;  
            For deriv=1, 2D array of (4,N) to store density and density derivatives 
            for x,y,z components; 
            For deriv=2, 2D array of (6,N) (with_lapl=True) array where last two rows 
            are nabla^2 rho and tau = 1/2(nabla f)^2 or (5,N) (with_lapl=False) where 
            the last row is tau = 1/2(nabla f)^2.

        Returns
        -------
        obj : ElectronDensity
            Instance of ElectronDensity class.
        """
        if rhos is None:             
            obj = np.asarray(0).view(cls)    
        else:    
            obj = np.asarray(rhos).view(cls)        
        return obj
         
    @classmethod
    def build(
        cls, 
        method, 
        grids_coords, 
        spin='ab', 
        deriv=2
    ):
        r""" Compute the electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.        
        grid_coords: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm')
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density,
            by default 'ab'.
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.

        Returns
        -------
        obj : ElectronDensity
            Instance of ElectronDensity class.
        """
        mol = method.mol
        rdm1 = method.make_rdm1()
        rdm1 = ElectronDensity.spin_reduction(np.array(rdm1), spin)

        aos = dft.numint.eval_ao(mol, grids_coords, deriv=deriv)
        if deriv==0:
            rhos = dft.numint.eval_rho(mol, aos, rdm1, xctype='LDA').reshape((1,-1))
        elif deriv==1:
            rhos = dft.numint.eval_rho(mol, aos, rdm1, xctype='GGA')      
        elif deriv==2:
            rhos = dft.numint.eval_rho(mol, aos, rdm1, xctype='mGGA')         
        else:
            raise ValueError('Level value not valid.')

        obj = cls(rhos)
        return obj

    @staticmethod 
    def spin_reduction(tensor, spin):
        """Unified the shape of mo_coeff to square matirx, mo_occ to vector 
        according to spin type. 

        Parameters
        ----------
        tensor : np.ndarray
            Tensor of mo_coeff or mo_occ.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density.

        Returns
        -------
        tensor : np.ndarray 
            Square matrix mo_coeff or vector mo_occ.
        """        
        if tensor.shape[0]==2 and len(tensor.shape)>1:
            if spin=='ab':
                tensor = tensor[0] + tensor[1]
            elif spin=='a':
                tensor = tensor[0]
            elif spin=='b':
                tensor = tensor[1]
            elif spin=='m':
                tensor = tensor[0] - tensor[1]
            else:
                raise ValueError("Value of spin not valid.")
        else:
            if spin=='ab':
                tensor = tensor
            elif spin=='a':
                tensor = tensor/2.0
            elif spin=='b':
                tensor = tensor/2.0
            elif spin=='m':
                tensor = np.zeros_like(tensor)
            else:
                raise ValueError("Value of spin not valid.")
        return tensor

    def density(self, mask=False, threshold=1.0e-30):
        r"""Electron density :math:`\rho\left(\mathbf{r}\right)`.
        
        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        rho : np.ndarray((N,), dtype=float)
            Electron density on grid of N points.        
        """
        if mask:
            rho = np.array(self[0])
            rho = np.ma.masked_less(rho, threshold)
            rho.filled(threshold)
        else:
            rho = np.array(self[0])
        return rho

    def gradient(self, mask=False, threshold=1.0e-30):
        r"""Gradient of electron density :math:`\nabla \rho\left(\mathbf{r}\right)`.

        This is the first-order partial derivatives of electron density w.r.t. coordinate
        :math:`\mathbf{r} = \left(x\mathbf{i}, y\mathbf{j}, z\mathbf{k}\right)`,

         .. math::
            \nabla\rho\left(\mathbf{r}\right) =
            \left(\frac{\partial}{\partial x}\mathbf{i}, \frac{\partial}{\partial y}\mathbf{j},
                  \frac{\partial}{\partial z}\mathbf{k}\right) \rho\left(\mathbf{r}\right)

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        rho_grad : np.ndarray((3,N), dtype=float)
        """
        rho_grad = np.array(self[1:4])
        if mask:
            rho_grad = np.ma.masked_less(rho_grad, threshold)
            rho_grad.filled(threshold)
        return rho_grad
            
    def gradient_norm(self, mask=False, threshold=1.0e-30):
        r"""Norm of the gradient of electron density.

        .. math::
           \lvert \nabla \rho\left(\mathbf{r}\right) \rvert = \sqrt{
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial x}\right)^2 +
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial y}\right)^2 +
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial z}\right)^2 }

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        grad_rho_norm : np.ndarray((N,), dtype=float)
        """
        grad_norm = np.linalg.norm(self.gradient(mask=mask,threshold=threshold), axis=0)
        return grad_norm

    def laplacian(self, mask=False, threshold=1.0e-30):
        r"""Laplacian of electron density :math:`\nabla ^2 \rho\left(\mathbf{r}\right)`.

        This is defined as the trace of Hessian matrix of electron density which is equal to
        the sum of its :math:`\left(\lambda_1, \lambda_2, \lambda_3\right)` eigen-values:

        .. math::
           \nabla^2 \rho\left(\mathbf{r}\right) = \nabla\cdot\nabla\rho\left(\mathbf{r}\right) =
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial x^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial y^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial z^2} =
                     \lambda_1 + \lambda_2 + \lambda_3

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        rho_lapl : np.ndarray((N,), dtype=float)
        """   
        rho_lapl = np.array(self[4])
        if mask:
            rho_lapl = np.ma.masked_less(rho_lapl, threshold)
            rho_lapl.filled(threshold)
        return rho_lapl
                     
    def tau(self, mask=False, threshold=1.0e-30):
        r"""Laplacian of electron density :math:`\nabla ^2 \rho\left(\mathbf{r}\right)`.

        This is defined as the trace of Hessian matrix of electron density which is equal to
        the sum of its :math:`\left(\lambda_1, \lambda_2, \lambda_3\right)` eigen-values:

        .. math::
           \nabla^2 \rho\left(\mathbf{r}\right) = \nabla\cdot\nabla\rho\left(\mathbf{r}\right) =
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial x^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial y^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial z^2} =
                     \lambda_1 + \lambda_2 + \lambda_3

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        rho_tau : np.ndarray((N,), dtype=float)
        """      
        rho_tau = np.array(self[5])
        if mask:
            rho_tau = np.ma.masked_less(rho_tau, threshold)
            rho_tau.filled(threshold)
        return rho_tau
              
class PartitionDensity(list):    
    """Class to generate the partition electron density of the molecule at the desired points.
  
        Examples:

        >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
        >>> grids = dft.gen_grid.Grids(mol)
        >>> grids.build()
        >>> mf = scf.HF(mol)
        >>> mf.kernel()
        >>> pd = ita.dens.PartitionDenstiy.orbital(mf, grids)
        """
    def __new__(cls, rhos=None):
        """ Class to generate the partition electron density of the molecule at the desired points.

        Parameters
        ----------
        rhos : np.ndarray((M, N), dtype=float)
            2D array of shape (M, N) to store electron density. 
            For deriv=0, 2D array of (1,N) to store density;  
            For deriv=1, 2D array of (4,N) to store density and density derivatives 
            for x,y,z components; 
            For deriv=2, 2D array of (6,N) (with_lapl=True) array where last two rows 
            are nabla^2 rho and tau = 1/2(nabla f)^2 or (5,N) (with_lapl=False) where 
            the last row is tau = 1/2(nabla f)^2.

        Returns
        -------
        obj : PartitionDensity
            Instance of PartitionDensity class.
        """
        if rhos is None:    
            obj = list().__new__(cls)        
        else:   
            obj = list().__new__(cls, rhos) 
        return obj

    @classmethod
    def orbital(
        cls, 
        method, 
        grids_coords, 
        spin='ab', 
        deriv=1
    ):
        r""" Compute the orbital electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.        
        grid_coords: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm')
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density,
            by default 'ab'.
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 1.

        Returns
        -------
        obj : PartitionDensity
            Instance of PartitionDensity class.
        """
        if 'mcscf' in str(method.__class__):
            from pyscf import mcscf
            noons, natorbs = mcscf.addons.make_natural_orbitals(method)
            method.mo_coeff = natorbs
            method.mo_occ = noons

        mol = method.mol
        rdm1 = method.make_rdm1()
        rdm1 = ElectronDensity.spin_reduction(np.array(rdm1), spin)
        mo_coeff = ElectronDensity.spin_reduction(np.array(method.mo_coeff), spin) 
        mo_occ = ElectronDensity.spin_reduction(np.array(method.mo_occ), spin)
        mo_coeff = mo_coeff[:,mo_occ>0]
        mo_occ = mo_occ[mo_occ>0]

        aos = dft.numint.eval_ao(mol, grids_coords, deriv=deriv)
        partition_density = []
        if deriv==0:                
            ao = aos
            for i, occ in enumerate(mo_occ):
                # Make One-Particle Reduced Density Matrix, 1-RDM
                Corb = mo_coeff[:,i].reshape((-1,1))
                orb_rdm1 = occ*np.einsum('ui,vi->uv', Corb, Corb)
                orb_rhos = np.einsum("gu, gv, uv -> g", ao, ao, orb_rdm1)

                ed = ElectronDensity(orb_rhos)
                partition_density.append(ed)

        elif deriv==1:
            ao = aos[0]
            ao_grad = aos[1:4]
            for i, occ in enumerate(mo_occ):
                # Make One-Particle Reduced Density Matrix, 1-RDM
                Corb = mo_coeff[:,i].reshape((-1,1))
                orb_rdm1 = occ*np.einsum('ui,vi->uv', Corb, Corb)
                orb_rho = np.einsum("gu, gv, uv -> g", ao, ao, orb_rdm1)
                orb_rho = orb_rho.reshape(1,-1)
                orb_rho_grad = 2 * np.einsum("rgu, gv, uv -> rg", ao_grad, ao, orb_rdm1)
                orb_rhos = np.concatenate([orb_rho, orb_rho_grad])

                ed = ElectronDensity(orb_rhos)
                partition_density.append(ed)

        elif deriv==2:
            ao = aos[0]
            ao_grad = aos[1:4]        
            ao_hess = np.array([
                [aos[4], aos[5], aos[6]],
                [aos[5], aos[7], aos[8]],
                [aos[6], aos[8], aos[9]],
            ])
            for i, occ in enumerate(mo_occ):
                Corb = mo_coeff[:,i].reshape((-1,1))
                orb_rdm1 = occ*np.einsum('ui,vi->uv', Corb, Corb) 
                orb_rho = np.einsum("gu, gv, uv -> g", ao, ao, orb_rdm1)
                orb_rho = orb_rho.reshape(1,-1)
                orb_rho_grad = 2 * np.einsum("rgu, gv, uv -> rg", ao_grad, ao, orb_rdm1)
                orb_rho_hess = (
                    + 2 * np.einsum("rwgu, gv, uv -> rwg", ao_hess, ao, orb_rdm1)
                    + 2 * np.einsum("rgu, wgv, uv -> rwg", ao_grad, ao_grad, orb_rdm1)
                )
                orb_rho_lapl = orb_rho_hess.trace().reshape((1,-1))
                orb_tau = 0.5 * np.einsum("rgu, wgv, uv -> rwg", ao_grad, ao_grad, orb_rdm1).trace().reshape((1,-1))
                orb_rhos = np.concatenate([orb_rho, orb_rho_grad, orb_rho_lapl, orb_tau])

                ed = ElectronDensity(orb_rhos)
                partition_density.append(ed)

        obj = cls(partition_density)       
        return obj

    def density(self, mask=False, threshold=1.0e-30):
        r"""Electron density :math:`\rho\left(\mathbf{r}\right)`.

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        rho : np.ndarray((N,), dtype=float)
        """
        if mask:
            rho = np.array(sum(self)[0])
            rho = np.ma.masked_less(rho, threshold)
            rho.filled(threshold)
        else:
            rho = np.array(self[0])
        return rho

    def gradient(self, mask=False, threshold=1.0e-30):
        r"""Gradient of electron density :math:`\nabla \rho\left(\mathbf{r}\right)`.

        This is the first-order partial derivatives of electron density w.r.t. coordinate
        :math:`\mathbf{r} = \left(x\mathbf{i}, y\mathbf{j}, z\mathbf{k}\right)`,

         .. math::
            \nabla\rho\left(\mathbf{r}\right) =
            \left(\frac{\partial}{\partial x}\mathbf{i}, \frac{\partial}{\partial y}\mathbf{j},
                  \frac{\partial}{\partial z}\mathbf{k}\right) \rho\left(\mathbf{r}\right)

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        rho_grad : np.ndarray((3,N), dtype=float)
        """
        rho_grad = np.array(sum(self)[1:4])
        if mask:
            rho_grad = np.ma.masked_less(rho_grad, threshold)
            rho_grad.filled(threshold)
        return rho_grad
            
    def gradient_norm(self, mask=False, threshold=1.0e-30):
        r"""Norm of the gradient of electron density.

        .. math::
           \lvert \nabla \rho\left(\mathbf{r}\right) \rvert = \sqrt{
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial x}\right)^2 +
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial y}\right)^2 +
                  \left(\frac{\partial\rho\left(\mathbf{r}\right)}{\partial z}\right)^2 }

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        grad_norm : np.ndarray((N,), dtype=float)
        """
        grad_norm = np.linalg.norm(self.gradient(mask=mask,fill=threshold), axis=0)
        return grad_norm

    def laplacian(self, mask=False, threshold=1.0e-30):
        r"""Laplacian of electron density :math:`\nabla ^2 \rho\left(\mathbf{r}\right)`.

        This is defined as the trace of Hessian matrix of electron density which is equal to
        the sum of its :math:`\left(\lambda_1, \lambda_2, \lambda_3\right)` eigen-values:

        .. math::
           \nabla^2 \rho\left(\mathbf{r}\right) = \nabla\cdot\nabla\rho\left(\mathbf{r}\right) =
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial x^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial y^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial z^2} =
                     \lambda_1 + \lambda_2 + \lambda_3

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        rho_lapl : np.ndarray((N,), dtype=float)
        """   
        rho_lapl = np.array(sum(self)[4])
        if mask:
            rho_lapl = np.ma.masked_less(rho_lapl, threshold)
            rho_lapl.filled(threshold)
        return rho_lapl
                     
    def tau(self, mask=False, threshold=1.0e-30):
        r"""Laplacian of electron density :math:`\nabla ^2 \rho\left(\mathbf{r}\right)`.

        This is defined as the trace of Hessian matrix of electron density which is equal to
        the sum of its :math:`\left(\lambda_1, \lambda_2, \lambda_3\right)` eigen-values:

        .. math::
           \nabla^2 \rho\left(\mathbf{r}\right) = \nabla\cdot\nabla\rho\left(\mathbf{r}\right) =
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial x^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial y^2} +
                     \frac{\partial^2\rho\left(\mathbf{r}\right)}{\partial z^2} =
                     \lambda_1 + \lambda_2 + \lambda_3

        Parameters
        ----------
        mask : Bool
            If mask the corresponding element of the associated array which is
            less than given the threshold.
        threshold : float
            Threshold for array element to mask or fill.

        Returns
        -------
        rho_tau : np.ndarray((N,), dtype=float)
        """      
        rho_tau = np.array(sum(self)[5])
        if mask:
            rho_tau = np.ma.masked_less(rho_tau, threshold)
            rho_tau.filled(threshold)
        return rho_tau