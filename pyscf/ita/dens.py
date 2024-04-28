import numpy as np

from pyscf.dft.rks import KohnShamDFT
from pyscf.scf.hf import SCF

from pyscf.ita.eval_dens import eval_rhos, eval_gammas


__all__ = ["ElectronDensity", "OneElectronDensity", "TwoElectronDensity"]

class ElectronDensity(np.ndarray):    
    """Class to generate the electron density of the molecule at the desired points.
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
            For deriv=2, 2D array of (5,N) array where last rows are nabla^2 rho.             
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
        rho = np.array(self[0])
        if mask:
            rho = np.ma.masked_less(rho, threshold)
            rho.filled(threshold)
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
                                      
class OneElectronDensity(ElectronDensity):    
    """Class to generate the electron density of the molecule at the desired points.
  
        Examples:

        >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
        >>> grids = dft.gen_grid.Grids(mol)
        >>> grids.build()
        >>> mf = scf.HF(mol)
        >>> mf.kernel()
        >>> ed = OneElectronDenstiy.build(mf, grids)
    """
    @staticmethod
    def spin_traced_rdm(rdm):
        """ Make sure the rdm is spin traced. 
 
        Parameters
        ----------
        rdm : Tuple(np.ndarray) or np.ndarray
            The rdm object.
 
        Returns
        -------
        rdm : np.ndarray 
            Spin traced rdm.
        """        
        # Convert a tuple/list of np.ndarray to np.ndarray
        rdm = np.array(rdm)
        if len(rdm.shape)==3:
            rdm = rdm[0] + rdm[1]
        
        return rdm
        
    @classmethod
    def build(
        cls, 
        method, 
        grids, 
        deriv=2,
        batch_mem = 100
    ):
        r""" Compute the electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.   
        grids : Grids
            Pyscf Grids instance.                 
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.
        batch_mem : int, optional
            Block memory in each loop, by default 100

        Returns
        -------
        obj : ElectronDensity
            Instance of ElectronDensity class.
        """

        mol = method.mol
        if isinstance(method,(SCF, KohnShamDFT)):
            # SCF/KSDFT default rdm in AO basis 
            rdm1 = method.make_rdm1()
        else:
            # Post-SCF default rdm in MO basis, here we transfer it to AO basis
            rdm1 = method.make_rdm1(ao_repr=True)
        rdm1 = cls.spin_traced_rdm(rdm1)

        rhos = eval_rhos(mol, grids, rdm1, deriv=deriv, batch_mem=batch_mem)
        
        obj = cls(rhos)
        return obj

    @classmethod
    def orbital_partition(
        cls, 
        method, 
        grids, 
        deriv=2,
        batch_mem = 100
    ):
        r""" Compute the orbital partition electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.        
        grids : Grids
            Pyscf Grids instance.                 
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.
        batch_mem : int, optional
            Block memory in each loop, by default 100

        Returns
        -------
        obj : PartitionDensity
            Instance of PartitionDensity class.
        """
        from pyscf import mcscf
        if hasattr(method, "_scf"):
            mf = method._scf       
        else:
            mf = method 
        noons, natorbs = mcscf.addons.make_natural_orbitals(method)
        method.mo_coeff = natorbs
        method.mo_occ = noons

        mol = method.mol
        mo_coeff = np.array(method.mo_coeff)
        mo_occ = np.array(method.mo_occ)
        partition_density = []

        # Make One-Particle Reduced Density Matrix, 1-RDM
        if len(np.array(mo_coeff).shape)==3:
            filter = list(mo_occ[0]>1e-8) or list(mo_occ[1]>1e-8)            
            mo_coeff = mo_coeff[:,:,filter]
            mo_occ = mo_occ[:,filter]
            for i in range(len(mo_occ[0])):
                orb_rdm1 = mf.make_rdm1(mo_coeff[:,:,[i]], mo_occ[:,[i]])
                orb_rdm1 = cls.spin_traced_rdm(orb_rdm1) 
                orb_rhos = eval_rhos(mol, grids, orb_rdm1, deriv=deriv, batch_mem=batch_mem)   
                ed = cls(orb_rhos)
                partition_density.append(ed)
        else:
            filter = mo_occ>1e-8           
            print(filter)
            mo_coeff = mo_coeff[:,filter]
            mo_occ = mo_occ[filter]
            for i in range(len(mo_occ)):
                orb_rdm1 = mf.make_rdm1(mo_coeff[:,[i]], mo_occ[[i]])
                orb_rdm1 = cls.spin_traced_rdm(orb_rdm1) 
                orb_rhos = eval_rhos(mol, grids, orb_rdm1, deriv=deriv, batch_mem=batch_mem)   
                ed = cls(orb_rhos)
                partition_density.append(ed)

        return PartitionDensity(partition_density)

class TwoElectronDensity(ElectronDensity):    
    """Class to generate the electron density of the molecule at the desired points.
  
        Examples:

        >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
        >>> grids = dft.gen_grid.Grids(mol)
        >>> grids.build()
        >>> mf = scf.HF(mol)
        >>> mf.kernel()
        >>> ed = TwoElectronDenstiy.build(mf, grids)
    """
    @staticmethod
    def spin_traced_rdm(rdm):
        """ Make sure the rdm is spin traced. 
 
        Parameters
        ----------
        rdm : Tuple(np.ndarray) or np.ndarray
            The rdm object.
 
        Returns
        -------
        rdm_st : np.ndarray 
            Spin traced rdm.
        """        
        # Convert a tuple/list of np.ndarray to np.ndarray
        rdm = np.array(rdm)

        if len(rdm.shape)==5:
            # spin rdm2
            rdm = rdm[0] + rdm[1] + rdm[1].transpose(2,3,0,1) + rdm[2]
        
        return rdm
        
    @classmethod
    def build(
        cls, 
        method, 
        grids, 
        deriv=2,
        batch_mem=5
    ):
        r""" Compute the electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.        
        grids : Grids
            Pyscf Grids instance.                 
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.
        batch_mem : int, optional
            Block memory in each loop, by default 5.

        Returns
        -------
        obj : ElectronDensity
            Instance of ElectronDensity class.
        """
        mol = method.mol
        if isinstance(method,(SCF, KohnShamDFT)):
            # SCF/KSDFT default rdm in AO basis 
            rdm2 = method.make_rdm2()
        else:
            # Post-SCF default rdm in MO basis, here we transfer it to AO basis
            rdm2 = method.make_rdm2(ao_repr=True)        
        rdm2 = cls.spin_traced_rdm(rdm2)

        gammas = eval_gammas(mol, grids, rdm2, deriv=deriv, batch_mem=batch_mem)

        obj = cls(gammas)
        return obj

    @classmethod
    def orbital_partition(
        cls, 
        method, 
        grids, 
        deriv=2,
        batch_mem = 5
    ):
        r""" Compute the orbital partition electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.        
        grids : Grids
            Pyscf Grids instance.                 
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.
        batch_mem : int, optional
            Block memory in each loop, by default 5.

        Returns
        -------
        obj : PartitionDensity
            Instance of PartitionDensity class.
        """
        from pyscf import mcscf
        if hasattr(method, "_scf"):
            mf = method._scf       
        else:
            mf = method 
        noons, natorbs = mcscf.addons.make_natural_orbitals(method)
        method.mo_coeff = natorbs
        method.mo_occ = noons

        mol = method.mol
        mo_coeff = np.array(method.mo_coeff)
        mo_occ = np.array(method.mo_occ)
        partition_density = []

        # Make One-Particle Reduced Density Matrix, 1-RDM
        if len(np.array(mo_coeff).shape)==3:
            filter = list(mo_occ[0]>1e-8) or list(mo_occ[1]>1e-8)            
            mo_coeff = mo_coeff[:,:,filter]
            mo_occ = mo_occ[:,filter]
            for i in range(len(mo_occ[0])):
                orb_rdm2 = mf.make_rdm2(mo_coeff[:,:,[i]], mo_occ[:,[i]])
                orb_rdm2 = cls.spin_traced_rdm(orb_rdm2) 
                orb_gammas = eval_gammas(mol, grids, orb_rdm2, deriv=deriv, batch_mem=batch_mem)   
                ed = cls(orb_gammas)
                partition_density.append(ed)
        else:
            filter = mo_occ>1e-8           
            mo_coeff = mo_coeff[:,filter]
            mo_occ = mo_occ[filter]
            for i in range(len(mo_occ)):
                orb_rdm2 = mf.make_rdm2(mo_coeff[:,[i]], mo_occ[[i]])
                orb_rdm2 = cls.spin_traced_rdm(orb_rdm2) 
                orb_gammas = eval_gammas(mol, grids, orb_rdm2, deriv=deriv, batch_mem=batch_mem)  
                ed = cls(orb_gammas)
                partition_density.append(ed)

        return PartitionDensity(partition_density)

class PartitionDensity(list):    
    """Class for the partition electron density of the molecule at the desired points.
        """
    def __new__(cls, rhos=None):
        """ Class for the partition electron density of the molecule at the desired points.

        Parameters
        ----------
        rhos : np.ndarray((M, N), dtype=float)
            2D array of shape (M, N) to store electron density. 
            For deriv=0, 2D array of (1,N) to store density;  
            For deriv=1, 2D array of (4,N) to store density and density derivatives 
            for x,y,z components; 
            For deriv=2, 2D array of (5,N) array where last rows are nabla^2 rho.             

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
        rho = np.array(sum(self)[0])
        if mask:
            rho = np.ma.masked_less(rho, threshold)
            rho.filled(threshold)
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