import numpy as np

from pyscf.dft.rks import KohnShamDFT
from pyscf.scf.hf import SCF

from pyscf.itchem.eval_dens import eval_rhos, eval_gammas


__all__ = ["ElectronDensity", "OneElectronDensity", "TwoElectronDensity"]

class ElectronDensity(np.ndarray):    
    """Class to generate the electron density of the molecule at the desired points.
    """
    def __new__(cls, rhos=None):
        """ Class to generate the electron density of the molecule at the desired points.

        Parameters
        ----------
        rhos : np.ndarray((M, Ngrids), dtype=float)
            2D array of shape (M, Ngrids) to store electron density. 
            For deriv=0, 2D array of (1,Ngrids) to store density;  
            For deriv=1, 2D array of (4,Ngrids) to store density and density derivatives 
            for x,y,z components;
            For deriv=2, 2D array of (5,Ngrids) array where last rows are nabla^2 rho.             
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
        rho : np.ndarray((Ngrids,), dtype=float)
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
        rho_grad : np.ndarray((3,Ngrids), dtype=float)
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
        grad_rho_norm : np.ndarray((Ngrids,), dtype=float)
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
        rho_lapl : np.ndarray((Ngrids,), dtype=float)
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
        >>> ed = OneElectronDenstiy.build(mf, grids_coords)
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
        grids_coords, 
        deriv=2
    ):
        r""" Compute the electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.   
        grids_coords : Grids
            Pyscf Grids instance.                 
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.

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

        rhos = eval_rhos(mol, grids_coords, rdm1, deriv=deriv)
        
        obj = cls(rhos)
        return obj

    @classmethod
    def orbital_partition(
        cls, 
        method, 
        grids_coords, 
        deriv=2
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

        Returns
        -------
        obj : PartitionDensity
            Instance of PartitionDensity class.
        """
        from pyscf import mcscf
        if hasattr(method, "_scf"):
            make_rdm1 = method._scf.to_rhf().make_rdm1       
        else:
            make_rdm1 = method.to_rhf().make_rdm1
        # spin traced nature orbital mo_occ and mo_coeff 
        noons, natorbs = mcscf.addons.make_natural_orbitals(method)
        mol = method.mol
        partition_density = []

        filter = noons>1e-8           
        natorbs = natorbs[:,filter]
        noons = noons[filter]
        for i in range(len(noons)):
            orb_rdm1 = make_rdm1(natorbs[:,[i]], noons[[i]])
            orb_rhos = eval_rhos(mol, grids_coords, orb_rdm1, deriv=deriv)   
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
            rdm = rdm[0] + rdm[1] + rdm[1].transpose(2,3,0,1) + rdm[2]
        
        return rdm
        
    @classmethod
    def build(
        cls, 
        method, 
        grids_coords, 
        deriv=2
    ):
        r""" Compute the electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.      
        grids_coords : [np.ndarray((Ngrids,3), dtype=float), np.ndarray((Ngrids,3), dtype=float)]
            List or tuple of grids coordinates.              
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.

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
        print(type(grids_coords))
        gammas = eval_gammas(mol, grids_coords, rdm2, deriv=deriv)

        obj = cls(gammas)
        return obj

    @classmethod
    def orbital_partition(
        cls, 
        method, 
        grids_coords, 
        deriv=2
    ):
        r""" Compute the orbital partition electron density of the molecule at the desired points.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.        
        grids_coords : [np.ndarray((Ngrids,3), dtype=float), np.ndarray((Ngrids,3), dtype=float)]
            List or tuple of grids coordinates.              
        deriv : int
            List of molecule and promolecule derivative+1 level, by default 2.

        Returns
        -------
        obj : PartitionDensity
            Instance of PartitionDensity class.
        """
        # This function is not verified.
        from pyscf import mcscf
        if hasattr(method, "_scf"):
            make_rdm2 = method._scf.to_rhf().make_rdm2       
        else:
            make_rdm2 = method.to_rhf().make_rdm2  
        # spin traced nature orbital mo_occ and mo_coeff 
        noons, natorbs = mcscf.addons.make_natural_orbitals(method)
        mol = method.mol
        partition_density = []

        filter = noons>1e-8           
        natorbs = natorbs[:,filter]
        noons = noons[filter]
        for i in range(len(noons)):
            orb_rdm2 = make_rdm2(natorbs[:,[i]], noons[[i]])
            orb_gammas = eval_gammas(mol, grids_coords, orb_rdm2, deriv=deriv)   
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
        rhos : np.ndarray((M, Ngrids), dtype=float)
            2D array of shape M, Ngrids) to store electron density. 
            For deriv=0, 2D array of (1,Ngrids) to store density;  
            For deriv=1, 2D array of (4,Ngrids) to store density and density derivatives 
            for x,y,z components; 
            For deriv=2, 2D array of (5,Ngrids) array where last rows are nabla^2 rho.             

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
        rho : np.ndarray((Ngrids,), dtype=float)
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
        rho_grad : np.ndarray((3,Ngrids), dtype=float)
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
        grad_norm : np.ndarray((Ngrids,), dtype=float)
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
        rho_lapl : np.ndarray((Ngrids,), dtype=float)
        """   
        rho_lapl = np.array(sum(self)[4])
        if mask:
            rho_lapl = np.ma.masked_less(rho_lapl, threshold)
            rho_lapl.filled(threshold)
        return rho_lapl