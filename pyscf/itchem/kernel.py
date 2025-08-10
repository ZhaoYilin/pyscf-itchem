import numpy as np
from functools import partial

from pyscf.itchem.aim import Hirshfeld, Becke
from pyscf.itchem.eval_odm import eval_odm1, eval_odm2
from pyscf.itchem.scope import Global
from pyscf.itchem.utils.constants import ITA_DICT

__all__ = ["ITA", "OrbitalEntanglement"]

def aaa(ita, ita_name, **kwargs):
    ita_func = getattr(ita, ita_name)
    ita_func = partial(ita_func, **kwargs)
    if ita_name in ['renyi_entropy','tsallis_entropy','onicescu_information']:
        itad_func = getattr(ita.itad, "rho_power")
    else:
        itad_func = getattr(ita.itad, ita_name)
        itad_func = partial(itad_func, **kwargs)     
    if ita.aim:
        omegas = ita.omegas
        result = []
        for atom_id, omega_i in enumerate(omegas):            
            if ita_name in ['fisher_information','G3']:
                prorho_i = ita.itad.prodens[atom_id].density(mask=True)
                prorho_grad_i = ita.itad.prodens[atom_id].gradient()
                itad_i = itad_func(omega=omega_i, prorho=prorho_i, prorho_grad=prorho_grad_i)                          
            else:
                itad_i = itad_func(omega=omega_i)
            atomic_partition = ita_func(ita_density=itad_i, **kwargs)
            result.append(atomic_partition)
        if ita.partition is None:
            result = sum(result) 
    else:
        if ita.partition:
            omegas = ita.omegas
            result = []
            for omega_i in omegas:            
                itad_i = itad_func(omega=omega_i)
                atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                result.append(atomic_partition)
        else:
            result = ita_func()   
    return result    


class ITA(Global):
    r"""Information-Theoretic Approch (ITA) class.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> mf = scf.HF(mol)
    >>> mf.kernel()
    >>> grids = dft.Grids(mol)
    >>> grids.build()    
    >>> ita = ITA()
    >>> ita.method = mf
    >>> ita.grids = grids
    >>> ita.category = 'individual'    
    >>> ita.build()        
    """
    def __init__(
        self, 
        method=None, 
        grids=None,
        rung=3, 
        normalize = None,
        category='individual',
        aim=None,
        partition=None,
        promolecule=None
    ):
        r"""Initialize a instance.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
        grids : Grids
            Pyscf Grids instance.
        rung : int, optional
            Density derivate+1 level, by default 3.
        normalize : bool, optional 
            If the electron density normalized, by default None.              
        category : ('individual' | 'relative' | 'joint' | 'conditional' | 'mutual'), optional
            Type of ITA, by default 'individual'.               
        partition : (None | 'hirshfeld' | 'bader' | 'becke'), optional
            Atoms in molecule partition method, by default None.                      
        promolecule : ProMolecule, optional
            ProMolecule instance, by default None.
        """
        self.method = method
        self.grids = grids  
        self.rung = rung
        self.normalize = normalize
        self.categrory = category
        self.aim = aim
        self.partition = partition
        self.promolecule = promolecule
        
    def build(
        self, 
        method=None, 
        grids=None,
        rung=None, 
        normalize = None,
        category=None,
        partition=None,
        aim=None,
        promolecule=None,
    ):
        r"""Setup ITA and initialize some control parameters. Whenever you
        change the value of the attributes of :class:`ITA`, you need call
        this function to refresh the internal data of ITA.

        Parameters
        ----------
        method : PyscfMethod, optional
            Pyscf scf method or post-scf method instance, by deafult None.
        grids : Grids, optional
            Pyscf Grids instance, by deafult None.
        rung : int
            Density derivate+1 level, by default 3,.    
        normalize : bool, optional 
            If the electron density normalized, by default None.                      
        category : ('local' | 'joint' | 'conditional' | 'relative' | 'mutual'), optional
            Type of ITA, by default 'individual'.   
        representation : ('electron density' | 'shape function' | 'atoms in molecules')
            Type of representation, by default 'electron density'.            
        partition : (None | 'hirshfeld' | 'bader' | 'becke'), optional
            Atoms in molecule partition method, by default None.                      
        promolecule : ProMolecule, optional
            ProMolecule instance, by default None.
        """
        if method is None: method = self.method 
        else: self.method = method
        if grids is None: grids = self.grids
        else: self.grids = grids
        if rung is None: rung = self.rung
        else: self.rung = rung
        if normalize is None: normalize = self.normalize
        else: self.normalize = normalize
        if category is None: category = self.category
        else: self.category = category
        if partition is None: partition = self.partition
        else: self.partition = partition
        if aim is None: aim = self.aim
        else: self.aim = aim
        if promolecule is None: promolecule = self.promolecule
        else: self.promolecule = promolecule

        if (partition=='hirshfeld' or category=='relative') and promolecule is None:
            from pyscf.itchem.promolecule import ProMolecule
            promolecule = ProMolecule(method).build()
            self.promolecule = promolecule

        # Build molecule electron density.
        if category=='individual':
            from pyscf.itchem.itad import ItaDensity
            itad = ItaDensity(method, grids.coords, rung, normalize)
        elif category=='relative':
            from pyscf.itchem.itad import RelativeItaDensity
            itad = RelativeItaDensity(method, grids.coords, rung, normalize, promolecule)
        elif category=='joint':
            from pyscf.itchem.itad import JointItaDensity
            itad = JointItaDensity(method, grids.coords, rung, normalize)
        elif category=='conditional':
            from pyscf.itchem.itad import ConditionalItaDensity
            itad = ConditionalItaDensity(method, grids.coords, rung, normalize)
        elif category=='mutual':
            from pyscf.itchem.itad import MutualItaDensity
            itad = MutualItaDensity(method, grids.coords, rung, normalize)
        else:
            raise ValueError("Not a valid category.")
        
        itad.build()              
        self.itad = itad

        # Build atoms-in-molecules
        if partition is not None:
            if partition is not None:
                if partition.lower()=='hirshfeld':                
                    prodens = promolecule.one_electron_density(grids.coords, deriv=0)          
                    omegas = Hirshfeld(prodens).sharing_function()  
                elif partition.lower()=='becke':
                    raise NotImplemented
                elif partition.lower()=='bader':
                    raise NotImplemented
                else:
                    raise ValueError("Not a valid partition.")                
            self.omegas = omegas

        return self

    def compute(self, code_name, partition=False, **kwargs):
        """Compute single ITA quantity.

        Parameters
        ----------
        code_name : int or str
            Ita code or name.
        partition : Bool
            If partition the result.
        """
        if code_name in ITA_DICT.keys():         
            ita_name = ITA_DICT[code_name]  
        elif code_name in ITA_DICT.values():
            ita_name = code_name
        else:
            raise ValueError("Not a valid ita code or name.")

        ita_func = getattr(self, ita_name)
        ita_func = partial(ita_func, **kwargs)
        if ita_name in ['renyi_entropy','tsallis_entropy','onicescu_information']:
            itad_func = getattr(self.itad, "rho_power")
        else:
            itad_func = getattr(self.itad, ita_name)
            itad_func = partial(itad_func, **kwargs) 

        if self.aim:
            omegas = self.omegas
            result = []
            for atom_id, omega_i in enumerate(omegas):            
                if ita_name in ['fisher_information','G3']:
                    prorho_i = self.itad.prodens[atom_id].density(mask=True)
                    prorho_grad_i = self.itad.prodens[atom_id].gradient()
                    itad_i = itad_func(omega=omega_i, prorho=prorho_i, prorho_grad=prorho_grad_i)                          
                else:
                    itad_i = itad_func(omega=omega_i)
                atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                result.append(atomic_partition)
            if partition is None:
                result = sum(result) 
        else:
            if partition:
                omegas = self.omegas
                result = []
                for omega_i in omegas:            
                    itad_i = itad_func(omega=omega_i)
                    atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                    result.append(atomic_partition)
            else:
                result = ita_func()   
        return result

    def batch_compute(
        self, 
        ita_code=[], 
        filename = 'pyita.log',
    ):
        r"""ITA batch calcuation.
 
        Parameters
        ----------
        ita : ITA
            Instance of ITA class.
        ita_code : List[int]
            List of ITA code to calculate.
        filename : str, optional
            File path and name of output, by default 'pyita.log'
        """
        from pyscf.itchem.utils.script import batch_compute
        return batch_compute(self, ita_code, filename)

    def rho_power(
        self, 
        n = 2,
        grids_weights=None, 
        ita_density=None
    ):
        r"""Electron density of power n defined as :math:`\int \rho(\mathbf{r})^n d\mathbf{r}`.

        Parameters
        ----------
        n : int, optional
            Order of rho power, by default 2.
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ita_density is None:
            ita_density = self.itad.rho_power(n=n)

        if len(ita_density.shape)==1:
            if self.partition:
                omegas = self.omegas
                partition_result = []
                for omega_i in omegas:           
                    if self.aim: 
                        itad_i = self.itad.rho_power(n=n,omega=omega_i)
                    else:
                        itad_i = self.itad.rho_power(n=n)*omega_i
                    atomic_partition = float(np.einsum("g, g -> ", grids_weights, itad_i))
                    partition_result.append(atomic_partition) 
                result = np.einsum("g, g -> ", grids_weights, ita_density)
                return result, partition_result
            else:           
                result = np.einsum("g, g -> ", grids_weights, ita_density)
            return result

        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
            return result

    def shannon_entropy(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Shannon entropy :math:`S_S` defined as:

        .. math::
            S_S = -\int \rho(\mathbf{r}) \ln \rho(\mathbf{r}) dr   

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ita_density is None:
            ita_density = self.itad.shannon_entropy()
                
        if len(ita_density.shape)==1:
            if self.partition:
                omegas = self.omegas
                partition_result = []
                for omega_i in omegas:           
                    if self.aim: 
                        itad_i = self.itad.shannon_entropy(omega=omega_i)
                    else:
                        itad_i = self.itad.shannon_entropy()*omega_i
                    atomic_partition = float(np.einsum("g, g -> ", grids_weights, itad_i))
                    partition_result.append(atomic_partition) 
                result = np.einsum("g, g -> ", grids_weights, ita_density)
                return result, partition_result                    
            else:           
                result = np.einsum("g, g -> ", grids_weights, ita_density)
                return result
            
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
            return result

    def fisher_information(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Fisher information :math:`I_F` defined as:

        .. math::
            I_F = \int \frac{|\nabla \rho(\mathbf{r})|^2}{\rho(\mathbf{r})} d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.fisher_information()

        if self.category=='conditional':
            term1 = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density[0])
            term2 = np.einsum("g, g -> ", grids_weights, ita_density[1])
            result = term1-term2
        else:
            if len(ita_density.shape)==1:
                result = np.einsum("g, g -> ", grids_weights, ita_density)
            if len(ita_density.shape)==2:
                result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
        return result

    def alternative_fisher_information(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Alternative Fisher information :math:`I^{\prime}_F` defined as:

        .. math::
            I^{\prime}_F = -\int \nabla^2 \rho(\mathbf{r}) \ln \rho(\mathbf{r}) d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """            
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.alternative_fisher_information()

        if len(ita_density.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ita_density)
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
        return result
    
    def GBP_entropy(
        self, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Ghosh-Berkowitz-Parr (GBP) entropy :math:`S_{GBP}` defined as:

        .. math::
            S_{GBP} = \int \frac{3}{2}k\rho(\mathbf{r}) 
                \left[ c+\ln \frac{t(\mathbf{r};\rho)}{t_{TF}(\mathbf{r};\rho)} \right] d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.GBP_entropy()

        if len(ita_density.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ita_density)
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
        return result 

    def renyi_entropy(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Renyi entropy :math:`R_n` defined as:

        .. math::
            R_n = \frac{1}{1-n} \ln \left[ \int \rho(\mathbf{r})^n d\mathbf{r} \right]

        Parameters
        ----------
        n : int, optional
            Order of the Renyi entropy, by default 2.
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """ 
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.rho_power(n)
            
        if len(ita_density.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ita_density)
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
        if self.category in ('relative','mutual') :
            result = (1/(n-1))*np.ma.log10(result)
        else:
            result = (1/(1-n))*np.ma.log10(result)
        return result
    
    def tsallis_entropy(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Tsallis entropy :math:`T_n` defined as:

        .. math::
            T_n = \frac{1}{n-1} \left[ 1- \int \rho(\mathbf{r})^n d\mathbf{r} \right]

        Parameters
        ----------
        n : int, optional
            Order of the Tsallis entropy, by default 2.
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.rho_power(n)

        if len(ita_density.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ita_density)
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)

        if self.category in ('relative','mutual') :
            result = (1/(1-n))*(1-result)
        else:
            result = (1/(n-1))*(1-result)
        return result 
    
    def onicescu_information(
        self, 
        n=2, 
        grids_weights=None, 
        ita_density=None
    ):
        r"""Onicescu information :math:`E_n` defined as:

        .. math::
            E_n = \frac{1}{n-1} \int \rho(\mathbf{r})^n d\mathbf{r}
         
        Parameters
        ----------
        n : int, optional
            Order of the Onicescu information, by default 2.
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.
        
        Returns
        -------
        result : float
            Scalar ITA result.
        """           
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.rho_power(n)

        if len(ita_density.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ita_density)
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)

        if self.category in ('relative','mutual') :
            result = (1/(1-n))*(1-result)
        else:
            result = (1/(n-1))*(1-result)            
        return result
     
    def G1(
        self, 
        grids_weights=None, 
        ita_density=None            
    ):
        r"""G1 density defined as:

        .. math::
            G_1(\mathbf{r}) = \sum_A \int
            \nabla^2 \rho_A(\mathbf{r}) 
            \ln \frac{\rho_A(\mathbf{r})}{\rho^0_A(\mathbf{r})} 
            d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.G1()

        if len(ita_density.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ita_density)
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
        return result        

    def G2(
        self, 
        grids_weights=None, 
        ita_density=None            
    ):
        r"""G2 density defined as:

        .. math::
            G_2(\mathbf{r}) = \sum_A \int 
            \rho_A(\mathbf{r}) \left[ 
            \frac{\nabla^2 \rho_A(\mathbf{r})}{\rho_A(\mathbf{r})} 
            -\frac{\nabla^2 \rho^0_A(\mathbf{r})}{\rho^0_A(\mathbf{r})}
            \right] d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.G2()

        if len(ita_density.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ita_density)
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
        return result    

    def G3(
        self, 
        grids_weights=None, 
        ita_density=None            
    ):
        r"""G3 density defined as:

        .. math::
            G_3(\mathbf{r})
            = \sum_A \int 
            \rho_A(\mathbf{r}) \left\vert 
            \frac{\nabla \rho_A(\mathbf{r})}{\rho_A(\mathbf{r})} 
            -\frac{\nabla \rho^0_A(\mathbf{r})}{\rho^0_A(\mathbf{r})} 
            \right\vert^2 d\mathbf{r}

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((Ngrids,), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar ITA result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights
        if ita_density is None:
            ita_density = self.itad.G3()

        if len(ita_density.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ita_density)
        if len(ita_density.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ita_density)
        return result            
    
class OrbitalEntanglement(Global):
    r"""Orbital entanglement class.
    """
    def __init__(self, odm1=None, odm2=None):
        """ Initialize a instance.

        Parameters
        ----------
        odm1 : np.ndarray, optional
            The one orbital reduced density matrix, by default None.
        odm2 : np.ndarray, optional
            The two orbital reduced density matrix, by default None.
        """        
        self.odm1 = odm1
        self.odm2 = odm2

    @property
    def no(self):
        """Number of orbitals.
        """        
        return self.odm1.shape[0]

    def build(self, method=None, seniority_zero=True):
        r"""Method to build the class.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
        seniority_zero : bool, optional
            If the electronic wavefunction is restricted to the seniority-zero, 
            by default True.

        Returns
        -------
        self : OrbitalEntanglement 
            OrbitalEntanglement instance.
        """        
        rdm1 = method.make_rdm1()
        rdm2 = method.make_rdm2()

        if len(np.array(rdm1))==2:
            rdm1s = (0.5*rdm1)*2
            rdm2s = (0.25*rdm2)*2
        else:
            rdm1s = rdm1
            rdm2s = rdm2

        if seniority_zero:
            odm1 = []
            for i in range(method.mol.nao):
                odm1_i = eval_odm1(i,rdm1s,rdm2s,seniority_zero=seniority_zero)
                odm1.append(odm1_i)
            odm2 = []
            for i in range(method.mol.nao):
                odm2_i = []
                for j in range(method.mol.nao):
                    odm2_ij = eval_odm2(i,j,rdm1s,rdm2s,rdm3s=None,rdm4s=None,seniority_zero=seniority_zero)
                    odm2_i.append(odm2_ij)
                odm2.append(odm2_i)
            
        else:
            raise NotImplementedError

        self = self(np.array(odm1), np.array(odm2))
        return self

    def one_orbital_entropy(self, odm1=None):
        r"""The entanglement entropy of one orbital defined as:

        .. math::
            s(1)_i = -\sum_{\alpha=1}^{4} \omega_{\alpha,i} \ln \omega_{\alpha,i}

        Parameters
        ----------
        odm1 : np.ndarray, optional
            The one orbital reduced density matrix, by default None.

        Returns
        -------
        s1 : List
            List of one orbital entropy.
        """        
        if odm1 is None:
            odm1 = self.odm1
        no = self.no
        s1 = np.zeros((no))
        for i in range(no):
            eigenvalues, _ = np.linalg.eig(odm1[i,:,:])
            s1[i] = -sum([sigma * np.ma.log(sigma) for sigma in eigenvalues])
        return s1

    def two_orbital_entropy(self, odm2=None):
        r"""The entanglement entropy of two orbital defined as:

        .. math::
            s(2)_{ij} = -\sum_{\alpha=1}^{16} \omega_{\alpha,i,j} \ln \omega_{\alpha,i,j}

        Parameters
        ----------
        odm2 : np.ndarray, optional
            The two orbital reduced density matrix, by default None.
        """        
        if odm2 is None:
            odm2 = self.odm2
        no = self.no
        s2 = np.zeros((no,no))
        for i in range(no):
            for j in range(no):
                eigenvalues, _ = np.linalg.eig(odm2[i,j,:,:])
                s2[i,j] = -sum([sigma * np.ma.log(sigma) for sigma in eigenvalues])
        return s2
    
    def mututal_information(self, s1=None, s2=None):
        r"""The orbital-pair mutual information defined as:

        .. math::
            I_{i|j} = \frac{1}{2} \left(s(2)_{i,j}-s(1)_{i}-s(1)_{j} \right)
                \left(1- \delta_{i,j} \right)
        """        
        if s1 is None:
            s1 = self.s1
        if s2 is None:
            s2 = self.s2
        no = self.no
        I = np.zeros((no,no))
        for i in range(no):
            for j in range(no):
                if i!=j:
                    I[i,j] = s2[i,j] - s1[i]- s1[j]
        return I