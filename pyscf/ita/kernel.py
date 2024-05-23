import numpy as np
from functools import partial

from pyscf.ita.dens import OneElectronDensity, TwoElectronDensity
from pyscf.ita.ked import KineticEnergyDensity, JointKineticEnergyDensity, ConditionalKineticEnergyDensity
from pyscf.ita.aim import Hirshfeld, Becke
from pyscf.ita.eval_odm import eval_odm1, eval_odm2
from pyscf.ita.utils.constants import ITA_DICT, KE_DICT

__all__ = ["ITA", "KineticEnergy", "EnergyDecompositionAnalysis", "OrbitalEntanglement"]

class ExpectedValue:
    def __init__(self):
        NotImplemented
    def build(self):
        NotImplemented
    def compute(self):
        NotImplemented

class ITA(ExpectedValue):
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
        category='individual',
        representation='electron density',
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
        category : ('individual' | 'joint' | 'conditional' | 'relative' | 'mutual'), optional
            Type of ITA, by default 'individual'.   
        representation : ('electron density' | 'shape function' | 'atoms in molecules')
            Type of representation, by default 'electron density'.            
        partition : (None | 'hirshfeld' | 'bader' | 'becke'), optional
            Atoms in molecule partition method, by default None.                      
        promolecule : ProMolecule, optional
            ProMolecule instance, by default None.
        """
        self.method = method
        self.grids = grids  
        self.rung = rung
        self.categrory = category
        self.representation = representation
        self.partition = partition
        self.promolecule = promolecule
        
    def build(
        self, 
        method=None, 
        grids=None,
        rung=None, 
        category=None,
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
        grids : Grids, optional
            Pyscf Grids instance, by deafult None.
        rung : int
            Density derivate+1 level, by default 3,.           
        category : ('individual' | 'joint' | 'conditional' | 'relative' | 'mutual'), optional
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
        if category is None: category = self.category
        else: self.category = category
        if representation is None: representation = self.representation
        else: self.representation = representation
        if partition is None: partition = self.partition
        else: self.partition = partition
        if promolecule is None: promolecule = self.promolecule
        else: self.promolecule = promolecule

        if (partition=='hirshfeld' or category=='relative') and promolecule is None:
            from pyscf.ita.promolecule import ProMolecule
            promolecule = ProMolecule(method).build()
            self.promolecule = promolecule

        # Build molecule electron density.
        if category=='individual':
            from pyscf.ita.itad import ItaDensity
            nelec = self.method.mol.nelectron
            dens = OneElectronDensity.build(method, grids, deriv=self.rung-1)
            if rung==3:
                if representation=='shape function': 
                    dens = dens/nelec
                    orbdens = orbdens/nelec
                orbdens = OneElectronDensity.orbital_partition(method, grids, deriv=1)
                keds = KineticEnergyDensity(dens,orbdens)
                itad = ItaDensity(dens, keds)
            else:
                if representation=='shape function': 
                    dens = dens/nelec
                itad = ItaDensity(dens)
        elif category=='joint':
            from pyscf.ita.itad import JointItaDensity
            nelec = self.method.mol.nelectron
            dens = TwoElectronDensity.build(method, grids, deriv=rung-1)
            if rung==3:
                orbdens = TwoElectronDensity.orbital_partition(method, grids, deriv=1)
                if representation=='shape function': 
                    dens = dens/(nelec*(nelec-1))
                    orbdens = orbdens/(nelec*(nelec-1))
                keds = KineticEnergyDensity(dens,orbdens)
                itad = JointItaDensity(dens, keds)
            else:
                if representation=='shape function': 
                    dens = dens/(nelec*(nelec-1))
                itad = JointItaDensity(dens)
        elif category=='conditional':
            from pyscf.ita.itad import ConditionalItaDensity
            nelec = self.method.mol.nelectron
            dens = TwoElectronDensity.build(method, grids, deriv=rung-1)
            if rung==3:
                orbdens = TwoElectronDensity.orbital_partition(method, grids, deriv=1)
                if representation=='shape function': 
                    dens = dens/(nelec*(nelec-1))
                    orbdens = orbdens/(nelec*(nelec-1))
                keds = KineticEnergyDensity(dens,orbdens)
                itad = ConditionalItaDensity(dens, keds)
            else:
                if representation=='shape function': 
                    dens = dens/(nelec*(nelec-1))
                itad = ConditionalItaDensity(dens)
        elif category=='relative':
            from pyscf.ita.itad import RelativeItaDensity
            nelec = self.method.mol.nelectron
            dens = OneElectronDensity.build(method, grids, deriv=rung-1)  
            prodens = promolecule.one_electron_density(self.grids, deriv=rung-1)
            if representation=='shape function': 
                dens = dens/nelec
                prodens = prodens/nelec
            self.dens = dens
            self.prodens = prodens
            itad = RelativeItaDensity(dens,prodens)
        elif category=='mutual':
            from pyscf.ita.itad import MutualItaDensity
            nelec = self.method.mol.nelectron
            onedens = OneElectronDensity.build(method, grids, deriv=rung-1)
            twodens = TwoElectronDensity.build(method, grids, deriv=rung-1)
            if representation=='shape function': 
                onedens = onedens/nelec
                twodens = twodens/(nelec*(nelec-1))
            itad = MutualItaDensity(onedens,twodens)
        else:
            raise ValueError("Not a valid category.")
        self.itad = itad

        # Build atoms-in-molecules
        if partition is not None:
            if partition is not None:
                if partition.lower()=='hirshfeld':                
                    prodens = promolecule.one_electron_density(grids, deriv=0)            
                    aim = Hirshfeld(prodens)
                elif partition.lower()=='becke':
                    aim = Becke(method.mol, grids)
                elif partition.lower()=='bader':
                    raise NotImplemented
                else:
                    raise ValueError("Not a valid partition.")                
            self.aim = aim

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

        if self.representation == 'electron density':
            if partition:
                omega = self.aim.sharing_function()
                result = []
                for omega_i in omega:            
                    itad_i = itad_func()*omega_i
                    atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                    result.append(atomic_partition)
            else:
                result = ita_func()
        if self.representation == 'shape function':
            if partition:
                omega = self.aim.sharing_function()
                result = []
                for omega_i in omega:            
                    itad_i = itad_func(omega=omega_i)
                    atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                    result.append(atomic_partition)
            else:
                result = ita_func()                
        if self.representation=='atoms in molecules':
            omega = self.aim.sharing_function()
            result = []
            for atom_id, omega_i in enumerate(omega):            
                if ita_name in ['fisher_information','G3']:
                    prorho_i = self.prodens[atom_id].density(mask=True)
                    prorho_grad_i = self.prodens[atom_id].gradient()
                    itad_i = itad_func(omega=omega_i, prorho=prorho_i, prorho_grad=prorho_grad_i)                          
                else:
                    itad_i = itad_func(omega=omega_i)
                atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                result.append(atomic_partition)
            if partition is None:
                result = sum(result) 
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
        from pyscf.ita.script import batch_compute
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
            result = np.einsum("g, g -> ", grids_weights, ita_density)
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
            result = np.einsum("g, g -> ", grids_weights, ita_density)
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ita_density : np.ndarray((N,), dtype=float), optional
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
    
class KineticEnergy(ExpectedValue):
    r"""Kinetic energy (KE) class.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> mf = scf.HF(mol)
    >>> mf.kernel()
    >>> grids = dft.Grids(mol)
    >>> grids.build()    
    >>> ke = KineticEnergy(mf,grids)
    >>> ke.category = 'individual'    
    >>> ke.build()        
    """
    def __init__(
        self, 
        method=None, 
        grids=None,
        category='individual',
        representation='electron density',
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
        category : ('individual' | 'joint' | 'conditional' | 'relative' | 'mutual'), optional
            Type of ITA, by default 'individual'.   
        representation : ('electron density' | 'shape function' | 'atoms in molecules')
            Type of representation, by default 'electron density'.            
        partition : (None | 'hirshfeld' | 'bader' | 'becke'), optional
            Atoms in molecule partition method, by default None.                      
        promolecule : ProMolecule, optional
            ProMolecule instance, by default None.
        """
        self.method = method
        self.grids = grids  
        self.category = category
        self.representation = representation
        self.partition = partition
        self.promolecule = promolecule
        
    def build(
        self, 
        method=None, 
        grids=None,
        category=None,
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
        grids : Grids, optional
            Pyscf Grids instance, by deafult None.
        category : ('individual' | 'joint' | 'conditional' | 'relative' | 'mutual'), optional
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
        if category is None: category = self.category
        else: self.category = category
        if representation is None: representation = self.representation
        else: self.representation = representation
        if partition is None: partition = self.partition
        else: self.partition = partition
        if promolecule is None: promolecule = self.promolecule
        else: self.promolecule = promolecule

        if (partition=='hirshfeld') and promolecule is None:
            from pyscf.ita.promolecule import ProMolecule
            promolecule = ProMolecule(method).build()
            self.promolecule = promolecule

        # Build molecule electron density.
        if category=='individual':
            dens = OneElectronDensity.build(method, grids, deriv=2)
            orbdens = OneElectronDensity.orbital_partition(method, grids, deriv=1)
            keds = KineticEnergyDensity(dens,orbdens)
        elif category=='joint':
            dens = TwoElectronDensity.build(method, grids, deriv=2)
            orbdens = TwoElectronDensity.orbital_partition(method, grids, deriv=1)
            keds = JointKineticEnergyDensity(dens,orbdens)
        elif category=='conditional':
            dens = TwoElectronDensity.build(method, grids, deriv=2)
            orbdens = TwoElectronDensity.orbital_partition(method, grids, deriv=1)
            keds = ConditionalKineticEnergyDensity(dens,orbdens)
        else:
            raise ValueError("Not a valid category.")
        
        # Build atoms-in-molecules
        if partition is not None:
            if partition is not None:
                if partition.lower()=='hirshfeld':                
                    prodens = promolecule.one_electron_density(grids, deriv=0)            
                    aim = Hirshfeld(prodens)
                elif partition.lower()=='becke':
                    aim = Becke(method.mol, grids)
                elif partition.lower()=='bader':
                    raise NotImplemented
                else:
                    raise ValueError("Not a valid partition.")                
            self.aim = aim

        # Build ITA density
        self.keds = keds
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
        if code_name in KE_DICT.keys():         
            ita_name = KE_DICT[code_name]  
        elif code_name in KE_DICT.values():
            ita_name = code_name
        else:
            raise ValueError("Not a valid ita code or name.")

        omega = self.aim.sharing_function()
        ita_func = getattr(self, ita_name)
        ita_func = partial(ita_func, **kwargs)
        if ita_name in ['renyi_entropy','tsallis_entropy','onicescu_information']:
            itad_func = getattr(self.itad, "rho_power")
        else:
            itad_func = getattr(self.itad, ita_name)
            itad_func = partial(itad_func, **kwargs) 

        if self.representation == 'electron density':
            if partition:
                result = []
                for omega_i in omega:            
                    itad_i = itad_func()*omega_i
                    atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                    result.append(atomic_partition)
            else:
                result = ita_func()
        if self.representation == 'shape function':
            if partition:
                result = []
                for omega_i in omega:            
                    nelec_i = (self.grids.weights * self.moldens.density() * omega_i).sum()
                    itad_i = itad_func(omega=1./nelec_i)
                    atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                    result.append(atomic_partition)
            else:
                nelec = (self.grids.weights * self.moldens.density()).sum()
                itad = itad_func(omega=1./nelec)
                result = ita_func(ita_density=itad, **kwargs)                
        if self.representation=='atoms in molecules':
            result = []
            for atom_id, omega_i in enumerate(omega):            
                if ita_name in ['fisher_information','G3']:
                    prorho_i = self.prodens[atom_id].density(mask=True)
                    prorho_grad_i = self.prodens[atom_id].gradient()
                    itad_i = itad_func(omega=omega_i, prorho=prorho_i, prorho_grad=prorho_grad_i)                          
                else:
                    itad_i = itad_func(omega=omega_i)
                atomic_partition = ita_func(ita_density=itad_i, **kwargs)
                result.append(atomic_partition)
            if partition is None:
                result = sum(result) 
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
        from pyscf.ita.script import batch_compute
        return batch_compute(self, ita_code, filename)
        
    def general(
        self, 
        a = 0.0,
        grids_weights=None, 
        ked=None
    ):
        r"""Compute general(ish) kinetic energy defined as:

        .. math::
            T_\text{G} (\alpha) = \int
               \tau_\text{PD} (\mathbf{r}) +
               \frac{1}{4} (a - 1) \nabla^2 \rho (\mathbf{r})
                d\mathbf{r} 

        Parameters
        ----------
        a : float
            Value of parameter :math:`a`.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.general(a=a)

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result
    
    def thomas_fermi(
        self, 
        grids_weights=None, 
        ked=None
    ):
        r"""Thomas-Fermi kinetic energy defined as:

        .. math::
            T_\text{TF} \left(\mathbf{r}\right) = \int
                \tfrac{3}{10} \left(6 \pi^2 \right)^{2/3}
                \left(\frac{\rho\left(\mathbf{r}\right)}{2}\right)^{5/3}
                d\mathbf{r} 

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """    
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.thomas_fermi()

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result          

    def dirac(
        self, 
        grids_weights=None, 
        ked=None
    ):
        r"""Dirac kinetic energy defined as:

        .. math::
            T_\text{D} = \int
                \tfrac{3}{10} \left(6 \pi^2 \right)^{2/3}
                \left(\frac{\rho\left(\mathbf{r}\right)}{2}\right)^{5/3}
                d\mathbf{r} 

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """   
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.dirac()

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result              
    
    def weizsacker(
        self, 
        grids_weights=None, 
        ked=None
    ):    
        r"""Weizsacker kinetic energy defined as:

        .. math::
            T_\text{W} (\mathbf{r}) = \int \frac{1}{8}
                \frac{\lvert \nabla\rho (\mathbf{r}) \rvert^2}{\rho (\mathbf{r})}
                d\mathbf{r} 

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """   
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.weizsacker()

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result  
    
    def gradient_expansion(
        self,
        a = None,
        b = None, 
        grids_weights=None, 
        ked=None
    ):
        r"""Gradient expansion approximation of kinetic energy defined as:

        .. math::
            T_\text{GEA}(a,b) = \int
                \tau_\text{TF} \left(\mathbf{r}\right) +
                a* \tau_\text{W} \left(\mathbf{r}\right) +
                b* \nabla^2 \rho\left(\mathbf{r}\right)
                d\mathbf{r}

        There is a special case of :func:`gradient_expansion` with
        :math:`a=\tfrac{1}{9}` and :math:`b=\tfrac{1}{6}`.

        Parameters
        ----------
        a : float, optional
            Value of parameter :math:`a`, by default None.
        b : float, optional
            Value of parameter :math:`b`, by default None.
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """   
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.gradient_expansion(a=a,b=b)

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result          
    
    def single_particle(
        self, 
        grids_weights=None, 
        ked=None
    ):
        r"""Single-particle kinetic energy defined as:

        .. math::
            T_\text{s} = \int \sum_i \frac{1}{8} 
                \frac{\nabla \rho_i \nabla \rho_i}{\rho_i}
                -\frac{1}{8} \nabla^2 \rho    
                d\mathbf{r} 

        where :math:`\rho_i` are the Kohn-Sham orbital densities.

        Parameters
        ----------
        grids_weights : np.ndarray((N,), dtype=float), optional
            Grids weights on N points, by default None.
        ked : np.ndarray((N,), dtype=float), optional
            Kenitic energy density, by default None.        

        Returns
        -------
        result : float
            Scalar kinetic energy result.
        """   
        if grids_weights is None:
            grids_weights = self.grids.weights 
        if ked is None:
            ked = self.keds.single_particle()

        if len(ked.shape)==1:
            result = np.einsum("g, g -> ", grids_weights, ked)
        if len(ked.shape)==2:
            result = np.einsum("g, h, gh -> ", grids_weights, grids_weights, ked)
        return result          
    
class EnergyDecompositionAnalysis(ExpectedValue):
    r"""Kinetic energy (KE) class.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> mf = scf.HF(mol)
    >>> mf.kernel()
    >>> grids = dft.Grids(mol)
    >>> grids.build()    
    >>> ke = KineticEnergy(mf,grids)
    >>> ke.category = 'individual'    
    >>> ke.build()        
    """
    def __init__(
        self, 
        method=None, 
        grids=None,
    ):
        r"""Initialize a instance.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
        grids : Grids
            Pyscf Grids instance.
        """
        self.method = method
        if grids is None:
            self.grids = method.grids  

    @property
    def kinetic(self):
        r"""The noninteracting kinetic energy.
        """
        T = self.method.mol.intor('int1e_kin')
        rdm1 = self.method.make_rdm1()
        result = np.einsum('uv,uv->',T,rdm1)
        return result  
       
    @property   
    def weizsacker(self):
        r""" Weizsacker kinetic energy.
        """
        result = self.kes.weizsacker()
        return result

    @property
    def coulomb(self):
        r""" interelectronic coulomb repulsion energy.
        """
        rdm1 = self.method.make_rdm1()
        I = self.method.mol.intor('int2e')
        J = np.einsum('pqrs,rs->pq', I, rdm1)
        result = np.einsum('uv,uv->',J,rdm1)
        return result
        
    @property
    def pauli(self):
        r""" Pauli kinetic energy.
        """
        ET = self.kinetic
        EW = self.weizsacker
        result = ET-EW
        return result

    @property
    def xc(self):
        r"""Exchange correlation energy.
        """
        result = self.method.scf_summary['exc']        
        return result

    @property
    def nuclear(self):
        r"""Internuclear repulsion energy.
        """
        result = self.method.mol.energy_nuc()
        return result
    
    @property
    def nuclear_electronic(self):
        r"""Nuclear electronic attraction energy.
        """
        V = self.method.mol.intor('int1e_nuc')
        rdm1 = self.method.make_rdm1()
        result = np.einsum('uv,uv->',V,rdm1)
        return result

    @property
    def steric(self):
        """_summary_
        """        
        result = self.method.e_tot - self.electrostatic - self.quantum
        return result

    @property
    def electrostatic(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """        
        result = self.coulomb + self.nuclear + self.nuclear_electronic
        return result

    @property
    def quantum(self):
        """_summary_

        .. math::

            E_{quantum} = E_{steric} + E_{electrostatic} 
        """       
        result = self.xc + self.kinetic -self.weizsacker
        return result

    def build(
        self, 
        method=None, 
        grids=None,
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
        """
        if method is None: method = self.method 
        else: self.method = method
        if grids is None: grids = self.grids
        else: self.grids = grids

        kes = KineticEnergy(method, grids)
        kes.build()
        self.kes = kes
        return self    

    def Liu(self):
        """The energy decomposition analysis method proposed by Shubin Liu decomposes 
        total molecular energy as follow:

        .. math::

            E_{tot} = E_{steric} + E_{electrostatic}+ E_{quantum} 
        """        
        result = {}
        result['E_steric'] = self.steric
        result['E_electrostatic'] = self.electrostatic
        result['E_quantum'] = self.quantum
        return result

class OrbitalEntanglement(ExpectedValue):
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