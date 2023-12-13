import numpy as np
from pyscf import dft

from pyscf.ita.itd import InformationTheoryDensity
from pyscf.ita.dens import Density
from pyscf.ita.ked import KineticEnergyDensity

__all__ = ["ITA"]


class ITA:
    def __init__(self, method=None, grids=None, rung=[3,3], 
                 promol_charges={}, promol_mults={}, spin='ab',
                 moldens=None, promoldens=None,
                 molkeds=None, promolkeds=None):
        """_summary_

        Parameters
        ----------
        grids_weights : _type_
            _description_
        """
        self.method = method
        self.grids = grids  
        self.rung = rung
        self.promol_charges = promol_charges
        self.promol_mults = promol_mults
        self.spin = spin
        self.moldens = moldens
        self.promoldens = promoldens
        self.molkeds = molkeds
        self.promolkeds = promolkeds

    def build(self):
        method = self.method
        spin = self.spin

        if self.grids is not None:
            grids_coords, grids_weights = self.grids.get_partition(method.mol) 
            self.grids_coords = grids_coords   
            self.grids_weights = grids_weights  
       
        moldens = Density.molecule(method, grids_coords, spin=spin, deriv=self.rung[0]-1)
        self.moldens = moldens
        if self.rung[0]==3:
            molkeds = KineticEnergyDensity(moldens)
            self.molkeds = molkeds

        charge = self.promol_charges
        multiplicity = self.promol_mults
        promoldens = Density.promolecule(method, grids_coords, charge=charge, multiplicity=multiplicity , spin=spin, deriv=self.rung[1]-1)
        self.promoldens = promoldens

        descriptor = InformationTheoryDensity(self.moldens, self.promoldens, self.molkeds, self.promolkeds)
        self.descriptor = descriptor

    @classmethod
    def from_fchk(cls, filename, ita_code=[], setting={'atom_grid': (75,302), 'prune':True}):
        """_summary_

        Parameters
        ----------
        filename : _type_
            _description_
        ita_code : list, optional
            _description_, by default []
        setting : dict, optional
            _description_, by default {'atom_grid': (75,302), 'prune':True}

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """        
        from pyita.io.fchk import FCHK
        
        # Build fchk
        fchk = FCHK.from_file(filename)

        # Build grids
        mol = fchk.dump_mol
        grids = dft.Grids(mol)
        grids.prune = setting['prune']
        grids.atom_grid = setting['atom_grid']
        grids.build()

        # Build density matrix
        nmo = fchk.dataset['Number of basis functions']   
        rdm1 = np.zeros((nmo, nmo))
        rdm1[np.triu_indices(nmo, k=0)] = fchk.dataset['Total SCF Density']
        grids_coords = grids.get_partition(mol)[0]

        # Get the max rho deriv 
        level_list = []
        for c in ita_code:
            level_list.append(int(str(c)[0]))
        level = max(level_list)
        
        # Build instance of ITA class
        if level==1:
            aos = dft.numint.eval_ao(mol, grids_coords, deriv=0)
            rho = dft.numint.eval_rho(mol, aos, rdm1, xctype='LDA')
            obj = cls(grids, rho)
        elif level==2:
            aos = dft.numint.eval_ao(mol, grids_coords, deriv=1)
            rhos = dft.numint.eval_rho(mol, aos, rdm1, xctype='GGA')
            rho  = rhos[0]
            rho_grad = rhos[1:4]
            obj = cls(grids, rho, rho_grad)
        elif level==3:
            aos = dft.numint.eval_ao(mol, grids_coords, deriv=2)
            rhos = dft.numint.eval_rho(mol, aos, rdm1, xctype='mGGA')
            rho  = rhos[0]
            rho_grad = rhos[1:4]
            rho_lap = rho[4:6]
            obj = cls(grids, rho, rho_grad, rho_lap)
        elif level==4:
            aos = dft.numint.eval_ao(mol, grids_coords, deriv=2)
            rhos = dft.numint.eval_rho(mol, aos, rdm1, xctype='mGGA')
            rho  = rhos[0]
            rho_grad = rhos[1:4]
            rho_lap = rho[4:6]
            obj = cls(grids, rho, rho_grad, rho_lap)
        else:
            raise ValueError('Level value not valid.')

        obj.ita_code = ita_code
        return obj

    @property
    def shannon_entropy(self):
        elements = self.method.mol.elements
        grids_weights = self.grids_weights
        kernel = self.descriptor.shannon_entropy

        partition_result = []

        total_result = (-grids_weights * kernel).sum()

        return partition_result, total_result

    @property
    def fisher_information(self):
        elements = self.method.mol.elements
        partition_grids_weights = np.split(self.grids_weights, len(elements))

        kernel = self.descriptor.fisher_information
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((grids_weights * kernel).sum()) 

        total_result = sum(partition_result)

        return partition_result, total_result        

    def onicescu_information(self, n):
        elements = self.method.mol.elements
        partition_grids_weights = np.split(self.grids_weights, len(elements))

        kernel = self.descriptor.rho_power(n)
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((1/(n-1))*(grids_weights * kernel).sum()) 

        total_result = sum(partition_result)

        return partition_result, total_result         
    
    def renyi_entropy(self, n):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """ 
        elements = self.method.mol.elements
        partition_grids_weights = np.split(self.grids_weights, len(elements))

        kernel = self.descriptor.rho_power(n)
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((grids_weights * kernel).sum()) 

        total_result = (1/(1-n))*np.log10(sum(partition_result))
        return partition_result, total_result
    
    def Tsallis_entropy(self, n):
        elements = self.method.mol.elements
        partition_grids_weights = np.split(self.grids_weights, len(elements))

        kernel = self.descriptor.rho_power(n)
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((1/(n-1))*(grids_weights * kernel).sum()) 

        total_result = sum(partition_result)

        return partition_result, total_result 

    @property
    def relative_shannon_entropy(self):
        elements = self.method.mol.elements
        partition_grids_weights = np.split(self.grids_weights, len(elements))

        kernel = self.descriptor.relative_shannon_entropy
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((grids_weights * kernel).sum()) 

        total_result = sum(partition_result)

        return partition_result, total_result    

    def relative_renyi_entropy(self, n):
        r"""Shannon information defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """ 
        elements = self.method.mol.elements
        partition_grids_weights = np.split(self.grids_weights, len(elements))

        kernel = self.descriptor.relative_renyi_entropy(n)
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((grids_weights * kernel).sum()) 

        total_result = (1/(1-n))*np.log10(sum(partition_result))
        return partition_result, total_result 

    def GBP_entropy(self):
        r"""Ghosh-BerkowitzParr (GBP) entropy defined as :math:`\rho(r) \ln \rho(r)`.

        Returns
        -------
        _type_
            _description_
        """
        aos = dft.numint.eval_ao(self.method.mol, self.grids_coords, deriv=1)
        mo_coeff = self.method.mo_coeff
        mo_occ = self.method.mo_occ

        elements = self.method.mol.elements
        partition_grids_weights = np.split(self.grids_weights, len(elements))

        kernel = self.descriptor.GBP_entropy(aos, mo_coeff, mo_occ)                    
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((grids_weights * kernel).sum()) 

        total_result = sum(partition_result)
        return partition_result, total_result 
