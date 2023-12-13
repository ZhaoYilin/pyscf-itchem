import numpy as np
from pyscf import dft

from pyita.io.log import log, timer
from pyita.itd import ItaDescriptor

__all__ = ["ITA"]


class ITA:
    def __init__(self, grids=None, rho=None, rho_grad=None, rho_lap=None):
        """_summary_

        Parameters
        ----------
        grids_weights : _type_
            _description_
        """
        self.grids = grids
        self.partition_grids = grids.get_partition(grids.mol)
        self.rho = rho
        self.rho_grad = rho_grad
        self.rho_lap = rho_lap
        self.descriptor = ItaDescriptor(rho, rho_grad, rho_lap)


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
    

    @classmethod
    def from_pyscf(cls, method=None, grids=None, ita_code=[]):
        """_summary_

        Parameters
        ----------
        method : _type_, optional
            _description_, by default None
        grids : _type_, optional
            _description_, by default None
        ita_code : list, optional
            _description_, by default []

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """        

        mol = method.mol
        rdm1 = method.make_rdm1()
        grids_coords = grids.get_partition(mol)[0]


        level_list = []
        for c in ita_code:
            level_list.append(int(str(c)[0]))
        level = max(level_list)
        
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


    def compute(self, filename = 'pyita.log'):
        """_summary_

        Parameters
        ----------

        filename : str, optional
            _description_, by default 'pyita.log'

        Raises
        ------
        ValueError
            _description_
        """
        log.target = open(filename,'w')
        ita_code = self.ita_code
        elements = self.grids.mol.elements

        # Grid information section
        grids = self.grids
        log.hline(char='=')
        log.blank()
        log('Grid Information Section'.format())   
        log.hline()   
        log('{0:<16s}{1:<16s}{2:<16s}{3:<16s}'.format('Scheme', 'Partition', 'Fineness', 'Prune'))
        log.hline()
        log('{0:<16s}{1:<16s}{2:<16s}{3:<16s}'.format('Lebedev', 'Hirshfeld', str(grids.atom_grid), str(grids.prune)))
        log.blank()  
        log.hline(char='=')
        log.blank()  

        # ITA Section
        for c in ita_code:
            if c==11:
                log.hline(char='=')
                log.blank()
                log('Shannon entropy Section'.format())      
                partition_result, total_result = self.shannon_entropy
                log.hline()
                log('{0:<16s}{1:<16s}{2:>16s}'.format('Atom id', 'Atom Label', 'Partition Kernel'))
                log.hline()
                for atom_id, (atom_label, result) in enumerate(zip(elements, partition_result)):
                    log('{0:<16d}{1:<16s}{2:>16.8E}'.format(atom_id, atom_label, result))
                log.hline()    
                log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Total:', '', total_result))
                log.blank()  
                log.hline(char='=')
                log.blank()  
            elif c==12:
                log.hline(char='=')
                log.blank()
                log('Onicescu Information Section'.format())      
                partition_result_2, total_result_2 = self.onicescu_information(2)
                partition_result_3, total_result_3 = self.onicescu_information(3)
                log.hline()
                log('{0:<16s}{1:<16s}{2:>16s}{3:>16s}'.format('Atom id', 'Atom Label', 'Quadratic Result', 'Cubic Result'))
                log.hline()
                for atom_id, (atom_label, result2, result3) in enumerate(zip(elements, partition_result_2, partition_result_3)):
                    log('{0:<16d}{1:<16s}{2:>16.8E}{3:>16.8E}'.format(atom_id, atom_label, result2, result3))
                log.hline()        
                log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Quadratic Total:', '', total_result_2))
                log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Cubic Total:', '', total_result_3))
                log.blank()  
                log.hline(char='=')
                log.blank()                 
            elif c==21:
                log.hline(char='=')
                log.blank()
                log('Fisher Information Section'.format())      
                partition_result, total_result = self.fisher_information
                log.hline()
                log('{0:<16s}{1:<16s}{2:>16s}'.format('Atom id', 'Atom Label', 'Result'))
                log.hline()
                for atom_id, (atom_label, result) in enumerate(zip(elements, partition_result)):
                    log('{0:<16d}{1:<16s}{2:>16.8E}'.format(atom_id, atom_label, result))
                log.hline()     
                log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Total', '', total_result))
                log.blank()  
                log.hline(char='=')
                log.blank()  
            else:
                raise ValueError('The function code are not avaible.')

    @property
    @timer.with_section('Shannon')
    def shannon_entropy(self):
        elements = self.grids.mol.elements
        partition_grids_weights = np.split(self.partition_grids[1], len(elements))
 
        kernel = self.descriptor.shannon_entropy
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((-grids_weights * kernel).sum()) 

        total_result = sum(partition_result)

        return partition_result, total_result

    @property
    @timer.with_section('Fisher')
    def fisher_information(self):
        elements = self.grids.mol.elements
        partition_grids_weights = np.split(self.partition_grids[1], len(elements))

        kernel = self.descriptor.fisher_information
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((grids_weights * kernel).sum()) 

        total_result = sum(partition_result)

        return partition_result, total_result        

    @timer.with_section('Onicescu')
    def onicescu_information(self, n):
        elements = self.grids.mol.elements
        partition_grids_weights = np.split(self.partition_grids[1], len(elements))

        kernel = self.descriptor.rho_power(n)
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((1/(n-1))*(grids_weights * kernel).sum()) 

        total_result = sum(partition_result)

        return partition_result, total_result         
    
    @timer.with_section('Relative Shannon')
    def relative_shannon_entropy(self):
        elements = self.grids.mol.elements
        partition_grids_weights = np.split(self.partition_grids[1], len(elements))

        kernel = self.descriptor.fisher_information
        partition_kernel = np.split(kernel,len(elements))

        partition_result = []
        for grids_weights, kernel in zip(partition_grids_weights, partition_kernel):
            partition_result.append((grids_weights * kernel).sum()) 

        total_result = sum(partition_result)

        return partition_result, total_result    