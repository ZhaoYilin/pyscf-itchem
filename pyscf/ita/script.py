from functools import partial
import numpy as np

from pyscf.ita.log import Log, TimerGroup, head_banner, foot_banner
from pyscf.ita.aim import Hirshfeld, Becke

__all__ = ["batch_compute","section_compute"]

ITA_CODE = {
    11 : 'shannon_entropy',
    12 : 'fisher_information',
    13 : 'alternative_fisher_information',
    14 : 'renyi_entropy',
    15 : 'tsallis_entropy',
    16 : 'onicescu_information',
    17 : 'GBP_entropy',

    21 : 'relative_shannon_entropy',
    22 : 'relative_fisher_information',
    23 : 'relative_alternative_fisher_information',
    24 : 'relative_renyi_entropy',
    25 : 'relative_tsallis_entropy',
    26 : 'relative_onicescu_information',

    28 : 'G1',
    29 : 'G2',
    30 : 'G3'
}


def batch_compute(
    ita, 
    ita_code=[], 
    representation='electron density',
    partition='hirshfeld',    
    filename = 'ita.log',
):
    r"""ITA batch calcuation.

    Parameters
    ----------
    ita : ITA
        Instance of ITA class.
    ita_code : List[int]
        List of ITA code to calculate.
    representation : ('electron density' | 'shape function' | 'atoms in molecules')
        Type of representation, by default 'electron density'.
    partition : ('hirshfeld' | 'bader' | 'becke'), optional
        Atoms in molecule partition method, by default 'hirshfeld'.        
    filename : str, optional
        File path and name of output, by default 'ita.log'
    """
    timer = TimerGroup()
    log = Log('PYSCF-ITA', head_banner, foot_banner, timer)
    log.target = open(filename,'w')

    # Grid information section
    grids = ita.grids
    log.hline(char='=')
    log.blank()
    log('SETTING INFORMATION'.format())   
    log.hline()   
    log('{0:<40s}{1:<20s}'.format('Radial Grid', grids.radi_method.__name__,))
    log('{0:<40s}{1:<20s}'.format('Angular Grid', 'Lebedev'))
    log('{0:<40s}{1:<20s}'.format('Atomic Grid Fineness', str(grids.atom_grid)))
    log('{0:<40s}{1:<20s}'.format('Atom to Molecule Reweight', grids.becke_scheme.__name__))
    log('{0:<40s}{1:<20s}'.format('Prune', str(grids.prune)))
    log('{0:<40s}{1:<20d}'.format('Number of Grids Points', grids.size))
    log('{0:<40s}{1:<20s}'.format('Representation', representation))
    log('{0:<40s}{1:<20s}'.format('Partition', partition))
    log.blank()  
    log.hline(char='=')
    log.blank()  

    for code in ita_code:
        if code in ITA_CODE.keys():
            if code in [14,15,16,24,25,26]:
                section_compute(ita, code, log, representation, partition, n=2)
                section_compute(ita, code, log, representation, partition, n=3)
            else:
                section_compute(ita, code, log, representation, partition)
        else:
            raise ValueError("{d} is not a valid ita code.".format(code))        
    log.print_footer()

def section_compute(
    ita, 
    code, 
    log, 
    representation='electron density',
    partition='hirshfeld',    
    **kwargs
):
    """Function to calculation a single ITA section.

    Parameters
    ----------
    ita : ITA
        Instance of ITA class.
    code : int
        Single ITA code.
    log : Log
        Instance of Log class.
    representation : ('electron density' | 'shape function' | 'atoms in molecules')
        Type of representation, by default 'electron density'.
    partition : ('hirshfeld' | 'bader' | 'becke'), optional
        Atoms in molecule partition method, by default 'hirshfeld'.        
    """
    ita_name = ITA_CODE[code]    
    ita_func = getattr(ita, ita_name)
    ita_func = partial(ita_func, **kwargs)
    if code in [14,15,16]:
        itad_func = getattr(ita.itad, "rho_power")
    elif code in [24,25,26]:
        itad_func = getattr(ita.itad, "relative_rho_power")
    elif code in [28,29,30] and representation!='atoms in molecules':
        raise ValueError("The G1, G2 and G3 calculation works only in atoms in molecules representation")
    else:
        itad_func = getattr(ita.itad, ita_name)
    itad_func = partial(itad_func, **kwargs)


    section_name = ita_name.replace("_", " ").upper() 
    if 'n' in kwargs.keys():
        section_name = section_name + ' ' + str(kwargs['n'])
    log.hline(char='=')
    log.blank()
    log('{}'.format(section_name))      
    log.hline()
    log('{0:<16s}{1:<16s}{2:>16s}'.format('Atom id', 'Atom Label', 'Atomic'))
    log.hline()  


    grids = ita.grids
    elements = ita.method.mol.elements

    # Build atoms-in-molecules
    if partition is not None:
        if partition.lower()=='hirshfeld':
            aim = Hirshfeld(ita.promoldens)
            omega = aim.sharing_function()
        elif partition.lower()=='becke':
            aim = Becke()
            omega = aim.sharing_function(ita.method.mol,grids)
        elif partition.lower()=='bader':
            raise ValueError("Not implemented yet.")
        else:
            raise ValueError("Not a valid partition.")

    # Section Header
    if representation=='electron density':
        atomic_sum = 0.
        molecular_total = ita_func()
        for atom_id, (atom_label, omega_i) in enumerate(zip(elements, omega)):
            itad_i = itad_func()*omega_i
            atomic_partition = ita_func(ita_density=itad_i, **kwargs)
            atomic_sum += atomic_partition
            log('{0:<16d}{1:<16s}{2:>16.8E}'.format(atom_id, atom_label, atomic_partition))
        log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Sum:', '', atomic_sum))
        log.hline()    
        log('{0:<32s}{1:>16.8E}'.format('Molecular ITA:', molecular_total))
        log.blank()  
        log.hline(char='=')
        log.blank()
    elif representation=='shape function':
        nelec = ita.method.mol.nelectron
        atomic_sum = 0.
        itad = itad_func(omega=1./nelec)
        molecular_total = ita_func(ita_density=itad, **kwargs)
        for atom_id, (atom_label, omega_i) in enumerate(zip(elements, omega)):
            nelec_i = (ita.grids.weights * ita.moldens.density() * omega_i).sum()
            itad_i = itad_func(omega=1./nelec_i)
            atomic_partition = ita_func(ita_density=itad_i, **kwargs)
            atomic_sum += atomic_partition
            log('{0:<16d}{1:<16s}{2:>16.8E}'.format(atom_id, atom_label, atomic_partition))
        log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Sum:', '', atomic_sum))
        log.hline()    
        log('{0:<32s}{1:>16.8E}'.format('Molecular ITA:', molecular_total))
        log.blank()  
        log.hline(char='=')
        log.blank()        
    elif representation=='atoms in molecules':
        atomic_sum = 0.        
        for atom_id, (atom_label, omega_i) in enumerate(zip(elements, omega)):
            if code in [22,30]:
                prorho_i = ita.promoldens[atom_id].density(mask=True)
                prorho_grad_i = ita.promoldens[atom_id].gradient()
                itad_i = itad_func(omega=omega_i, prorho=prorho_i, prorho_grad=prorho_grad_i)                          
            else:
                itad_i = itad_func(omega=omega_i)
            atomic_partition = ita_func(ita_density=itad_i, **kwargs)
            atomic_sum += atomic_partition
            log('{0:<16d}{1:<16s}{2:>16.8E}'.format(atom_id, atom_label, atomic_partition))
        log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Sum:', '', atomic_sum))
        log.hline()    
        log('{0:<32s}{1:>16.8E}'.format('Molecular ITA:', atomic_sum))
        log.blank()  
        log.hline(char='=')
        log.blank()     
    else:
        raise ValueError("Not valid representation.")    