from pyscf.ita.utils.log import Log, TimerGroup, head_banner, foot_banner
from pyscf.ita.utils.constants import ITA_DICT, KE_DICT
from pyscf.ita.kernel import ITA, KineticEnergy

__all__ = ["batch_compute","section_compute"]

def batch_compute(
    ev, 
    ev_code=[],    
    filename = 'ita.log',
):
    r"""ITA batch calcuation.

    Parameters
    ----------
    ev : ExpectedValue
        Instance of ExpectedValue class.
    ev_code : List[int]
        List of expected value code to calculate.       
    filename : str, optional
        File path and name of output, by default 'ita.log'
    """
    timer = TimerGroup()
    log = Log('PYSCF-ITA', head_banner, foot_banner, timer)
    log.target = open(filename,'w')

    # ITA setting information section
    log.hline(char='=')
    log.blank()
    log('SETTING INFORMATION'.format())   
    log.hline()   
    log('{0:<40s}{1:<20s}'.format('Radial Grid', ev.grids.radi_method.__name__,))
    log('{0:<40s}{1:<20s}'.format('Angular Grid', 'Lebedev'))
    log('{0:<40s}{1:<20s}'.format('Atomic Grid Fineness', str(ev.grids.atom_grid)))
    log('{0:<40s}{1:<20s}'.format('Atom to Molecule Reweight', ev.grids.becke_scheme.__name__))
    log('{0:<40s}{1:<20s}'.format('Prune', str(ev.grids.prune)))
    log('{0:<40s}{1:<20d}'.format('Number of Grids Points', ev.grids.size))
    log('{0:<40s}{1:<20s}'.format('Category', ev.category))
    log('{0:<40s}{1:<20s}'.format('Representation', ev.representation))
    log('{0:<40s}{1:<20s}'.format('Partition', str(ev.partition)))
    log.blank()  
    log.hline(char='=')
    log.blank()  

    if isinstance(ev, ITA):
        for code in ev_code:
            if code in ITA_DICT.keys():
                if code in [4,5,6]:
                    section_compute(ev, code, log, n=2)
                    section_compute(ev, code, log, n=3)
                else:
                    section_compute(ev, code, log)
    log.print_footer()

def section_compute(
    ev, 
    ev_code, 
    log, 
    **kwargs
):
    """Function to calculation a single ITA section.

    Parameters
    ----------
    ev : ExpectedValue
        Instance of ExpectedValue class.
    code_code : int
        Expected value code to calculate.       
    log : Log
        Instance of Log class.       
    """
    if isinstance(ev, ITA):
        func_name = ITA_DICT[ev_code]  
    if isinstance(ev, KineticEnergy):
        func_name = KE_DICT[ev_code]  

    section_name = func_name.replace("_", " ").upper() 
    if 'n' in kwargs.keys():
        section_name = section_name + ' ' + str(kwargs['n'])
    log.hline(char='=')
    log.blank()
    log('{}'.format(section_name))      

    if ev.partition is not None:
        atomic_result = ev.compute(func_name, partition=True, **kwargs)
        log.hline()
        log('{0:<16s}{1:<16s}{2:>16s}'.format('Atom id', 'Atom Label', 'Atomic'))
        log.hline()  
        for atom_id, atom_label in enumerate(ev.method.mol.elements):
            log('{0:<16d}{1:<16s}{2:>16.8E}'.format(atom_id+1, atom_label, atomic_result[atom_id]))
        log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Sum:', '', sum(atomic_result)))

    if ev.representation=='atoms in molecules':
        molecular_result = sum(atomic_result)
    else:
        molecular_result = ev.compute(func_name, partition=False, **kwargs)

    log.hline()    
    log('{0:<32s}{1:>16.8E}'.format('Molecular Result:', molecular_result))
    log.blank()  
    log.hline(char='=')
    log.blank() 