from pyscf.itchem.utils.log import Log, TimerGroup, head_banner, foot_banner
from pyscf.itchem.utils.constants import ITA_DICT
from pyscf.itchem.kernel import ITA

__all__ = ["batch_compute","section_compute"]

def batch_compute(
    globle, 
    globle_code=[],    
    filename = 'ita.log',
):
    r"""ITA batch calcuation.

    Parameters
    ----------
    globle : Globle
        Instance of Globle class.
    globle_code : List[int]
        List of globle value code to calculate.       
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
    log('{0:<40s}{1:<20s}'.format('Radial Grid', globle.grids.radi_method.__name__,))
    log('{0:<40s}{1:<20s}'.format('Angular Grid', 'Lebedev'))
    log('{0:<40s}{1:<20s}'.format('Atomic Grid Fineness', str(globle.grids.atom_grid)))
    log('{0:<40s}{1:<20s}'.format('Atom to Molecule Reweight', globle.grids.becke_scheme.__name__))
    log('{0:<40s}{1:<20s}'.format('Prune', str(globle.grids.prune)))
    log('{0:<40s}{1:<20d}'.format('Number of Grids Points', globle.grids.size))
    log('{0:<40s}{1:<20s}'.format('Category', globle.category))
    log('{0:<40s}{1:<20s}'.format('Normalized', str(globle.normalize)))
    log('{0:<40s}{1:<20s}'.format('Partition', str(globle.partition)))
    log.blank()  
    log.hline(char='=')
    log.blank()  

    if isinstance(globle, ITA):
        for code in globle_code:
            if code in ITA_DICT.keys():
                if code in [4,5,6]:
                    section_compute(globle, code, log, n=2)
                    section_compute(globle, code, log, n=3)
                else:
                    section_compute(globle, code, log)
    log.print_footer()

def section_compute(
    globle, 
    globle_code, 
    log, 
    **kwargs
):
    """Function to calculation a single ITA section.

    Parameters
    ----------
    globle : Globle
        Instance of Globle class.
    globle_code : int
        Globle value code to calculate.       
    log : Log
        Instance of Log class.       
    """
    if isinstance(globle, ITA):
        func_name = ITA_DICT[globle_code]  

    section_name = func_name.replace("_", " ").upper() 
    if 'n' in kwargs.keys():
        section_name = section_name + ' ' + str(kwargs['n'])
    log.hline(char='=')
    log.blank()
    log('{}'.format(section_name))      

    if globle.partition is not None:
        ita_func = getattr(globle, func_name)
        ita_func = globle(ita_func, **kwargs)
        atomic_result = ita_func()
        #atomic_result = globle.compute(func_name, partition=True, **kwargs)
        log.hline()
        log('{0:<16s}{1:<16s}{2:>16s}'.format('Atom id', 'Atom Label', 'Atomic'))
        log.hline()  
        for atom_id, atom_label in enumerate(globle.method.mol.elements):
            log('{0:<16d}{1:<16s}{2:>16.8E}'.format(atom_id+1, atom_label, atomic_result[atom_id]))
        log('{0:<16s}{1:<16s}{2:>16.8E}'.format('Sum:', '', sum(atomic_result)))

    if globle.aim:
        molecular_result = sum(atomic_result)
    else:
        molecular_result = globle.compute(func_name, partition=False, **kwargs)

    log.hline()    
    log('{0:<32s}{1:>16.8E}'.format('Molecular Result:', molecular_result))
    log.blank()  
    log.hline(char='=')
    log.blank() 