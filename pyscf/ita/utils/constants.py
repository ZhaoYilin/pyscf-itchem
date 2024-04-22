"""Physical constants in atomic units.
"""
boltzmann = 3.1668154051341965e-06
avogadro = 6.0221415e23
planck = 6.2831853071795864769
lightspeed = 137.035999206

"""Dictionary of the multiplicities for each isoelectronic series (up to 100 electrons).
"""
MULTIPLICITIES = {
    "H": 2, "He":1, 
    "Li":2, "Be":1, "B": 2, "C": 3, "N": 4, "O": 3, "F": 2, "Ne":1, 
    "Na":2, "Mg":1, "Al":2, "Si":3, "P": 4, "S": 3, "Cl":2, "Ar":1, 
    "K": 2, "Ca":1, "Sc":2, "Ti":3, "V": 4, "Cr":7, "Mn":6, "Fe":5, 
    "Co":4, "Ni":3, "Cu":2, "Zn":1, "Ga":2, "Ge":3, "As":4, "Se":3, 
    "Br":2, "Kr":1, "Rb":2, "Sr":1, "Y": 2, "Zr":3, "Nb":6, "Mo":7, 
    "Tc":6, "Ru":5, "Rh":4, "Pd":1, "Ag":2, "Cd":1, "In":2, "Sn":3, 
    "Sb":4, "Te":3, "I" :2, "Xe":1, "Cs":2, "Ba":1, "La":2, "Ce":1, 
    "Pr":4, "Nd":5, "Pm":6, "Sm":7, "Eu":8, "Gd":9, "Tb":6, "Dy":5,
    "Ho":4, "Er":3, "Tm":2, "Yb":1, "Lu":2, "Hf":3, "Ta":4, "W" :5,
    "Re":6, "Os":5, "Ir":4, "Pt":3, "Au":2, "Hg":1, "Tl":2, "Pb":3, 
    "Bi":4, "Po":3, "At":2, "Rn":1, "Fr":2, "Ra":1, "Ac":2, "Th":3, 
    "Pa":4, "U" :5, "Np":6, "Pu":7, "Am":8, "Cm":9, "Bk":6, "Cf":5, 
    "Es":4, "Fm":3
}

"""Dictionary of the ita code/name pair.
"""
ITA_DICT = {
    1 : 'shannon_entropy',
    2 : 'fisher_information',
    3 : 'alternative_fisher_information',
    4 : 'renyi_entropy',
    5 : 'tsallis_entropy',
    6 : 'onicescu_information',
    7 : 'GBP_entropy',
    8 : 'G1',
    9 : 'G2',
    10 : 'G3',
}

"""Dictionary of the kinetic energy code/name pair.
"""
KE_DICT = {
    1 : 'general',
    2 : 'thomas_fermi',
    3 : 'dirac',
    4 : 'weizsacker',
    5 : 'gradient_expansion',
    6 : 'single_particle'
}