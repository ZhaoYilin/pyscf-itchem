import os
from functools import partial
import numpy as np

__version__ = '0.1.0'

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])

from pyscf.itchem.dens import *
from pyscf.itchem.ked import *

from pyscf.itchem.aim import *
from pyscf.itchem.promolecule import *

from pyscf.itchem.eval_dens import *
from pyscf.itchem.eval_odm import *

from pyscf.itchem.utils import *
from pyscf.itchem.kernel import *