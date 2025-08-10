import numpy as np

__all__ = ['FCHK', 'fchk']


# The following items are among those currently defined:

FIELDS = [
# Molecule information
'IOpCl',
'IROHF',
'Number of atoms',
'Charge',
'Multiplicity',
'Number of electrons',
'Number of alpha electrons',
'Number of beta electrons',
'Number of basis functions',
'Number of independent functions',
'Atomic numbers',
'Nuclear charges',
'Current cartesian coordinates',
'Integer atomic weights',
'Real atomic weights',
# Basis set information
'Number of primitive shells',
'Number of contracted shells',
'Pure/Cartesian d shells',               
'Pure/Cartesian f shells',   
'Highest angular momentum',
'Largest degree of contraction',
'Shell types',
'Number of primitives per shell',
'Shell to atom map',
'Primitive exponents',
'Contraction coefficients',
'P(S=P) Contraction coefficients',
'Coordinates of each shell',
# Wavefunction information
'Alpha Orbital Energies',
'Beta Orbital Energies',
'Alpha MO coefficients',
'Beta MO coefficients',
'Total SCF Density',
'Spin SCF Density',
'Total MP2 Density',
'Spin MP2 Density',
'Total CI Density',
'Spin CI Density',
'Total CC Density',
'Spin CC Density',
# Energy information
'SCF Energy',
'Total Energy'
]

class FCHK:
    """ Class to load and dump wavefunction information in Gaussian FCHK format.
    """  
    def __init__(self, title='', route=[], dataset={}):
        """ Initiaize

        Parameters
        ----------
        title : str
            Destination file name for FCHK file.
        route : List
        """
        self.title = title  
        self.route = route    
        self.dataset = dataset

    @classmethod
    def from_pyscf(cls, title, scf, post_scf=None):
        """_summary_

        Parameters
        ----------
        filename : str
            Destination file name for FCHK file.
        scf : _type_
            _description_
        post_scf : _type_, optional
            _description_, by default None
        """       

        mol = scf.mol
        dataset = {}

        # Header section
        if post_scf is None:
            from pyscf.dft.rks import KohnShamDFT
            method = scf.__class__.__name__
            if isinstance(scf, KohnShamDFT):
                method = scf.xc
        else:
            method = post_scf.__class__.__name__
        route = ['SP', method, scf.mol.basis]

        # Molecular information section
        dataset['Number of atoms'] = mol.natm
        dataset['Charge'] = mol.charge
        dataset['Multiplicity'] = mol.multiplicity
        if mol.multiplicity==1:
            dataset["IOpCl"] = 0
        else:
            dataset["IOpCl"] = 1
        if method in ['RHF','RKS']:
            dataset['IROHF'] = 0
        else:
            dataset['IROHF'] = 1
        dataset['Number of electrons'] = sum(mol.nelec)
        dataset['Number of alpha electrons'] = mol.nelec[0]
        dataset['Number of beta electrons'] = mol.nelec[1]
        dataset['Number of basis functions'] = int(mol.nao)
        dataset['Number of independent functions'] = int(mol.nao)
        dataset['Atomic numbers'] = mol.atom_charges()
        nuc_charges = np.zeros(mol.natm)
        for i in range(mol.natm):
            nuc_charges[i]=mol.atom_nelec_core(i)+mol.atom_charge(i)
        dataset['Nuclear charges'] = nuc_charges
        dataset['Current cartesian coordinates'] = mol.atom_coords()
        dataset['Integer atomic weights'] = mol.atom_mass_list()
        dataset['Real atomic weights'] = mol.atom_mass_list().astype(float)

        # Basis set information section
        shell_types = []
        num_primitives_per_shell = []
        shell_atom_map = []

        exps_list = []
        coefs_list = []
        coords_list = []

        for atom_id, (atom_symbol, atom_coord) in enumerate(mol._atom):
            atom_basis = mol._basis[atom_symbol]
            for orb in atom_basis:
                shell_types.append(orb[0])
                num_primitives_per_shell.append(len(orb[1:]))
                shell_atom_map.append(atom_id+1)
                shell= orb[0]
                coords_list += atom_coord
                for exp, coef in orb[1:]:
                    exps_list.append(exp)
                    coefs_list.append(coef)

        dataset["Number of primitive shells"] = sum(num_primitives_per_shell)
        dataset["Number of contracted shells"] = len(num_primitives_per_shell)
        dataset["Pure/Cartesian d shells"] = 1                  
        dataset["Pure/Cartesian f shells"] = 1
        dataset["Highest angular momentum"] = max([abs(shell_type) for shell_type in shell_types])
        dataset["Largest degree of contraction"] = max(num_primitives_per_shell)
        dataset["Shell types"] = shell_types
        dataset["Number of primitives per shell"] = num_primitives_per_shell
        dataset["Shell to atom map"] = shell_atom_map
        dataset["Primitive exponents"] = exps_list
        dataset["Contraction coefficients"] = coefs_list
        dataset["Coordinates of each shell"] = coords_list

        # Energy information section
        if post_scf is None:
            dataset['SCF Energy'] = float(scf.e_tot)
            dataset['Total Energy'] = float(scf.e_tot)
        else:
            dataset['SCF Energy'] = float(scf.e_tot)
            dataset['Total Energy'] = float(post_scf.e_tot)

        # Load the basis set
        if mol.multiplicity==1:
            dataset['Alpha Orbital Energies'] = scf.mo_energy
            dataset['Alpha MO coefficients'] = scf.mo_coeff
            # Convert the triangular part of a symmetric square matrix to list storage.
            dm = scf.make_rdm1()
            dm = dm[np.triu_indices(mol.nao, k = 0)]
            dataset['Total SCF Density'] = np.array(dm)
            if post_scf is not None:
                post_scf_dm = post_scf.make_rdm1()
                post_scf_dm = post_scf_dm[np.triu_indices(mol.nao, k = 0)]
                if 'MP'.lower() in post_scf.__class__.__name__.lower():
                    dataset['Total MP2 Density'] = np.array(post_scf_dm)
                elif 'CI'.lower() in post_scf.__class__.__name__.lower():
                    dataset['Total CI Density'] = np.array(post_scf_dm)
                elif 'CC'.lower() in post_scf.__class__.__name__.lower():
                    dataset['Total CC Density'] = np.array(post_scf_dm)
                else:
                    raise TypeError("The given post Hartree Fock method not supported.")
        elif mol.multiplicity>1 :
            dataset['Alpha Orbital Energies'] = scf.mo_energy[0,:]
            dataset['Beta Orbital Energies'] = scf.mo_energy[1,:]
            dataset['Alpha MO coefficients'] = scf.mo_coeff[0,:,:]
            dataset['Beta MO coefficients'] = scf.mo_coeff[1,:,:]
            dm = scf.make_rdm1()
            dm_tot = dm[0,:,:] + dm[1,:,:]
            dm_spin = dm[0,:,:] - dm[1,:,:]
            dataset['Total SCF Density'] = dm_tot[np.triu_indices(mol.nao, k = 0)]
            dataset['Spin SCF Density'] = dm_spin[np.triu_indices(mol.nao, k = 0)]
            if post_scf is not None:
                post_scf_dm = post_scf.make_rdm1()
                post_scf_dm_tot = dm[0,:,:] + dm[1,:,:]
                post_scf_dm_spin = dm[0,:,:] - dm[1,:,:]
                post_scf_dm_tot = post_scf_dm_tot[np.triu_indices(mol.nao, k = 0)]
                post_scf_dm_spin = post_scf_dm_spin[np.triu_indices(mol.nao, k = 0)]
                if 'MP'.lower() in post_scf.__class__.__name__.lower():
                    dataset['Total MP2 Density'] = np.array(post_scf_dm_tot)
                    dataset['Spin MP2 Density'] = np.array(post_scf_dm_spin)
                elif 'CI'.lower() in post_scf.__class__.__name__.lower():
                    dataset['Total CI Density'] = np.array(post_scf_dm_tot)
                    dataset['Spin CI Density'] = np.array(post_scf_dm_spin)
                elif 'CC'.lower() in post_scf.__class__.__name__.lower():
                    dataset['Total CC Density'] = np.array(post_scf_dm_tot)
                    dataset['Spin CC Density'] = np.array(post_scf_dm_spin)
                else:
                    raise TypeError("The given post Hartree Fock method not supported.")
        else:
            raise ValueError("Multiplicity must be integer larger than 0.")


        obj = cls(title, route, dataset)
        return obj

    @classmethod
    def from_file(cls, filename, fields=FIELDS):
        """_summary_

        Parameters
        ----------
        filename : _type_
            _description_
        field_labels : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        IOError
            _description_
        IOError
            _description_
        IOError
            _description_
        IOError
            _description_
        IOError
            _description_
        """        
        dataset ={}
        with open(filename, 'r') as handel:
            title = handel.readline().strip()
            route = handel.readline().strip().split()

            for line in handel:
                print(line)
                # Check if it is a field line
                if line[0].isalpha():
                    label = line[:40].strip()
                    format = line[43:].strip().split()
                    # Scalar field 
                    if len(format)==2:
                        data_type = format[0]
                        if data_type=='I':
                            data_value = int(format[1])
                        elif data_type=='R':
                            data_value = float(format[1])
                        elif data_type=='C':
                            data_value = format[1]
                        elif data_type=='L':
                            data_value = bool(format[1])
                        else:
                            raise TypeError("Data type can only be int, float, str or bool.")

                    # Array field
                    elif len(format)==3:
                        data_type = format[0]
                        data_length = int(format[2])
                        data_value = []
                        if data_type=='I':
                            nline = int(np.ceil(data_length/6.))
                            for _ in range(nline):
                                line = handel.readline()
                                data_value += [int(i) for i in line.strip().split()]
                        elif data_type=='R':
                            nline = int(np.ceil(data_length/5.))
                            for _ in range(nline):
                                line = handel.readline()
                                data_value += [float(i) for i in line.strip().split()]
                        elif data_type=='C':
                            nline = int(np.ceil(data_length/5.))
                            for _ in range(nline):
                                line = handel.readline()
                                data_value += [str(i) for i in line.strip().split()]
                        elif data_type=='L':
                            nline = int(np.ceil(data_length/72.))
                            for _ in range(nline):
                                line = handel.readline()
                                data_value += [bool(i) for i in line.strip().split()]
                        else:
                            raise TypeError("Data type can only be int, float, str or bool.")              
                    else:
                        raise ValueError("FLFLF")
                    
                    if label in fields:
                        dataset[label] = data_value
            handel.close()   

        obj = cls(title, route, dataset)
        return obj

    @property
    def dump_mol(self):
        """_summary_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        from pyscf import gto

        basis = self.route[2]
        dataset = self.dataset

        # Build Molecule
        mol = gto.Mole()
        atom_number = np.array([dataset['Atomic numbers']],dtype=np.int64)
        coords = np.array(dataset['Current cartesian coordinates'])
        coords = coords.reshape((-1,3))
        atom = np.concatenate((atom_number.T, coords),axis=1)
        atom = np.array(atom,dtype=object)
        atom[:,0] = atom[:,0].astype(int)
        atom[:,1:] = atom[:,1:].astype(float)
 
        mol.atom = atom
        mol.basis = basis
        mol.charge = dataset['Charge']
        mol.multiplicity = dataset['Multiplicity']
        mol.verbose = 0
        mol.unit = 'B'
        mol.build()

        return mol
 
    @property
    def dump_scf(self):
        mol = self.dump_mol
        method = self.route[1]
        dataset = self.dataset
        nmo = dataset['Number of basis functions']   

        # Build SCF 
        if method.find('HF') != -1:
            from pyscf import scf
            mf = getattr(scf, method)(mol)
        elif method.find('CAS') !=-1:
            from pyscf import mcscf, fci
            #ncore = mycas.ncore
            #ncas = int(mycas.ncas)
            #nelecas = mycas.nelecas
            ## Spin resolved rdm1 
            #casdm1s = mycas.fcisolver.make_rdm1s(mycas.ci,mycas.ncas, mycas.nelecas)
        else:
            from pyscf import dft
            mf = getattr(dft, method)(mol)
 
        mf.e_tot = dataset['SCF Energy']

        if 'Beta MO coefficients' in dataset.keys():
            mo_coeff_a = np.array(dataset['Alpha MO coefficients']).reshape((nmo,nmo))
            mo_coeff_b = np.array(dataset['Beta MO coefficients']).reshape((nmo,nmo))
            mf.mo_coeff = np.array([mo_coeff_a,mo_coeff_b])       
        else:
            mf.mo_coeff = np.array(dataset['Alpha MO coefficients']).reshape((nmo,nmo))

        if 'Beta Orbital Energies' in dataset.keys():
            mo_energy_a = np.array(dataset['Alpha Orbital Energies']) 
            mo_energy_b = np.array(dataset['Beta Orbital Energies']) 
            mf.mo_energy = np.array([mo_energy_a,mo_energy_b])
        else:
            mf.mo_energy = np.array(dataset['Alpha Orbital Energies']) 

        return mf

    def dump_post_scf(self):
        method = self.route[1]
        dataset = self.dataset
        mf = self.dump_scf

        # Build Post SCF 
        if dataset['IROHF']:
            print(method)
            if 'CC'.lower() in method.lower():
                from pyscf import cc
                post_scf = getattr(cc, method)(mf)
            elif 'CI'.lower() in method.lower():
                from pyscf import ci
                post_scf = getattr(ci, method)(mf)
            elif 'MP'.lower() in method.lower():
                from pyscf import mp
                post_scf = getattr(mp, method)(mf)
            elif 'CAS'.lower() in method.lower():
                pass
            elif 'DMRG'.lower() in method.lower():
                pass
            else:
                raise ValueError("fffff")        
        return post_scf

    def dump_fchk(self, filename):
        """_summary_

        Parameters
        ----------
        dataset : _type_
            _description_
        """        
        with open(filename+'.fchk', 'w') as handel:
            self.write_header(handel, self.title, self.route)

            for key, value in self.dataset.items():
                if type(value) in  (int, float, str, bool):
                    self.write_scalar(handel, key, value)
                elif type(value) in (tuple ,list, np.ndarray):
                    self.write_array(handel, key, value)

            handel.close()

    def write_header(self, handle, title, route):
        """Type is one of the following keywords:

        SP Single point

        Method is the method of computing the energy (AM1, RHF, CASSCF, MP4, etc.), 
        
        Basis is the basis set.


        Returns
        -------
        _type_
            _description_
        """
        handle.write('{0:<43s}\n'.format(title))
        handle.write('{0:>10s}{1:>30s}{2:>30s}\n'.format(*route))
      
    def write_scalar(self, handle, key, value):
        """Scalar values appear on the same line as their data label. This line consists of a string describing the data item, a flag indicating the data type, and finally the value:

        Integer scalars: Name,I,IValue, using format A40,3X,A1,5X,I12.

        Real scalars: Name,R,Value, using format A40,3X,A1,5X,E22.15.

        Character string scalars: Name,C,Value, using format A40,3X,A1,5X,A12.

        Logical scalars: Name,L,Value, using format A40,3X,A1,5X,L1.

        Parameters
        ----------
        handle : str
            Handle of the file.
        key : str
            Name of the data.
        value : int, float, str, bool
            Value of the data
        """
        if value is None:
            handle.write('{0:<43s}{1:>1s}{2:>17d}\n'.format(key, 'I', 0))
        elif type(value) is int:
            handle.write('{0:<43s}{1:>1s}{2:>17d}\n'.format(key, 'I', value))
        elif type(value) is float:
            handle.write('{0:<43s}{1:>1s}{2:>27.15E}\n'.format(key, 'R', value))
        elif type(value) is str:
            handle.write('{0:<43s}{1:>1s}{2:>17s}\n'.format(key, 'C', value))
        elif type(value) is bool:
            handle.write('{0:<43s}{1:>1s}{2:>17s}\n'.format(key, 'L', value))
        else:
            raise TypeError('Type not supported.')

    def write_array(self, handle, key, value):
        """Vector and array data sections begin with a line naming the data and giving the type and number of values, followed by the data on one or more succeeding lines (as needed):

        Integer arrays: Name,I,Num, using format A40,3X,A1,3X, N=,I12. The N= indicates that this is an array, and the string is followed by the number of values. The array elements then follow starting on the next line in format 6I12.

        Real arrays: Name,R,Num, using format A40,3X,A1,3X,N=,I12, where the N= string again indicates an array and is followed by the number of elements. The elements themselves follow on succeeding lines in format 5E16.8. Note that the Real format has been chosen to ensure that at least one space is present between elements, to facilitate reading the data in C.

        Character string arrays (first type): Name,C,Num, using format A40,3X,A1,3X,N=,I12, where the N= string indicates an array and is followed by the number of elements. The elements themselves follow on succeeding lines in format 5A12.

        Character string arrays (second type): Name,H,Num, using format A40,3X,A1,3X,N=,I12, where the N= string indicates an array and is followed by the number of elements. The elements themselves follow on succeeding lines in format 9A8.

        Logical arrays: Name,H,Num, using format A40,3X,A1,3X,N=,I12, where the N= string indicates an array and is followed by the number of elements. The elements themselves follow on succeeding lines in format 72L1.

        Parameters
        ----------
        key : str
            Name of the data.

        value : ndarray
            Value of the data
        """
        if type(value)==np.ndarray:
            value = value.flatten().tolist() 
        n = len(value)
        data_type = type(value[0])

        if data_type is int:
            flatten_value = [value[i:i+6] for i in range(0, n, 6)]
            handle.write('{0:<43s}{1:<1s}{2:>5s}{3:>12d}\n'.format(key, 'I', 'N=', n))
            for line in flatten_value:
                line =["{:>12d}".format(value) for value in line]
                handle.write(''.join(line)+'\n')

        elif data_type is float:
            flatten_value = [value[i:i+5] for i in range(0, n, 5)]
            handle.write('{0:<43s}{1:<1s}{2:>5s}{3:>12d}\n'.format(key, 'R', 'N=', n))
            for line in flatten_value:
                line =["{:16.8E}".format(value) for value in line]
                handle.write(''.join(line)+'\n') 

        elif data_type is str:
            flatten_value = [value[i:i+6] for i in range(0, n, 6)]
            handle.write('{0:<43s}{1:<1s}{2:>5s}{3:>12d}\n'.format(key, 'C', 'N=', n))
            for line in flatten_value:
                line =["{:16s}".format(value) for value in line]
                handle.write(''.join(line)+'\n') 

        elif data_type is bool:
            flatten_value = [value[i:i+72] for i in range(0, n, 72)]
            handle.write('{0:<43s}{1:<1s}{2:>5s}{3:>12d}\n'.format(key, 'L', 'N=', n))
            for line in flatten_value:
                line =["{:1s}".format(value) for value in line]
                handle.write(''.join(line)+'\n') 
        else:
            raise TypeError('Type not supported.')

fchk = FCHK()        


def build_pyscf_system_from_gaussian_fchk(gaussian_fchk_file_path):
    from pyscf import gto

    fchk_data = parse_fchk(gaussian_fchk_file_path)
    print(fchk_data)
    mol = gto.M(atom=fchk_data['atom'])

    mf = scf.RHF(mol)
    mf.mo_coeff = fchk_data['mo_coefficients'][:, :mol.nelectron//2]
    mf.kernel()  # This recalculates everything but uses our orbitals initially

    # Save the result to a CHK file
    chkfile_path = gaussian_fchk_file_path.rsplit('.', maxsplit=1)[0] + '.chk'
    mf.dump(chkfile_path)