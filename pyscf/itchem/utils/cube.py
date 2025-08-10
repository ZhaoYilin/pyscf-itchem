import time
import numpy as np
from pyscf import itchem

'''
Gaussian cube file format.  Reference:
http://paulbourke.net/dataformats/cube/
https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html
http://gaussian.com/cubegen/

The output cube file has the following format

Comment line
Comment line
N_atom Ox Oy Oz         # number of atoms, followed by the coordinates of the cube origin
N1 vx1 vy1 vz1          # number of grids along each axis, followed by the step size in x/y/z direction.
N2 vx2 vy2 vz2          # ...
N3 vx3 vy3 vz3          # ...
Atom1 Z1 x y z          # Atomic number, charge, and coordinates of the atom
...                     # ...
AtomN ZN x y z          # ...
Data on grids           # (N1*N2) lines of records, each line has N3 elements
'''

class Cube(object):
    r"""Class for read-write of the Gaussian CUBE files.

    Attributes
    ----------
    coords : np.ndarray((Ngrids,3), dtype=float), optional
        Grids coordinates on N points, by default None.                 
    weights : np.ndarray((Ngrids), dtype=float), optional
        Grids weights on N points, by default None. Denoting the volume/area of the 
        uniform grid by :math:`V`.
    """
    coords = None
    weights = None
    def __init__(self, 
        mol, 
        shape=None,
        resolution=0.2,
        extension=5.0,
        margin=False
    ):
        r"""Construct the UniformGrid object in either two or three dimensions.

        Parameters
        ----------
        mol : Mole
            Pyscf Mole instance.
        shape : [int, int, int], optional
            Number of grid points along each axix.
        resolution : float, optional
            Increment between grid points along each axes direction, by default 0.2.
        extension : float, optional
            The extension of the length of the cube on each side of the molecule, by default 5.0.
        margin : bool, optional
            Bool value to indicating type of weighting function. This can be:

            True :
                The weights are the standard Riemannian weights,

                .. math::
                    \begin{align*}
                        w_{ij} &= \frac{V}{M_x \cdot M_y} \tag{Two-Dimensions} \\
                         w_{ijk} &= \frac{V}{M_x\cdot M_y \cdot M_z}  \tag{Three-Dimensions}
                    \end{align*}

                where :math:`V` is the volume or area of the uniform grid.

            False :
                The weights are the Trapezoid weights, equivalent to rectangle rule 
                with the assumption function is zero on the margin.

                 .. math::
                    \begin{align*}
                        w_{ij} &= \frac{V}{(M_x + 1) \cdot (M_y + 1)}  \tag{Two-Dimensions} \\
                        w_{ijk} &= \frac{V}{(M_x + 1) \cdot (M_y + 1) \cdot (M_z + 1)}
                        \tag{Three-Dimensions}
                    \end{align*}
                where :math:`V` is the volume or area of the uniform grid.
        """
        self.mol = mol
        self.extension = extension
        self.margin = margin
        if shape is not None:
            self.shape = shape
            print(self.resolution_from_shape(shape))
            self.resolution = self.resolution_from_shape(shape)
        else:
            self.resolution = resolution        
            self.shape = self.shape_from_resolution(resolution)

    @property
    def axes(self):
        """The spatial directions in Cartesian coordinates. These axes define the 
        orientation and arrangement of the grid points in space.
            The :math:`D` vectors, stored as rows of axes array, whose rows
            define the Cartesian coordinate system used to build the
            cubic grid, i.e. the directions of the "(x,y)\(x,y,z)"-axis
            whose norm tells us the distance between points in that direction.   

        Returns
        -------
        axes : np.ndarray((3,3), dtype=float)
            The vectors along the x, y, and z axes define the grid directions.
        """
        axes = np.diag([self.resolution, self.resolution, self.resolution])
        return axes

    @property
    def origin(self):
        """Cartesian coordinates of the grid origin.

        Returns
        -------
        origin : np.ndarray((3,), dtype=float)
            Cartesian coordinates of the uniform grid origin.
        """        
        mol = self.mol
        # calculate center of mass of the nuclear charges:
        totz = np.sum(mol.atom_charges())
        com = np.dot(mol.atom_charges(), mol.atom_coords()) / totz

        # Compute the unit vectors of the cubic grid's coordinate system
        axes = self.axes

        # maximum and minimum value of x, y and z coordinates/grid.
        max_coordinate = np.amax(mol.atom_coords(), axis=0)
        min_coordinate = np.amin(mol.atom_coords(), axis=0)
        # Compute the required number of points along x, y, and z axis
        shape = (max_coordinate - min_coordinate + 2.0 * self.extension) / self.resolution
        # Add one to include the upper-bound as well.
        shape = np.ceil(shape)
        shape = np.array(shape, int)
        # Compute origin by taking the center of mass then subtracting the half of the number
        #    of points in the direction of the axes.
        origin = com - np.dot((0.5 * shape), axes)
        return origin

    def shape_from_resolution(self, resolution):
        """_summary_

        Parameters
        ----------
        resolution : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """        
        # maximum and minimum value of x, y and z coordinates/grid.
        max_coordinate = np.amax(self.mol.atom_coords(), axis=0)
        min_coordinate = np.amin(self.mol.atom_coords(), axis=0)
        # Compute the required number of points along x, y, and z axis
        shape = (max_coordinate - min_coordinate + 2.0 * self.extension) / resolution
        # Add one to include the upper-bound as well.
        shape = np.ceil(shape)
        shape = np.array(shape, int)        

        return shape

    def resolution_from_shape(self, shape):
        """_summary_

        Parameters
        ----------
        shape : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """        
        # maximum and minimum value of x, y and z coordinates/grid.
        max_coordinate = np.amax(self.mol.atom_coords(), axis=0)
        min_coordinate = np.amin(self.mol.atom_coords(), axis=0)
        # Compute the required number of points along x, y, and z axis
        resolution = (max_coordinate - min_coordinate + 2.0 * self.extension) / shape    

        return resolution
    
    def build_grids(self,
        mol=None, 
        shape=None,
        resolution=0.2,
        extension=5.0,
        margin=False
    ):
        r"""Construct a uniform grid with evenly-spaced points in each axes.

        Grid whose points in each (x, y, z) direction has a constant step-size/evenly
        spaced. Given a origin :math:`\mathbf{o} = (o_x, o_y, o_z)` and three directions forming
        the axes :math:`\mathbf{a_1}, \mathbf{a_2}, \mathbf{a_3}` with shape
        :math:`(M_x, M_y, M_z)`, then the :math:`(i, j, k)-\text{th}` point of the grid are:

        .. math::
            x_i &= o_x + i \mathbf{a_1} \quad 0 \leq i \leq M_x \\
            y_i &= o_y + j \mathbf{a_2} \quad 0 \leq j \leq M_y \\
            z_i &= o_z + k \mathbf{a_3} \quad 0 \leq k \leq M_z

        The grid enumerates through the z-axis first, then y-axis then x-axis (i.e.,
        lexicographical ordering).

        Parameters
        ----------
        mol : Mole
            Pyscf Mole instance.
        shape : [int, int, int], optional
            Number of grid points along each axix.
        resolution : float, optional
            Increment between grid points along each axes direction, by default 0.2.
        extension : float, optional
            The extension of the length of the cube on each side of the molecule, by default 5.0.
        margin : bool, optional
            Bool value to indicating type of weighting function. This can be:

            True :
                The weights are the standard Riemannian weights,

                .. math::
                    \begin{align*}
                        w_{ij} &= \frac{V}{M_x \cdot M_y} \tag{Two-Dimensions} \\
                         w_{ijk} &= \frac{V}{M_x\cdot M_y \cdot M_z}  \tag{Three-Dimensions}
                    \end{align*}

                where :math:`V` is the volume or area of the uniform grid.

            False :
                The weights are the Trapezoid weights, equivalent to rectangle rule 
                with the assumption function is zero on the margin.

                 .. math::
                    \begin{align*}
                        w_{ij} &= \frac{V}{(M_x + 1) \cdot (M_y + 1)}  \tag{Two-Dimensions} \\
                        w_{ijk} &= \frac{V}{(M_x + 1) \cdot (M_y + 1) \cdot (M_z + 1)}
                        \tag{Three-Dimensions}
                    \end{align*}
                where :math:`V` is the volume or area of the uniform grid.            
        """
        if mol is None: mol = self.mol
        else: self.mol = mol
        if shape is None: shape = self.shape
        else: self.shape = shape
        if resolution is None: resolution = self.resolution
        else: self.resolution = resolution
        if extension is None: extension = self.extension
        else: self.extension = extension
        if margin is None: margin = self.margin
        else: self.margin = margin

        axes = self.axes
        origin = self.origin

        coords = np.array(
            np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
            )
        coords = np.swapaxes(coords, 1, 2)
        coords = coords.reshape(3, -1)
        coords = coords.T.dot(axes) + origin

        volume = np.dot(
            np.cross(shape[0] * axes[0], shape[1] * axes[1]),
            shape[2] * axes[2],
        )
        numpnt = 1.0 * np.prod(shape)
        weights = np.full(np.prod(shape), volume / numpnt)
                
        # assign attributes
        self.coords = coords
        self.weights = weights

        return self
    
    @classmethod
    def read(cls, cube_file, margin=False):
        r"""Initialize ``UniformGrid`` class based on the grid specifications of a cube file.

        Parameters
        ----------
        cube_file : str
            Cube file name with \*.cube extension.
        margin : bool, optional
            Bool value to indicating type of weighting function. This can be: 

        Returns
        -------
        cube : Cube
            Instance of Cube class generate from given cube file.
        field : np.ndarray((Ngrids), dtype=float)
            Data read from given cube file. 
        """
        from pyscf import gto

        if not cube_file.endswith(".cube"):
            raise ValueError("Argument fname should be a cube file with *.cube extension!")

        with open(cube_file) as f:
            # skip the title and second line
            f.readline()
            f.readline()

            def read_grid_line(line):
                """Read a number and (x, y, z) coordinate from the cube file line."""
                words = line.split()
                return (
                    int(words[0]),
                    np.array([float(words[1]), float(words[2]), float(words[3])], float),
                    # all coordinates in a cube file are in atomic units
                )

            # number of atoms and origin of the grid
            natom, origin = read_grid_line(f.readline())

            # number of grid points in A direction and step vector A, and so on
            nx, axis_x = read_grid_line(f.readline())
            ny, axis_y = read_grid_line(f.readline())
            nz, axis_z = read_grid_line(f.readline())
            shape = np.array([nx, ny, nz], int)
            axes = np.array([axis_x, axis_y, axis_z])


            # read molecular from cube file
            atoms = []
            for _ in range(natom):
                d = f.readline().split()
                atoms.append([int(d[0]), [float(x) for x in d[2:]]])
            mol = gto.M(atom=atoms, unit='Bohr')



            # load field data stored in the cube file
            data = f.read()
            field = np.array([float(x) for x in data.split()])

        # build Cube instance and cube grids
        max_coordinate = np.amax(mol.atom_coords(), axis=0)
        min_coordinate = np.amin(mol.atom_coords(), axis=0)
        extension = (shape*np.diag(axes) +min_coordinate-max_coordinate)/2.0  
        cube = cls(mol,extension=extension, margin=margin, shape=shape)
        cube.build_grids()
                
        return cube, field


    def write(self, field, fname, comment=None, pseudo_numbers=None):
        r"""Write the data evaluated on grid points into a cube file.

        Parameters
        ----------
        field : np.ndarray((Ngrids,), dtype=float)
            Given value in finite field space.
        fname : str
            Cube file name with \*.cube extension.
        comment : str, optional
            Second cube file commnet line to specity the type of cube data, by default None.
        pseudo_numbers : np.ndarray, shape (M,), optional
            Pseudo-numbers (core charges) of :math:`M` atoms in the molecule.            
        """
        mol = self.mol

        if not fname.endswith(".cube"):
            raise ValueError("Argument fname should be a cube file with *.cube extension!")
        if comment is None:
            comment = 'Cube file created with PySCF-ITA'
        if pseudo_numbers is None:
            pseudo_numbers = mol.atom_charges()            

        # Write data into the cube file
        with open(fname, "w") as f:
            # writing the cube header:
            f.write(comment+'\n')
            f.write(f'PySCF-ITA Version: {itchem.__version__}  Date: {time.ctime()}\n')        

            # NAtoms, X-Origin, Y-Origin, Z-Origin
            x, y, z = self.origin
            f.write(f"{mol.natm:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")

            # Number of point on each axes, and vector to represente the axes
            for i, (x, y, z) in zip(self.shape, self.axes):
                f.write(f"{i:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")

            # Atomic number, charge, and coordinates of each atom
            for p, q, (x, y, z) in zip(mol.atom_charges(), pseudo_numbers, mol.atom_coords()):
                f.write(f"{p:5d} {q:11.6f} {x:11.6f} {y:11.6f} {z:11.6f}\n")

            # writing the cube data:
            num_chunks = 6
            for i in range(0, field.size, num_chunks):
                row_data = field.flat[i : i + num_chunks]
                f.write((row_data.size * " {:12.5E}").format(*row_data))
                f.write("\n")