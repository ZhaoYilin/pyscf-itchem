import numpy as np

class Global:
    """Variables declared in the global scope are accessible from anywhere in the file 
    after their definition.
    
    Attributes
    ----------
    scalar : float, optional
        Given float value, by default None.
    """    
    scalar=None
    def __init__(self):
        NotImplemented

class Local:
    """Variables declared within a function belong to the local scope of that function.

    Attributes
    ----------
    grids_weights : np.ndarray((Ngrids,), dtype=float), optional
        Grids weights on N points, by default None.
    field : np.ndarray((Ngrids,), dtype=float), optional
        Given field, by default None.
    """    
    grids_weights=None
    field=None
    def __init__(self):
        NotImplemented
    def integral(self, 
        grids_weights=None,
        field=None
    ):
        """Integrating the given field to obtain a scalar result.

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        field : np.ndarray((Ngrids,), dtype=float), optional
            Given field, by default None.

        Returns
        -------
        result : float
            Scalar result.        
        """
        if grids_weights is None:
            grids_weights = self.grids_weights
        if field is None:
            field = self.field    
        result = np.einsum("g, g -> ", grids_weights, field)
        return result

class NonLocal:
    """The nonlocal keyword is used inside nested functions to refer to variables in the 
    nearest enclosing scope (excluding global).

    Attributes
    ----------
    grids_weights : np.ndarray((Ngrids,), dtype=float), optional
        Grids weights on N points, by default None.
    kernel : np.ndarray((Ngrids,Ngrids), dtype=float), optional
        ITA density, by default None.
    """    
    grids_weights=None
    kernel=None
    def __init__(self):
        NotImplemented
    def integral(self, 
        grids_weights=None,
        kernel=None
    ):
        """Integrating the given kernel to obtain a scalar result.

        Parameters
        ----------
        grids_weights : np.ndarray((Ngrids,), dtype=float), optional
            Grids weights on N points, by default None.
        kernel : np.ndarray((Ngrids,Ngrids), dtype=float), optional
            ITA density, by default None.

        Returns
        -------
        result : float
            Scalar result.        
        """
        if grids_weights is None:
            grids_weights = self.grids_weights
        if kernel is None:
            kernel = self.kernel  
        result = np.einsum("g, h, gh -> ", grids_weights[0], grids_weights[1], kernel)                
        return result