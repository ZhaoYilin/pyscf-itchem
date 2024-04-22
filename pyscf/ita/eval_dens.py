import numpy as np
from pyscf import dft

ni = dft.numint.NumInt()

def eval_rhos(mol, grids, rdm1, deriv=0, batch_mem=100):
    """ Method to generate the one electron density of the molecule at the desired points.

    Parameters
    ----------
    mol : Mole
        Pyscf Mole instance.
    grids : Grids
        Pyscf Grids instance.
    rdm1 : np.ndarray((Nao,Nao), dtype=float)
        One electron reduced matrix, where Nao is the number of orbitals.
    deriv : int, optional
        Electron density derivative level, by default 0.
    batch_mem : int, optional
        Block memory in each loop, by default 100

    Returns
    -------
    rhos : np.ndarray((M, N), dtype=float)
        2D array of shape (M, N) to store electron density. 
        For deriv=0, 2D array of (1,N) to store density;  
        For deriv=1, 2D array of (4,N) to store density and density derivatives 
        for x,y,z components; 
        For deriv=2, 2D array of (5,N) array where last rows are nabla^2 rho.
    """    
    if deriv==0:
        rho = np.zeros((1,grids.weights.size))
        batch_start, batch_end = 0, 0
        # aos, mask, weights, coords
        for aos, _, _, _ in ni.block_loop(mol, grids, mol.nao, deriv, batch_mem):   
            ao = aos  
            batch_end = batch_start + aos.shape[0]
            batch_slice = slice(batch_start, batch_end)
            rho[:,batch_slice] = eval_rho(rdm1,ao)
            batch_start = batch_end    

        rhos = rho 

    elif deriv==1:
        rho = np.zeros((1,grids.weights.size))
        rho_grad = np.zeros((3,grids.weights.size))

        batch_start, batch_end = 0, 0
        # aos, mask, weights, coords
        for aos, _, _, _ in ni.block_loop(mol, grids, mol.nao, deriv, batch_mem): 
            ao = aos[0]
            ao_grad = aos[1:4]

            batch_end = batch_start + aos.shape[1]
            batch_slice = slice(batch_start, batch_end)

            rho[:,batch_slice] = eval_rho(rdm1, ao)
            rho_grad[:,batch_slice] = eval_rho_grad(rdm1, ao, ao_grad) 

            batch_start = batch_end    

        rhos = np.concatenate([rho, rho_grad])  

    elif deriv==2:
        rho = np.zeros((1,grids.weights.size))
        rho_grad = np.zeros((3,grids.weights.size))
        rho_hess = np.zeros((3,3,grids.weights.size))

        batch_start, batch_end = 0, 0
        # aos, mask, weights, coords
        for aos, _, _, _ in ni.block_loop(mol, grids, mol.nao, deriv, batch_mem):  
            ao = aos[0]
            ao_grad = aos[1:4]
            ao_hess = np.array([
                [aos[4], aos[5], aos[6]],
                [aos[5], aos[7], aos[8]],
                [aos[6], aos[8], aos[9]],
            ])

            batch_end = batch_start + aos.shape[1]
            batch_slice = slice(batch_start, batch_end)

            rho[:,batch_slice] = eval_rho(rdm1, ao)
            rho_grad[:,batch_slice] = eval_rho_grad(rdm1, ao, ao_grad) 
            rho_hess[:,:,batch_slice] = eval_rho_hess(rdm1, ao, ao_grad, ao_hess) 

            batch_start = batch_end    

        rho_lapl = np.array([rho_hess.trace()])
        rhos = np.concatenate([rho, rho_grad, rho_lapl])

    else:
        raise ValueError("Max deriv order is 2.")

    return rhos

def eval_gammas(mol, grids, rdm2=None, deriv=0, batch_mem=5):
    """ Method to generate the one electron density of the molecule at the desired points.

    Parameters
    ----------
    mol : Mole
        Pyscf Mole instance.
    grids : Grids
        Pyscf Grids instance.
    rdm2 : np.ndarray((Nao,Nao,Nao,Nao), dtype=float)
        Two electron reduced matrix, where Nao is the number of orbitals.
    deriv : int, optional
        Electron density derivative level, by default 0.
    batch_mem : int, optional
        Block memory in each loop, by default 1.

    Returns
    -------
    gammas : np.ndarray((M, N, N), dtype=float)
        3D array of shape (M, N, N) to store electron density. 
        For deriv=0, 2D array of (1,N,N) to store density;  
        For deriv=1, 2D array of (4,N,N) to store density and density derivatives 
        for x,y,z components; 
        For deriv=2, 2D array of (5,N,N) array where last rows are nabla^2 rho.
    """  
    if deriv==0:
        gamma = np.zeros((1,grids.weights.size,grids.weights.size))
        batch_start, batch_end = 0, 0
        # aos, mask, weights, coords
        for aos, _, _, _ in ni.block_loop(mol, grids, mol.nao, deriv, batch_mem):  
            ao = aos      
            batch_end = batch_start + aos.shape[0]
            batch_slice = slice(batch_start, batch_end)
            gamma[:,batch_slice,batch_slice] = eval_gamma(rdm2,ao) 
            batch_start = batch_end   

        gammas = gamma


    elif deriv==1:
        gamma = np.zeros((1,grids.weights.size,grids.weights.size))
        gamma_grad = np.zeros((3,grids.weights.size,grids.weights.size))

        batch_start, batch_end = 0, 0
        # aos, mask, weights, coords
        for aos, _, _, _ in ni.block_loop(mol, grids, mol.nao, deriv, batch_mem):  
            ao = aos[0]
            ao_grad = aos[1:4] 

            batch_end = batch_start + aos.shape[1]
            batch_slice = slice(batch_start, batch_end)
            gamma[:,batch_slice,batch_slice] = eval_gamma(rdm2, ao)
            gamma_grad[:,batch_slice,batch_slice] = eval_gamma_grad(rdm2, ao, ao_grad) 

            batch_start = batch_end  

        gammas = np.concatenate([gamma, gamma_grad])

    elif deriv==2:
        gamma = np.zeros((1,grids.weights.size,grids.weights.size))
        gamma_grad = np.zeros((3,grids.weights.size,grids.weights.size))        
        gamma_hess = np.zeros((3,3,grids.weights.size,grids.weights.size))
        gamma_lapl = np.zeros((1,grids.weights.size,grids.weights.size))

        batch_start, batch_end = 0, 0
        # aos, mask, weights, coords
        for aos, _, _, _ in ni.block_loop(mol, grids, mol.nao, deriv, batch_mem):  
            ao = aos[0]
            ao_grad = aos[1:4]
            ao_hess = np.array([
                [aos[4], aos[5], aos[6]],
                [aos[5], aos[7], aos[8]],
                [aos[6], aos[8], aos[9]],
            ])   

            batch_end = batch_start + aos.shape[1]
            batch_slice = slice(batch_start, batch_end)

            gamma[:,batch_slice,batch_slice] = eval_gamma(rdm2, ao)
            gamma_grad[:,batch_slice,batch_slice] = eval_gamma_grad(rdm2, ao, ao_grad) 
            gamma_hess[:,:,batch_slice,batch_slice] = eval_gamma_hess(rdm2, ao, ao_grad, ao_hess) 

            batch_start = batch_end  

        gamma_lapl = np.array([gamma_hess.trace()])
        gammas = np.concatenate([gamma, gamma_grad, gamma_lapl])

    else:
        raise ValueError("Max deriv order is 2.")

    return gammas


def eval_rho(rdm1=None, ao=None):
    """ Evaluate one electron density in finite field space. 

    .. math::

        \\rho = {}^1D_{\\mu \\nu} \\phi_\\mu \\phi_\\nu

    Parameters
    ----------
    rdm1 : np.ndarray((Nao,Nao), dtype=float)
        One electron reduced matrix, where Nao is the number of orbitals.
    ao : np.ndarray((Ngrids,), dtype=float)
        Atomic orbital value in finite field space.

    Returns
    -------
    rho : np.ndarray((Ngrids,), dtype=float)
        One electron density in finite field space.
    """
    rho = np.einsum("uv, gu, gv -> g", rdm1, ao, ao)
    return rho

def eval_rho_grad(rdm1=None, ao=None, ao_grad=None):
    """ Evaluate first derivative of one electron density in finite field space. 

    .. math::

        \\rho_r = 2 * {}^1D_{\\mu \\nu} \\phi_{r \\mu} \\phi_\\nu

    Parameters
    ----------
    rdm1 : np.ndarray((Nao,Nao), dtype=float)
        One electron reduced matrix, where Nao is the number of orbitals.
    ao : np.ndarray((Ngrids,), dtype=float)
        Atomic orbital value in finite field space.
    ao_grad : np.ndarray((3, Ngrids), dtype=float)
        First derivative of atomic orbital value in finite field space.

    Returns
    -------
    rho_grad : np.ndarray((3,Ngrids), dtype=float)
        First derivative of one electron density in finite field space.
    """
    rho_grad = 2 * np.einsum("uv, rgu, gv -> rg", rdm1, ao_grad, ao)
    return rho_grad

def eval_rho_hess(rdm1=None, ao=None, ao_grad=None, ao_hess=None):
    """ Evaluate second derivative of one electron density in finite field space. 

    .. math::

        \\rho_{rw} = 2 * {}^1D_{\\mu \\nu} (\\phi_{rw \\mu} \\phi_\\nu + \\phi_{r \\mu} \\phi_{w \\nu})

    Parameters
    ----------
    rdm1 : np.ndarray((Nao,Nao), dtype=float)
        One electron reduced matrix, where Nao is the number of orbitals.
    ao : np.ndarray((Ngrids,), dtype=float)
        Atomic orbital value in finite field space.
    ao_grad : np.ndarray((3, Ngrids), dtype=float)
        First derivative of atomic orbital value in finite field space.
    ao_hess : np.ndarray((3,3,Ngrids), dtype=float)
        Second derivative of atomic orbital value in finite field space.

    Returns
    -------
    rho_hess : np.ndarray((3,3,Ngrids), dtype=float)
        Second derivative of one electron density in finite field space.
    """
    rho_hess = (
        + 2 * np.einsum("uv, rwgu, gv -> rwg", rdm1, ao_hess, ao)
        + 2 * np.einsum("uv, rgu, wgv -> rwg", rdm1, ao_grad, ao_grad)
    )
    return rho_hess

def eval_gamma(rdm2=None, ao=None):
    """ Evaluate two electron density in finite field space. 

    .. math::

        \\gamma = {}^2D_{\\mu\\nu\\kappa\\lambda} \\phi_\\mu \\phi_\\nu \\phi_\\kappa \\phi_\\lambda

    Parameters
    ----------
    rdm2 : np.ndarray((Nao,Nao,Nao,Nao), dtype=float)
        One electron reduced matrix, where Nao is the number of orbitals.
    ao : np.ndarray((Ngrids,), dtype=float)
        Atomic orbital value in finite field space.

    Returns
    -------
    gamma : np.ndarray((Ngrids,Ngrids), dtype=float)
        Two electron density in finite field space.
    """   
    gamma = np.einsum("uvkl, gu -> gvkl", rdm2, ao)
    gamma = np.einsum("gvkl, gv -> gkl", gamma, ao)
    gamma = np.einsum("gkl, hk -> ghl", gamma, ao)
    gamma = np.einsum("ghl, hl -> gh", gamma, ao)
    return gamma

def eval_gamma_grad(rdm2=None, ao=None, ao_grad=None):
    """ Evaluate first derivative of two electron density in finite field space. 

    .. math::

        \\gamma_r = 2 * {}^2D_{\\mu\\nu\\kappa\\lambda} 
        (\\phi_{r \\mu} \\phi_\\nu \\phi_\\kappa\\phi_\\lambda
        +\\phi_{\\mu} \\phi_\\nu \\phi_{r \\kappa}\\phi_\\lambda)


    Parameters
    ----------
    rdm2 : np.ndarray((Nao,Nao,Nao,Nao), dtype=float)
        Two electron reduced matrix, where Nao is the number of orbitals.
    ao : np.ndarray((Ngrids,), dtype=float)
        Atomic orbital value in finite field space.
    ao_grad : np.ndarray((3, Ngrids), dtype=float)
        First derivative of atomic orbital value in finite field space.

    Returns
    -------
    gamma_grad : np.ndarray((3,Ngrids,Ngrids), dtype=float)
        First derivative of two electron density in finite field space.
    """    
    term1 = np.einsum("uvkl, rgu -> rgvkl", rdm2, ao_grad)
    term1 = np.einsum("rgvkl, gv -> rgkl", term1, ao)
    term1 = np.einsum("rgkl, hk -> rghl", term1, ao)
    term1 = np.einsum("rghl, hl -> rgh", term1, ao)

    term2 = np.einsum("uvkl, gu -> gvkl", rdm2, ao)
    term2 = np.einsum("gvkl, gv -> gkl", term2, ao)
    term2 = np.einsum("gkl, rhk -> rghl", term2, ao_grad)
    term2 = np.einsum("rghl, hl -> rgh", term2, ao)

    gamma_grad = 2*term1 + 2*term2
    return gamma_grad

def eval_gamma_hess(rdm2=None, ao=None, ao_grad=None, ao_hess=None):
    """ Evaluate second derivative of two electron density in finite field space. 

    .. math::
        :nowrap:

        $$
        \\begin{eqnarray}
        \\gamma_{rw} &= 2 * {}^2D_{\\mu\\nu\\kappa\\lambda} 
        (\\phi_{rw \\mu}\\phi_{\\nu}\\phi_{\\kappa}\\phi_{\\lambda}
        +\\phi_{\\mu} \\phi_{\\nu}\\phi_{rw \\mu} \\phi_{\\nu}
        +\\phi_{r\\mu}\\phi_{w\\nu}\\phi_{\\kappa}\\phi_{\\lambda}\\\\
        &+2*\\phi_{r\\mu}\\phi_{\\nu}\\phi_{w\\kappa}\\phi_{\\lambda}
        +2*\\phi_{r\\mu}\\phi_{\\nu}\\phi_{\\kappa}\\phi_{w\\lambda}
        +\\phi_{\\mu}\\phi_{\\nu}\\phi_{r\\kappa}\\phi_{w\\lambda}
        )
        \\end{eqnarray}
        $$

    Parameters
    ----------
    rdm2 : np.ndarray((Nao,Nao,Nao,Nao), dtype=float)
        One electron reduced matrix, where Nao is the number of orbitals.
    ao : np.ndarray((Ngrids,), dtype=float)
        Atomic orbital value in finite field space.
    ao_grad : np.ndarray((3, Ngrids), dtype=float)
        First derivative of atomic orbital value in finite field space.
    ao_hess : np.ndarray((3,3,Ngrids), dtype=float)
        Second derivative of atomic orbital value in finite field space.

    Returns
    -------
    rho_hess : np.ndarray((3,3,Ngrids), dtype=float)
        Second derivative of one electron density in finite field space.
    """    
    term1 = np.einsum("uvkl, rwgu -> rwgvkl", rdm2, ao_hess)
    term1 = np.einsum("rwgvkl, gv -> rgkl", term1, ao)
    term1 = np.einsum("rgkl, hk -> rghl", term1, ao)
    term1 = np.einsum("rghl, hl -> rgh", term1, ao)

    term2 = np.einsum("uvkl, gu -> gvkl", rdm2, ao)
    term2 = np.einsum("gvkl, gv -> gkl", term2, ao)
    term2 = np.einsum("gkl, rwhk -> rwghl", term2, ao_hess)
    term2 = np.einsum("rwghl, hl -> rwgh", term2, ao)

    term3 = np.einsum("uvkl, rgu -> rgvkl", rdm2, ao_grad)
    term3 = np.einsum("rgvkl, wgv -> rwgkl", term3, ao_grad)
    term3 = np.einsum("rwgkl, hk -> rwghl", term3, ao)
    term3 = np.einsum("rwghl, hl -> rwgh", term3, ao)

    term4 = np.einsum("uvkl, rgu -> rgvkl", rdm2, ao_grad)
    term4 = np.einsum("rgvkl, gv -> rgkl", term4, ao)
    term4 = np.einsum("rgkl, whk -> rwghl", term4, ao_grad)
    term4 = np.einsum("rwghl, hl -> rwgh", term4, ao)

    term5 = np.einsum("uvkl, rgu -> rgvkl", rdm2, ao_grad)
    term5 = np.einsum("rgvkl, gv -> rgkl", term5, ao)
    term5 = np.einsum("rgkl, hk -> rghl", term5, ao)
    term5 = np.einsum("rghl, whl -> rwgh", term5, ao_grad)

    term6 = np.einsum("uvkl, gu -> gvkl", rdm2, ao)
    term6 = np.einsum("gvkl, gv -> gkl", term6, ao)
    term6 = np.einsum("gkl, rhk -> rghl", term6, ao_grad)
    term6 = np.einsum("rghl, whl -> rwgh", term6, ao_grad)

    gamma_hess = 2*term1+2*term2+2*term3+4*term4+4*term5+2*term6

    return gamma_hess