import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pyscf import dft

def eval_rhos(mol, grids_coords, rdm1, deriv=0, batch_size=8000, max_workers=None):
    """ Method to generate the one electron density of the molecule at the desired points.

    Parameters
    ----------
    mol : Mole
        Pyscf Mole instance.
    grids_coords : np.ndarray((Ngrids,3), dtype=float)
        Grids coordinates.
    rdm1 : np.ndarray((Nao,Nao), dtype=float)
        One electron reduced matrix, where Nao is the number of orbitals.
    deriv : int, optional
        Electron density derivative level, by default 0.
    batch_size : int, optional
        Batch size in each loop, by default 8000.
    max_workers : int, optional
        Maximum number of workers in thread pool, defaults to CPU count.

    Returns
    -------
    rhos : np.ndarray((M, Ngrids), dtype=float)
        2D array of shape (M, Ngrids) to store electron density. 
        For deriv=0, 2D array of (1,Ngrids) to store density;  
        For deriv=1, 2D array of (4,Ngrids) to store density and density derivatives 
        for x,y,z components; 
        For deriv=2, 2D array of (5,Ngrids) array where last rows are nabla^2 rho.
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    ngrids = grids_coords.shape[0]  # Number of grid points
    batch_size = min(batch_size, ngrids)

    if deriv == 0:
        rho = np.zeros((1, ngrids))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, ngrids, batch_size):
                batch_slice = slice(i, i + batch_size)
                batch_aos = dft.numint.eval_ao(mol, grids_coords[batch_slice, :], 0)
                future = executor.submit(eval_rho, rdm1, batch_aos)
                rho[:, batch_slice] = future.result()

        rhos = rho

    elif deriv == 1:
        rho = np.zeros((1, ngrids))
        rho_grad = np.zeros((3, ngrids))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, ngrids, batch_size):
                batch_slice = slice(i, i + batch_size)
                batch_aos = dft.numint.eval_ao(mol, grids_coords[batch_slice, :], 1)
                ao = batch_aos[0]
                ao_grad = batch_aos[1:4]

                future_rho = executor.submit(eval_rho, rdm1, ao)
                future_rho_grad = executor.submit(eval_rho_grad, rdm1, ao, ao_grad)

                rho[:, batch_slice] = future_rho.result()
                rho_grad[:, batch_slice] = future_rho_grad.result()

        rhos = np.concatenate([rho, rho_grad])

    elif deriv == 2:
        rho = np.zeros((1, ngrids))
        rho_grad = np.zeros((3, ngrids))
        rho_hess = np.zeros((3, 3, ngrids))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, ngrids, batch_size):
                batch_slice = slice(i, i + batch_size)
                batch_aos = dft.numint.eval_ao(mol, grids_coords[batch_slice, :], 2)

                ao = batch_aos[0]
                ao_grad = batch_aos[1:4]
                ao_hess = np.array([
                    [batch_aos[4], batch_aos[5], batch_aos[6]],
                    [batch_aos[5], batch_aos[7], batch_aos[8]],
                    [batch_aos[6], batch_aos[8], batch_aos[9]],
                ])

                future_rho = executor.submit(eval_rho, rdm1, ao)
                future_rho_grad = executor.submit(eval_rho_grad, rdm1, ao, ao_grad)
                future_rho_hess = executor.submit(eval_rho_hess, rdm1, ao, ao_grad, ao_hess)

                rho[:, batch_slice] = future_rho.result()
                rho_grad[:, batch_slice] = future_rho_grad.result()
                rho_hess[:, :, batch_slice] = future_rho_hess.result()

        rho_lapl = np.array([rho_hess.trace()])
        rhos = np.concatenate([rho, rho_grad, rho_lapl])

    else:
        raise ValueError("Max deriv order is 2.")

    return rhos

def eval_gammas(mol, grids_coords, rdm2=None, deriv=0, batch_size=4000, max_workers=None):
    """ Method to generate the pair electron density of the molecule at the desired points.

    Parameters
    ----------
    mol : Mole
        Pyscf Mole instance.
    grids_coords : [np.ndarray((Ngrids,3), dtype=float), np.ndarray((Ngrids,3), dtype=float)]
        List or tuple of grids coordinates.
    rdm2 : np.ndarray((Nao,Nao,Nao,Nao), dtype=float)
        Two electron reduced matrix, where Nao is the number of orbitals.
    deriv : int, optional
        Electron density derivative level, by default 0.
    batch_size : int, optional
        Batch size in each loop, by default 4000.
    max_workers : int, optional
        Maximum number of workers in thread pool, defaults to CPU count.

    Returns
    -------
    gammas : np.ndarray((M, Ngrids, Ngrids), dtype=float)
        3D array of shape (M, Ngrids, Ngrids) to store electron density. 
        For deriv=0, 2D array of (1,Ngrids,Ngrids) to store density;  
        For deriv=1, 2D array of (4,Ngrids,Ngrids) to store density and density derivatives 
        for x,y,z components; 
        For deriv=2, 2D array of (5,Ngrids,Ngrids) array where last rows are nabla^2 rho.
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    ngrids = [grids_coords[0].shape[0], grids_coords[1].shape[0]]  # Number of grid points
    batch_size = [min(batch_size, ngrids[0]), min(batch_size, ngrids[1])]

    if deriv == 0:
        gamma = np.zeros((1, ngrids[0], ngrids[1]))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, ngrids[0], batch_size[0]):
                batch_slice_i = slice(i, i + batch_size[0])
                ao_i = dft.numint.eval_ao(mol, grids_coords[0][batch_slice_i, :], 0)
                for j in range(0, ngrids[1], batch_size[1]):
                    batch_slice_j = slice(j, j + batch_size[1])
                    ao_j = dft.numint.eval_ao(mol, grids_coords[1][batch_slice_j, :], 0)
                    future = executor.submit(eval_gamma, rdm2, [ao_i, ao_j])

                    gamma[:, batch_slice_i, batch_slice_j] = future.result()

        gammas = gamma

    elif deriv == 1:
        gamma = np.zeros((1, ngrids[0], ngrids[1]))
        gamma_grad = np.zeros((3, ngrids[0], ngrids[1]))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, ngrids[0], batch_size[0]):
                batch_slice_i = slice(i, i + batch_size[0])
                batch_aos_i = dft.numint.eval_ao(mol, grids_coords[0][batch_slice_i, :], 1)
                ao_i = batch_aos_i[0]
                ao_grad_i = batch_aos_i[1:4]
                for j in range(0, ngrids[1], batch_size[1]):
                    batch_slice_j = slice(j, j + batch_size[1])
                    batch_aos_j = dft.numint.eval_ao(mol, grids_coords[1][batch_slice_j, :], 1)
                    ao_j = batch_aos_j[0]
                    ao_grad_j = batch_aos_j[1:4]

                    ao = [ao_i, ao_j]
                    ao_grad = [ao_grad_i, ao_grad_j]
                    future_gamma = executor.submit(eval_gamma, rdm2, ao)
                    future_gamma_grad = executor.submit(eval_gamma_grad, rdm2, ao, ao_grad)

                    gamma[:, batch_slice_i, batch_slice_j] = future_gamma.result()
                    gamma_grad[:, batch_slice_i, batch_slice_j] = future_gamma_grad.result()

        gammas = np.concatenate([gamma, gamma_grad])

    elif deriv == 2:
        gamma = np.zeros((1, ngrids[0], ngrids[1]))
        gamma_grad = np.zeros((3, ngrids[0], ngrids[1]))
        gamma_hess = np.zeros((3, 3, ngrids[0], ngrids[1]))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, ngrids[0], batch_size[0]):
                batch_slice_i = slice(i, i + batch_size[0])
                batch_aos_i = dft.numint.eval_ao(mol, grids_coords[0][batch_slice_i, :], 2)
                ao_i = batch_aos_i[0]
                ao_grad_i = batch_aos_i[1:4]
                ao_hess_i = np.array([
                    [batch_aos_i[4], batch_aos_i[5], batch_aos_i[6]],
                    [batch_aos_i[5], batch_aos_i[7], batch_aos_i[8]],
                    [batch_aos_i[6], batch_aos_i[8], batch_aos_i[9]],
                ])
                for j in range(0, ngrids[1], batch_size[1]):
                    batch_slice_j = slice(j, j + batch_size[1])
                    batch_aos_j = dft.numint.eval_ao(mol, grids_coords[1][batch_slice_j, :], 2)
                    ao_j = batch_aos_j[0]
                    ao_grad_j = batch_aos_j[1:4]
                    ao_hess_j = np.array([
                        [batch_aos_j[4], batch_aos_j[5], batch_aos_j[6]],
                        [batch_aos_j[5], batch_aos_j[7], batch_aos_j[8]],
                        [batch_aos_j[6], batch_aos_j[8], batch_aos_j[9]],
                    ])

                    ao = [ao_i, ao_j]
                    ao_grad = [ao_grad_i, ao_grad_j]
                    ao_hess = [ao_hess_i, ao_hess_j]
                    future_gamma = executor.submit(eval_gamma, rdm2, ao)
                    future_gamma_grad = executor.submit(eval_gamma_grad, rdm2, ao, ao_grad)
                    future_gamma_hess = executor.submit(eval_gamma_hess, rdm2, ao, ao_grad, ao_hess)

                    gamma[:, batch_slice_i, batch_slice_j] = future_gamma.result()
                    gamma_grad[:, batch_slice_i, batch_slice_j] = future_gamma_grad.result()
                    gamma_hess[:, :, batch_slice_i, batch_slice_j] = future_gamma_hess.result()

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
        Two electron reduced matrix, where Nao is the number of orbitals.
    ao : [np.ndarray((Ngrids,), dtype=float), np.ndarray((Ngrids,), dtype=float)]
        List or tuple of atomic orbital value in finite field space.

    Returns
    -------
    gamma : np.ndarray((Ngrids,Ngrids), dtype=float)
        Two electron density in finite field space.
    """   
    gamma = np.einsum("uvkl, gu -> gvkl", rdm2, ao[0])
    gamma = np.einsum("gvkl, gv -> gkl", gamma, ao[0])
    gamma = np.einsum("gkl, hk -> ghl", gamma, ao[1])
    gamma = np.einsum("ghl, hl -> gh", gamma, ao[1])

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
    ao : [np.ndarray((Ngrids,), dtype=float), np.ndarray((Ngrids,), dtype=float)]
        List or tuple of atomic orbital value in finite field space.
    ao_grad : [np.ndarray((3, Ngrids), dtype=float), np.ndarray((3, Ngrids), dtype=float)]
        List or tuple of first derivative of atomic orbital value in finite field space.

    Returns
    -------
    gamma_grad : np.ndarray((3,Ngrids,Ngrids), dtype=float)
        First derivative of two electron density in finite field space.
    """    
    term1 = np.einsum("uvkl, rgu -> rgvkl", rdm2, ao_grad[0])
    term1 = np.einsum("rgvkl, gv -> rgkl", term1, ao[0])
    term1 = np.einsum("rgkl, hk -> rghl", term1, ao[1])
    term1 = np.einsum("rghl, hl -> rgh", term1, ao[1])

    term2 = np.einsum("uvkl, gu -> gvkl", rdm2, ao[0])
    term2 = np.einsum("gvkl, gv -> gkl", term2, ao[0])
    term2 = np.einsum("gkl, rhk -> rghl", term2, ao_grad[1])
    term2 = np.einsum("rghl, hl -> rgh", term2, ao[1])

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
        Two electron reduced matrix, where Nao is the number of orbitals.
    ao : [np.ndarray((Ngrids,), dtype=float), np.ndarray((Ngrids,), dtype=float)]
        List or tuple of atomic orbital value in finite field space.
    ao_grad : [np.ndarray((3, Ngrids), dtype=float), np.ndarray((3, Ngrids), dtype=float)]
        List or tuple of first derivative of atomic orbital value in finite field space.
    ao_hess : [np.ndarray((3,3,Ngrids), dtype=float), np.ndarray((3,3,Ngrids), dtype=float)]
        List or tuple of second derivative of atomic orbital value in finite field space.

    Returns
    -------
    rho_hess : np.ndarray((3,3,Ngrids), dtype=float)
        Second derivative of one electron density in finite field space.
    """    
    term1 = np.einsum("uvkl, rwgu -> rwgvkl", rdm2, ao_hess[0])
    term1 = np.einsum("rwgvkl, gv -> rgkl", term1, ao[0])
    term1 = np.einsum("rgkl, hk -> rghl", term1, ao[1])
    term1 = np.einsum("rghl, hl -> rgh", term1, ao[1])

    term2 = np.einsum("uvkl, gu -> gvkl", rdm2, ao[0])
    term2 = np.einsum("gvkl, gv -> gkl", term2, ao[0])
    term2 = np.einsum("gkl, rwhk -> rwghl", term2, ao_hess[1])
    term2 = np.einsum("rwghl, hl -> rwgh", term2, ao[1])

    term3 = np.einsum("uvkl, rgu -> rgvkl", rdm2, ao_grad[0])
    term3 = np.einsum("rgvkl, wgv -> rwgkl", term3, ao_grad[0])
    term3 = np.einsum("rwgkl, hk -> rwghl", term3, ao[1])
    term3 = np.einsum("rwghl, hl -> rwgh", term3, ao[1])

    term4 = np.einsum("uvkl, rgu -> rgvkl", rdm2, ao_grad[0])
    term4 = np.einsum("rgvkl, gv -> rgkl", term4, ao[0])
    term4 = np.einsum("rgkl, whk -> rwghl", term4, ao_grad[1])
    term4 = np.einsum("rwghl, hl -> rwgh", term4, ao[1])

    term5 = np.einsum("uvkl, rgu -> rgvkl", rdm2, ao_grad[0])
    term5 = np.einsum("rgvkl, gv -> rgkl", term5, ao[0])
    term5 = np.einsum("rgkl, hk -> rghl", term5, ao[1])
    term5 = np.einsum("rghl, whl -> rwgh", term5, ao_grad[1])

    term6 = np.einsum("uvkl, gu -> gvkl", rdm2, ao[0])
    term6 = np.einsum("gvkl, gv -> gkl", term6, ao[0])
    term6 = np.einsum("gkl, rhk -> rghl", term6, ao_grad[1])
    term6 = np.einsum("rghl, whl -> rwgh", term6, ao_grad[1])

    gamma_hess = 2*term1+2*term2+2*term3+4*term4+4*term5+2*term6

    return gamma_hess
