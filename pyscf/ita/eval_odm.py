import numpy as np

def eval_odm1(i, rdm1s=None, rdm2s=None, seniority_zero=True):
    r"""Evaluate the one orbital reduced density matrix (1-ODM).

    Parameters
    ----------    
    i : int
        Orbital index.
    rdm1s : np.ndarray, optional
        Spin-dependent one electron reduced density matrix (rdm1aa,rdm1bb), 
        by default None.
    rdm2s : np.ndarray, optional
        Spin-dependent two electron reduced density matrix (rdm2aaaa, rdm2abba, rdm2bbbb),
        by default None.
    seniority_zero : bool, optional
        If the electronic wavefunction is restricted to the seniority-zero, 
        by default True.

    Returns
    -------
    odm1_i: np.ndarray
        The one orbital reduced density matrix of indice i. 
    """
    if seniority_zero:
        rdm1aa = rdm1s[0]
        odm1_i = np.zeros((2,2))
        odm1_i[0,0] = 1-rdm1aa[i,i]
        odm1_i[1,1] = rdm1aa[i,i]           
    else:
        rdm1aa, rdm1bb = rdm1s[0], rdm1s[1]
        rdm2abba = rdm2s[1]
        odm1_i = np.zeros((4,4))
        odm1_i[0,0] = 1-rdm1aa[i,i]-rdm1bb[i,i]+rdm2abba[i,i,i,i]
        odm1_i[1,1] = rdm1aa[i,i]-rdm2abba[i,i,i,i]           
        odm1_i[2,2] = rdm1bb[i,i]-rdm2abba[i,i,i,i]           
        odm1_i[3,3] = rdm2abba[i,i,i,i]           
    return odm1_i


def eval_odm2(i, j, rdm1s=None, rdm2s=None, rdm3s=None, rdm4s=None, seniority_zero=True):
    r"""Evaluate the two orbital reduced density matrix (2-ODM).

    Parameters
    ----------    
    i : int
        Orbital index.
    j : int
        Orbital index.
    rdm1s : np.ndarray, optional
        Spin-dependent one electron reduced density matrix (rdm1aa,rdm1bb), 
        by default None.
    rdm2s : np.ndarray, optional
        Spin-dependent two electron reduced density matrix (rdm2aaaa, rdm2abba, 
        rdm2bbbb), by default None.
    rdm3s : np.ndarray, optional
        Spin-dependent three electron reduced density matrix (rdm3aaaaaa, rdm3aabbaa, 
        rdm3abbbba, rdm3bbbbbb), by default None.
    rdm4s : np.ndarray, optional
        Spin-dependent four electron reduced density matrix (rdm4aaaaaaaa, rdm4aaabbaaa, 
        rdm4aabbbbaa, rdm4abbbbbba, rdm4bbbbbbbb), by default None.
    seniority_zero : bool, optional
        If the electronic wavefunction is restricted to the seniority-zero, 
        by default True.

    Returns
    -------
    odm2_ij: np.ndarray
        The two orbital reduced density matrix of indice ij. 
    """
    if seniority_zero:
        rdm1aa = rdm1s[0]
        rdm2abba = rdm2s[1]
        odm2_ij = np.zeros((4,4))
        odm2_ij[0,0] = (
            1.0
            - rdm1aa[i,i]
            - rdm1aa[j,j]
            + rdm2abba[i,j,j,i]
        )
        odm2_ij[1,1] = rdm1aa[i,i] - rdm2abba[i,j,j,i]
        odm2_ij[1,2] = rdm2abba[i,i,j,j]
        odm2_ij[2,1] = rdm2abba[j,j,i,i]
        odm2_ij[2,2] = rdm1aa[j,j]- rdm2abba[j,i,i,j]
        odm2_ij[3,3] = rdm2abba[i,j,j,i]
    else:
        rdm1aa, rdm1bb = rdm1s[0], rdm1s[1]

        rdm2aaaa, rdm2abba, rdm2bbbb = rdm2s[0], rdm2s[1], rdm2s[2]
        rdm2baab = rdm2abba.transpose(2,3,0,1) 

        rdm3aabbaa, rdm3abbbba = rdm3s[1], rdm3s[2]
        rdm3abaaba = rdm3aabbaa.transpose(0,3,4,1,2,5)
        rdm3baaaab = rdm3aabbaa.transpose(3,4,5,0,1,2)
        rdm3babbab = rdm3abbbba.transpose(4,5,2,3,0,1)

        rdm4aabbbbaa = rdm4s[2]
        rdm4ababbaba = rdm4aabbbbaa.transpose(0,2,1,3,4,6,5,7)

        # (1,1)
        odm2_ij = np.zeros((16,16))
        # submatrix of shape 1*1                          
        # 1 (1,1)                          
        odm2_ij[0,0] = (
            1.0
            - rdm1aa[i,i]
            - rdm1bb[i,i]
            - rdm1aa[j,j]
            - rdm1bb[j,j]
            + rdm2abba[i,i,i,i]
            + rdm2abba[j,j,j,j]
            + rdm2aaaa[i,j,j,i]
            + rdm2abba[i,j,j,i]
            + rdm2baab[i,j,j,i]
            + rdm2bbbb[i,j,j,i]
            - rdm3aabbaa[i,j,j,j,j,i]
            - rdm3abbbba[i,j,j,j,j,i]
            - rdm3abbbba[i,i,j,j,i,i]
            + rdm4ababbaba[i,i,j,j,j,j,i,i]
        )               
        # submatrix of shape 2*2                           
        # 2 (2,2)                           
        odm2_ij[1,1] = (
            rdm1aa[j,j]
            - rdm2aaaa[i,j,j,i]
            - rdm2baab[i,j,j,i]
            - rdm2abba[j,j,j,j]
            + rdm3abaaba[i,j,j,j,j,i] 
            + rdm3aabbaa[i,j,j,j,j,i]
            + rdm3babbab[i,j,j,j,j,i]
            - rdm4ababbaba[i,i,j,j,j,j,i,i]
        )   
        # 3 (2,3)              
        odm2_ij[1,2] = (
            - rdm1aa[j,i]
            + rdm2baab[i,j,i,i]
            + rdm2abba[j,j,j,i]
            - rdm3abbbba[i,j,j,j,j,i]
        )
        # 4 (3,2) = (2,3)            
        odm2_ij[2,1] = odm2_ij[1,2]
        # 5 (3,3)
        odm2_ij[2,2] = (
            rdm1aa[i,i]
            - rdm2abba[i,i,i,i]
            - rdm2aaaa[i,j,j,i]
            - rdm2abba[i,j,j,i]
            + rdm3aabbaa[i,j,j,j,j,i] 
            + rdm3abaaba[i,i,j,j,i,i]
            + rdm3abbbba[i,i,j,j,i,i]
            - rdm4ababbaba[i,i,j,j,j,j,i,i]

        ) 
        # submatrix of shape 2*2                           
        # 6 (4,4)
        odm2_ij[3,3] = (
            rdm1bb[j,j]
            - rdm2abba[i,j,j,i]
            - rdm2bbbb[i,j,j,i]
            - rdm2abba[j,j,j,j]
            + rdm3abbbba[i,i,j,j,i,i] 
            + rdm3aabbaa[i,j,j,j,j,i]
            + rdm3babbab[i,j,j,j,j,i]
            - rdm4ababbaba[i,i,j,j,j,j,i,i]
        )   
        # 7 (4,5)
        odm2_ij[3,4] = (
            - rdm1bb[j,i]
            + rdm2abba[i,j,j,i]
            + rdm2abba[j,j,i,j]
            - rdm3aabbaa[i,j,j,i,j,i] 
        )                       
        # 8 (5,4)=(4,5)
        odm2_ij[4,3] = odm2_ij[3,4]
        # 9 (5,5)
        odm2_ij[4,4] = (
            rdm1bb[i,i]
            - rdm2baab[i,j,j,i]
            - rdm2bbbb[i,j,j,i]
            - rdm2abba[i,i,i,i]
            + rdm3babbab[i,j,j,j,j,i] 
            + rdm3abaaba[i,i,j,j,i,i]
            + rdm3abbbba[i,i,j,j,i,i]
            - rdm4ababbaba[i,i,j,j,j,j,i,i]
        )                   
        # submatrix of shape 2*2                           
        # 10 (6,6)
        odm2_ij[5,5] = (
            rdm2aaaa[i,j,j,i]
            - rdm3abaaba[i,i,j,j,i,i]
            - rdm3aabbaa[i,j,j,j,j,i]
            + rdm4ababbaba[i,i,j,j,j,j,i,i]
        )     
        # 11 (7,7)
        odm2_ij[6,6] = (
            rdm2bbbb[i,j,j,i]
            - rdm3abbbba[i,j,i,i,j,i]
            - rdm3abbbba[j,j,i,i,j,j]
            + rdm4ababbaba[i,i,j,j,j,j,i,i]
        )          
        # submatrix of shape 4*4                           
        # 12 (8,8)
        odm2_ij[7,7] = (
            rdm2abba[j,j,j,j]
            - rdm3aabbaa[i,j,j,j,j,i]
            - rdm3babbab[i,j,j,j,j,i]
            + rdm4ababbaba[i,i,j,j,j,j,i,i]
        )                       
        # 13 (8,9)
        odm2_ij[7,8] = (
            - rdm2abba[j,j,j,i]
            + rdm3abbbba[j,i,j,i,j,i]
        )                        
        # 14 (8,10)
        odm2_ij[7,9] = (
            - rdm2abba[j,j,i,j]
            + rdm3aabbaa[i,j,j,i,j,i]                    
        )                       
        # 15 (8,11)
        odm2_ij[7,10] = (
            - rdm2abba[j,j,i,i]
        )                       
        # 16 (9,8)
        odm2_ij[8,7] = odm2_ij[7,8]
        # 17 (9,9)
        odm2_ij[8,8] = (
            rdm2abba[i,j,j,i]
            - rdm3abaaba[i,i,j,j,i,i]
            - rdm3abaaba[i,j,j,j,j,i]
            + rdm4ababbaba[i,i,j,j,j,j,i,i]                 
        )                       
        # 18 (9,10)
        odm2_ij[8,9] = (
            - rdm2abba[i,j,i,j]
        )                       
        # 19 (9,11)
        odm2_ij[8,10] = (
            rdm2abba[i,j,i,i]
            - rdm3aabbaa[i,j,j,i,j,i]
        )                   
        # 20 (10,8)=(8,10)
        odm2_ij[9,7] = odm2_ij[7,9]
        # 21 (10,9)=(9,10)
        odm2_ij[9,8] = odm2_ij[8,9]
        # 22 (10,10)
        odm2_ij[9,9] = (
            rdm2abba[j,i,i,j]
            - rdm3abaaba[i,i,j,j,i,i]
            - rdm3babbab[i,j,j,j,j,i]
            + rdm4ababbaba[i,i,j,j,j,j,i,i]
        )                       
        # 23 (10,11)
        odm2_ij[9,10] = (
            rdm2abba[j,i,i,j]
            - rdm3abbbba[j,i,j,j,i,i]                    
        )                       
        # 24 (11,8)=(8,11)
        odm2_ij[10,7] = odm2_ij[7,10]                    
        # 25 (11,9)=(9,11)
        odm2_ij[10,8] = odm2_ij[8,10]                    
        # 26 (11,10)=(10,11)
        odm2_ij[10,9] = odm2_ij[9,10]    
        # 27 (11,11)
        odm2_ij[10,10] = (
            rdm2abba[j,j,j,j]
            - rdm3aabbaa[i,j,j,j,j,i] 
            - rdm3babbab[i,j,j,j,j,i] 
            + rdm4ababbaba[i,i,j,j,j,j,i,i]
        )                   
        # submatrix of shape 2*2
        # 28 (12,12)
        odm2_ij[11,11] = (
            rdm3aabbaa[i,j,j,j,j,i] 
            - rdm4ababbaba[i,i,j,j,j,j,i,i]
        )                   
        # 29 (12,13)
        odm2_ij[11,12] = (
            - rdm3aabbaa[i,j,j,i,j,i] 
        )                   
        # 30 (13,11)
        odm2_ij[12,11] = odm2_ij[11,12]
        # 31 (13,13)
        odm2_ij[12,12] = (
            rdm3abaaba[i,i,j,j,i,i] 
            - rdm4ababbaba[i,i,j,j,j,j,i,i]
        )                   
        # submatrix of shape 2*2
        # 32 (14,14)
        odm2_ij[13,13] = (
            rdm3babbab[i,j,j,j,j,i] 
            - rdm4ababbaba[i,i,j,j,j,j,i,i]
        )                   
        # 33 (14,15)
        odm2_ij[13,14] = (
            rdm3baaaab[i,i,j,j,j,i] 
        )                   
        # 34 (15,14)
        odm2_ij[14,13] = odm2_ij[13,14]
        # 35 (15,15)
        odm2_ij[14,14] = (
            rdm3abbbba[i,i,j,j,i,i] 
            - rdm4ababbaba[i,i,j,j,j,j,i,i]
        )                   
        # submatrix of shape 1*1
        # 35 (16,16)
        odm2_ij[15,15] = rdm4ababbaba[i,i,j,j,j,j,i,i]

    return odm2_ij