import numpy as np

from pyscf.ita.eval_odm import eval_odm1, eval_odm2

__all__ = ["OrbitalEntanglement"]

class OrbitalEntanglement:
    r"""Orbital entanglement class.
    """
    def __init__(self, odm1=None, odm2=None):
        """ Initialize a instance.

        Parameters
        ----------
        odm1 : np.ndarray, optional
            The one orbital reduced density matrix, by default None.
        odm2 : np.ndarray, optional
            The two orbital reduced density matrix, by default None.
        """        
        self.odm1 = odm1
        self.odm2 = odm2

    @property
    def no(self):
        """Number of orbitals.
        """        
        return self.odm1.shape[0]

    def build(self, method, seniority_zero=True):
        r"""Method to build the class.

        Parameters
        ----------
        method : PyscfMethod
            Pyscf scf method or post-scf method instance.
        seniority_zero : bool, optional
            If the electronic wavefunction is restricted to the seniority-zero, 
            by default True.

        Returns
        -------
        self : OrbitalEntanglement 
            OrbitalEntanglement instance.
        """        
        rdm1 = method.make_rdm1()
        rdm2 = method.make_rdm2()

        if len(np.array(rdm1))==2:
            rdm1s = (0.5*rdm1)*2
            rdm2s = (0.25*rdm2)*2
        else:
            rdm1s = rdm1
            rdm2s = rdm2

        if seniority_zero:
            odm1 = []
            for i in range(method.mol.nao):
                odm1_i = eval_odm1(i,rdm1s,rdm2s,seniority_zero=seniority_zero)
                odm1.append(odm1_i)
            odm2 = []
            for i in range(method.mol.nao):
                odm2_i = []
                for j in range(method.mol.nao):
                    odm2_ij = eval_odm2(i,j,rdm1s,rdm2s,rdm3s=None,rdm4s=None,seniority_zero=seniority_zero)
                    odm2_i.append(odm2_ij)
                odm2.append(odm2_i)
            
        else:
            raise NotImplementedError

        self = self(np.array(odm1), np.array(odm2))
        return self

    def one_orbital_entropy(self, odm1=None):
        r"""The entanglement entropy of one orbital defined as:

        .. math::
            s(1)_i = -\sum_{\alpha=1}^{4} \omega_{\alpha,i} \ln \omega_{\alpha,i}

        Parameters
        ----------
        odm1 : np.ndarray, optional
            The one orbital reduced density matrix, by default None.

        Returns
        -------
        s1 : List
            List of one orbital entropy.
        """        
        if odm1 is None:
            odm1 = self.odm1
        no = self.no
        s1 = np.zeros((no))
        for i in range(no):
            eigenvalues, _ = np.linalg.eig(odm1[i,:,:])
            s1[i] = -sum([sigma * np.log(sigma) for sigma in eigenvalues])
        return s1

    def two_orbital_entropy(self, odm2=None):
        r"""The entanglement entropy of two orbital defined as:

        .. math::
            s(2)_{ij} = -\sum_{\alpha=1}^{16} \omega_{\alpha,i,j} \ln \omega_{\alpha,i,j}

        Parameters
        ----------
        odm2 : np.ndarray, optional
            The two orbital reduced density matrix, by default None.
        """        
        if odm2 is None:
            odm2 = self.odm2
        no = self.no
        s2 = np.zeros((no,no))
        for i in range(no):
            for j in range(no):
                eigenvalues, _ = np.linalg.eig(odm2[i,j,:,:])
                s2[i,j] = -sum([sigma * np.log(sigma) for sigma in eigenvalues])
        return s2
    
    def mututal_information(self, s1=None, s2=None):
        r"""The orbital-pair mutual information defined as:

        .. math::
            I_{i|j} = \frac{1}{2} \left(s(2)_{i,j}-s(1)_{i}-s(1)_{j} \right)
                \left(1- \delta_{i,j} \right)
        """        
        if s1 is None:
            s1 = self.s1
        if s2 is None:
            s2 = self.s2
        no = self.no
        I = np.zeros((no,no))
        for i in range(no):
            for j in range(no):
                if i!=j:
                    I[i,j] = s2[i,j] - s1[i]- s1[j]
        return I