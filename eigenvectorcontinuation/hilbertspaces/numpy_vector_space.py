"""
    NumPyVectorSpace:       An example concrete implementation of HilbertSpaceAbstract in which
                            data is represented in the form of type np.ndarray
"""


# General Imports
import numpy as np

# Local Imports
from eigenvectorcontinuation.hilbertspaces.hilbert_space_abstract import HilbertSpaceAbstract

__author__ = "Jack H. Howard, Akhil Francis, Alexander F. Kemper"
__citation__ = "" # TODO Arxiv or doi
__copyright__ = "Copyright (c) 2022 Kemper Lab"
__credits__ = ["Jack H. Howard", "Akhil Francis", "Alexander F. Kemper",
               "Anjali A. Agrawal", "Efekan Kökcü"]
__license__ = "BSD-2-Clause-Patent"
__version__ = "0.1"
__maintainer__ = "Jack H. Howard"
__email__ = "jhhoward@ncsu.edu"
__status__ = "Development"


class NumPyVectorSpace(HilbertSpaceAbstract):
    """ defines Hilbert Space behavior for numpy arrays

        contains inner class to help construct hamiltonian
    """

    @property
    def implementation_type(self):
        """ I'm the current space's implementation type """
        return np.ndarray

    @property
    def num_qubits(self):
        """ I'm the current space's number of qubits """
        return self._num_qubits

    def __init__(self, training_points, num_qubits):
        """ initializes an instance of a NumPyVectorSpace and sets state variables

            :param points:      the sets of points to use to construct the Hilbert Space
            :param num_qubits:  the number of qubits in the space
        """

        self._num_qubits = num_qubits

        super().__init__(training_points)

    def calc_basis_vecs(self):
        """ calculates the basis vectors for the given space
            creates a hamiltonian for each point, and determines eigenvecs for each hamiltonian

            :returns:   the calculated basis vecs
        """

        # number of points used to construct the Hilbert Space
        num_points = len(self.training_points)

        # initialize hamiltonians
        hamiltonian_initializer = self.HamiltonianInitializer()
        hams = [None] * num_points
        for idx, training_points in enumerate(self.training_points):
            hams[idx] = hamiltonian_initializer.xxztype_hamiltonian(training_points,
                                                                    self.num_qubits)

        # calculate evecs for each ham; selects lowest energy evec to go in evec_set
        evec_set = [None] * num_points
        for idx, ham in enumerate(hams):
            current_evecs = hamiltonian_initializer.calc_eigenpairs(ham)[1]
            evec_set[idx] = self.select_vec(current_evecs)

        self._basis_vecs = evec_set

    def inner_product(self, vec1, vec2):
        """ defines inner product for numpy array space

            :param vec1:    the left vector of the inner product
            :param vec2:    the right vector of the inner product

            :returns:       inner product of vec1 & vec2
        """

        # Raises error if argument argument types are not np.ndarray (np.matrix is allowed)
        if (not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray)):
            raise ValueError("both vec1 and vec2 should be of type np.ndarray")

        return vec1.conj() @ vec2

    def expectation_value(self, vec1, ham, vec2):
        """ defines expectation value calculation for numpy array space

            :param vec1:    the left vector of the expectation value calculation
            :param ham:     retrieve the expectation value w.r.t. this hamiltonian
            :param vec2:    the right vector of the expectation value calculation

            :returns:       the expectation value of the system
        """

        # Raises error if argument types are not np.ndarray (np.matrix is allowed)
        if (not isinstance(vec1, np.ndarray) or
            not isinstance(ham, np.ndarray) or
            not isinstance(vec2, np.ndarray)):
            raise ValueError("both vec1 and vec2 should be of type np.ndarray")

        return vec1.conj() @ ham @ vec2

    def calc_overlap_matrix(self, points=None):
        """ defines the overlap matrix for a NumPyVectorSpace

            if points are passed in, these become the new training points of the space
            otherwise, the existing training points are used

            For an overlap matrix S:
            S[i,j] = inner_product(basis_vec_i, basis_vec_j)

            :param points:  points to use as training points (optional)

            :returns:       the calculated overlap matrix
        """

        if points is not None:
            self._training_points = points
            self.calc_basis_vecs()

        # dimensions of square matrix will be number of basis vectors
        dim = len(self.basis_vecs)
        overlap_s = np.zeros([dim, dim], dtype=complex)

        # S[i,j] = inner_product(basis_vec_i, basis_vec_j)
        for idx_i, vec_i in enumerate(self.basis_vecs):
            for idx_j, vec_j in enumerate(self.basis_vecs):
                overlap_s[idx_i, idx_j] = self.inner_product(vec_i, vec_j)

        return overlap_s

    def calc_sub_ham(self, ham):
        """ defines a subspace hamiltonian for space given a hamiltonian in the space and
            a set of spanning vectors (basis_vecs)

            NB: ham cannot be constructed using the same points used to calc basis_vecs

            Subspace Ham[i,j] = expectation_value(basis_vec_i, ham, basis_vec_j)

            :param ham:     the hamiltonian used to find the subspace of

            :returns:       the subspace hamiltonian
        """

        # dimensions of square matrix will be number of basis vectors
        dim = len(self.basis_vecs)
        sub_ham = np.zeros([dim, dim], dtype=complex)

        # SubspaceHam[i,j] = expectation_value(basis_vec_i, ham, basis_vec_j)
        for idx_i, vec_i in enumerate(self.basis_vecs):
            for idx_j, vec_j in enumerate(self.basis_vecs):
                sub_ham[idx_i, idx_j] = self.expectation_value(vec_i, ham, vec_j)

        return sub_ham

    def select_vec(self, evecs):
        """ returns the lowest energy evec

            :param evecs:   the set of evecs

            :returns:       the selected vector
        """

        if len(evecs) == 0:
            pass

        return evecs[:,0]

    class HamiltonianInitializer:
        """ initializes the hamiltonian """

        _PAULIS = {}
        """ defines dict of Paulis to use below """

        # ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
        # """" useful tuple when dealing with param sets in this space """

        def __init__(self):
            """ initializes class instance and Paulis dict """
            self._PAULIS['X'] = np.array([[0,1],[1,0]], dtype=complex)
            self._PAULIS['Y'] = np.array([[0,-1.j],[1.j,0]], dtype=complex)
            self._PAULIS['Z'] = np.array([[1,0],[0,-1]], dtype=complex)
            self._PAULIS['I'] = np.array([[1,0], [0,1]], dtype=complex)

        def many_kron(self, ops):
            """ produces Kronecker (Tensor) product from list of Pauli charaters

                :param ops: the operations [as characters] to apply to a matrix
            """

            result = self._PAULIS[ops[0]]    # set result equal to first pauli given by the param
            if len(ops) == 1:
                return result

            for opj in ops[1:]:             # for all the operations in the parameter
                result = np.kron(result, self._PAULIS[opj])  # tensor product the current matrix w/
                                                             # the next pauli in the parameter list
            return result

        def xxztype_hamiltonian(self, param_set, n_qubits, pbc=False):
            """ produces the hamiltonian for a system where j_x = j_y and b_x = b_y

                :param param_set:   the set of parameters: j_x, j_z, b_x, b_z
                :param n_qubits:    the number of quibits
                :param pbc:         periodic boundary condition wrap around logic boolean
                :returns:           hamiltonian of the system
            """

            j_x = param_set.j_x
            j_z = param_set.j_z
            b_x = param_set.b_x
            b_z = param_set.b_z

            ham = np.zeros([2**n_qubits, 2**n_qubits], dtype=complex) # initializes the hamiltonian

            # build hamiltonian matrix
            for isite in range(n_qubits):

                # Apply the Bz information to the hamiltonian matrix
                oplist = ['I']*n_qubits     # makes list of operators (default = identity matrix)
                oplist[isite] = 'Z'         # sets the isite-th entry to Z
                ham += b_z * self.many_kron(oplist)  # applies the operations specified to the ham

                # Apply the Bx information to the hamiltonian matrix
                oplist = ['I']*n_qubits     # makes list of operators (default = identity matrix)
                oplist[isite] = 'X'         # sets the isite-th entry to X
                ham += b_x * self.many_kron(oplist)  # applies the operations specified to the ham

                # checks whether to apply wrap-around rules
                jsite = (isite + 1) % n_qubits
                if (jsite != isite + 1 ) and not pbc:
                    continue                            # skips the XX, YY, ZZ

                # Apply the XX information to the hamiltonian
                oplist = ['I']*n_qubits     # makes list of operators (default = identity matrix)
                oplist[isite] = 'X'         # sets the isite-th entry to X
                oplist[jsite] = 'X'         # sets the jsite-th entry to X
                ham += j_x * self.many_kron(oplist)  # applies the operations specified to ham

                # Apply the YY information to the hamiltonian
                oplist = ['I']*n_qubits     # makes list of operators (default = identity matrix)
                oplist[isite] = 'Y'         # sets the isite-th entry to Y
                oplist[jsite] = 'Y'         # sets the jsite-th entry to Y
                ham += j_x * self.many_kron(oplist)  # applies the operations specified to ham

                # Apply the Z information to the hamiltonian
                oplist = ['I']*n_qubits     # makes list of operators (default = identity matrix)
                oplist[isite] = 'Z'         # sets the isite-th entry to Z
                oplist[jsite] = 'Z'         # sets the jsite-th entry to Z
                ham += j_z * self.many_kron(oplist)  # applies the operations specified to ham

            return ham

        def calc_eigenpairs(self, ham):
            """ calcs the eigenpairs for a given param_setinate in a system

                :param ham: the hamiltonian to get the eigenpairs from

                :returns:   the eigenpairs as: evals, evecs
            """

            evals, evecs = np.linalg.eigh(ham)

            return evals, evecs
