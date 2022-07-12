#%%
"""
    Provides tools to showcase the creation and testing of Eigenvector Continuation.
    -----------------------------------------------------------------------------------------------
    INCLUDES:
        HilbertSpaceAbstract:   Abstract class used to outline how concrete Hilbert Spaces should
                                behave. Hilbert Spaces are used to create EigenvectorContinuer
                                objects. (abbr: HSA)

        NumPyVectorSpace:       An example concrete implementation of HilbertSpaceAbstract in which
                                data is represented in the form of type np.ndarray

        EigenvectorContinuer:   A class used to take in any type of HSA and perform eigenvector
                                continuation operations using the HSA and some representation of
                                a target point. (abbr: EC)

        sample code used to showcase the EigenvectorContinuation process

    -----------------------------------------------------------------------------------------------
    SOON TO INCLUDE:
        plotting tools:         currently in development. Multiple target points are used to
                                produce a plot. Details may vary depending on implementation
                                specifics
    -----------------------------------------------------------------------------------------------
"""

# import random
# import math
from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eigh
from scipy.linalg import null_space
from matplotlib import pyplot as plt

# TODO copyright
__author__ = "Jack Howard"
__copyright__ = "TODO 2022, Kemper Lab -- North Carolina State University"
__credits__ = "Jack Howard, Akhil Francis, Lex Kemper"

class HilbertSpaceAbstract(ABC):
    """ defines behavior for objects to have a hamiltonian, inner product,
        and expectation value
    """

    @property
    def training_points(self):
        """ I'm the current space's set of training points """
        return self._training_points

    @property
    def basis_vecs(self):
        """ I'm the current space's basis vectors """
        return self._basis_vecs

    @training_points.setter
    def training_points(self, value):
        """ defines behavior of getting training_points """

        self._training_points = value

    def __init__(self, training_points):
        """ initializes an instance of a Hilbert Space and sets state variables

            :param training_points: the sets of points to use to construct the Hilbert Space
        """

        self._training_points = training_points

        self._basis_vecs = None
        self.calc_basis_vecs()

    @abstractmethod
    def calc_basis_vecs(self):
        """ calculates the basis vectors to span the space

            :returns:   the calculated basis vecs

            should be implemented by concrete class
        """

    @abstractmethod
    def inner_product(self, vec1, vec2):
        """ defines inner product for space

            :param vec1:    the first vector in the calculation
            :param vec2:    the second vector in the calculation

            :returns:       the result of the inner product

            should be implemented by concrete class
        """

    @abstractmethod
    def expectation_value(self, vec1, ham, vec2):
        """ defines expectation value calculation for space

            :param vec1:    the first vector in the calculation
            :param ham:     the hamiltonian in the calculation
            :param vec2:    the second vector in the calculation

            :returns:       the expectation value

            should be implemented by concrete class
        """

    @abstractmethod
    def calc_overlap_matrix(self, points=None):
        """ defines the overlap matrix for space given some set of spanning vecs (basis_vecs)

            :param points:  points to use as training points (optional depending on implementation)

            :returns:       the overlap matrix

            should be implemented by concrete class
        """

    @abstractmethod
    def calc_sub_ham(self, ham):
        """ defines a subspace hamiltonian for space given a hamiltonian in the space and
            a set of spanning vectors (basis_vecs)

            NB: ham cannot be constructed using the same points used to calc basis_vecs

            :param ham:     the hamiltonian used to find the subspace of

            :returns:       the subspace hamiltonian

            should be implemented by concrete class
        """

    @abstractmethod
    def select_vec(self, evecs):
        """ defines which vector to select when chooosing from a set of evecs

            :param evecs:   the set of evecs

            :returns:       the selected vector

            should be implemented by concrete class
        """

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

        ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
        """" useful tuple when dealing with param sets in this space """

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

class UnitarySpace(HilbertSpaceAbstract):
    """ defines Hilbert Space behavior for unitary matrices stored as numpy arrays

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

    @property
    def unitaries(self):
        """ I'm this space's list of unitaries. Each unitary is calculated from a basis vector
            (all basis vectors are calculated from the input training points)
        """
        return self._unitaries

    @property
    def zero_bra(self):
        """ I'm this space's zero_bra, to be used in many calculations. zero_bra is often
            denoted as <0| and is defined as the row vector: (1 0 0 ... 0)  with length
            2**num_qubits
        """
        return self._zero_bra

    @property
    def zero_ket(self):
        """ I'm this space's zero_ket, to be used in many calculations. zero_ket is often
            denoted as |0> and is defined as the column vector: (1 0 0 ... 0).T  with length
            2**num_qubits
        """
        return self._zero_ket

    def __init__(self, training_points, num_qubits):
        """ initializes an instance of a UnitarySpace and sets state variables

            :param points:      the sets of points to use to construct the Hilbert Space
            :param num_qubits:  the number of qubits in the space
        """
        self._num_qubits = num_qubits

        super().__init__(training_points)

        # construct |0> for calculations
        zero_ket = np.zeros([2**self._num_qubits], dtype="complex")
        zero_ket[0] = 1.0
        zero_ket = np.reshape(zero_ket, [2**self._num_qubits, 1])
        self._zero_ket = zero_ket

        self._zero_bra = np.reshape(self.zero_ket, [1, 2**self.num_qubits])

        self._unitaries = self.calc_unitaries()

    def calc_unitaries(self):
        """ calculates the unitary for each vector in this instance's basis_vecs

            :param vecs:    the vectors from which to calculate the unitaries
            :returns:       the unitary matrix, one for each vector.
        """

        unitaries = [None] * len(self._basis_vecs)
        for idx, vec in enumerate(self._basis_vecs):
            unitaries[idx] = self.calc_unitary(vec)

        return unitaries

    def calc_unitary(self, input_vec):
        """ calculates the unitary for a given vector

            :param input_vec: the vector from which to calculate the unitary
            :returns:   the unitary matrix for the vector
        """
        length = len(input_vec)
        assert length == 2**self.num_qubits

        # get from class property
        zero_ket = self._zero_ket

        # construct an orthonormal basis for the unitary
        input_bra = np.reshape(input_vec, (1, length))
        input_as_matrix = zero_ket @ input_bra
        orthonormal_basis = null_space(input_as_matrix)

        # construct the unitary
        unitary = np.zeros([length, length], dtype="complex")
        unitary[:,0] = input_vec
        for idx in range(length - 1):
            unitary[:, idx + 1] = orthonormal_basis[:, idx]

        return unitary

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
        """ defines inner product for numpy array space. The vectors used as input are
            assumed to be unitary matrices for this class

            :param vec1:    the left vector of the inner product
            :param vec2:    the right vector of the inner product

            :returns:       inner product of vec1 & vec2
        """

        # renaming of variables to match this class's use case
        uni1 = vec1
        uni2 = vec2

        # Raises error if argument argument types are not np.ndarray (np.matrix is allowed)
        if (not isinstance(uni1, np.ndarray) or not isinstance(uni2, np.ndarray)):
            raise ValueError("both vec1 and vec2 should be of type np.ndarray")

        return self._zero_bra @ uni1.conj().T @ uni2 @ self._zero_ket

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

        # renaming of variables to match this class's use case
        uni1 = vec1
        uni2 = vec2

        return self.zero_bra @ uni1.conj().T @ ham @ uni2 @ self.zero_ket

    def calc_overlap_matrix(self, points=None):
        """ defines the overlap matrix for a UnitarySpace

            if points are passed in, these become the new training points of the space
            otherwise, the existing training points are used

            For an overlap matrix S:
            S[i,j] = inner_product(unitary_i, unitary_j)

            :param points:  points to use as training points (optional)

            :returns:       the calculated overlap matrix
        """

        if points is not None:
            self._training_points = points
            self.calc_basis_vecs()
            self.calc_unitaries()

        # dimensions of square matrix will be number of unitaries
        dim = len(self.unitaries)
        overlap_s = np.zeros([dim, dim], dtype=complex)

        # S[i,j] = inner_product(unitary_i, unitary_j)
        for idx_i, vec_i in enumerate(self.unitaries):
            for idx_j, vec_j in enumerate(self.unitaries):
                overlap_s[idx_i, idx_j] = self.inner_product(vec_i, vec_j)

        return overlap_s

    def calc_sub_ham(self, ham):
        """ defines a subspace hamiltonian for space given a hamiltonian in the space and
            a set of spanning vectors (basis_vecs)

            NB: ham cannot be constructed using the same points used to calc basis_vecs

            Subspace Ham[i,j] = expectation_value(unitary_i, ham, unitary_j)

            :param ham:     the hamiltonian used to find the subspace of

            :returns:       the subspace hamiltonian
        """

        # dimensions of square matrix will be number of basis vectors
        dim = len(self.unitaries)
        sub_ham = np.zeros([dim, dim], dtype=complex)

        # SubspaceHam[i,j] = expectation_value(unitary_i, ham, unitary_j)
        for idx_i, vec_i in enumerate(self.unitaries):
            for idx_j, vec_j in enumerate(self.unitaries):
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

        ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
        """" useful tuple when dealing with param sets in this space """

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

class EigenvectorContinuer():
    """ Houses the functionality to create a Hilbert Space of specified type and perform
        Eigenvector Continuation for a given set of training points and target points

        USE CASE:
        1.  Create an instance of a HilbertSpaceAbstract concrete class/subclass
                - Will need training points (and/or other input depending on implementation)
        2.  Input:
                a target point to use and calculate the subspace hamiltonian

        OUTPUT:
            Eigenvalues and Eigenvectors from the Generalized Eigenvalue Problem constructed from
            the subspace hamiltonian and overlap matrix calculated by the training and target
            points

        USEFUL METHODS (documentation given below):
            __init__(...)
            calc_overlap_matrix(...)
            calc_sub_ham(...)
            solve_gep(...)

    """

    @property
    def hilbert_space(self):
        """ I'm this EC's hilbert space """
        return self._hilbert_space

    @property
    def overlap_matrix(self):
        """ I'm this EC's last calculated overlap matrix """
        return self._overlap_matrix

    @property
    def sub_ham(self):
        """ I'm this EC's last calculated subspace hamiltonian """
        return self._sub_ham

    @property
    def current_target_point(self):
        """ I'm this EC's current target point. I'm used to create the sub_ham"""
        return self._current_target_point

    @property
    def evals(self):
        """ I'm this EC's last calculated set of eigenvalues for the diagonalized subspace ham """
        return self._evals

    @property
    def evecs(self):
        """ I'm this EC's last calculated set of eigenvectors for the diagonalized subspace ham """
        return self._evecs

    def __init__(self, hilbert_space, target_point):
        """ initializes the EC

            :param hilbert_space:   the Hilbert Space used for Eigenvector Continuation in
                                    conjunction with a target point
            :param target_point:    the point in the Hilbert Space to compare the existing
                                    Hilbert Space. Used to create a subspace hamiltonian to
                                    then solve the GEP and obtain the final eigenvectors and
                                    eigenvalues
        """

        # Validate type of hilbert space
        if not isinstance(hilbert_space, HilbertSpaceAbstract):
            raise ValueError("concrete_type must be a subclass of HilbertSpaceAbstract")

        # Setting properties
        self._hilbert_space = hilbert_space
        self._overlap_matrix = self.hilbert_space.calc_overlap_matrix()
        self._current_target_point = target_point
        self.refresh_sub_ham()
        self._evals = None
        self._evecs = None

    def calc_overlap_matrix(self, input_training_points=None):
        """ caclulates the overlap matrix based on the Hilbert Space's current training points

            :param input_training_points:   [OPTIONAL] can be used to update the hilbert
                                            space's training points as needed

            :returns:                       the overlap matrix
        """

        if input_training_points is not None:
            self.hilbert_space.training_points = input_training_points
            self.refresh_overlap_matrix()

        return self.overlap_matrix

    def calc_sub_ham(self, input_target_point=None):
        """ calculates the subspace hamiltonian based on the EC's current Hilbert Space and
            target point

            :param input_target_point:  [OPTIONAL] can be used to update the current target
                                        point of the EC as needed

            :returns:                   the subspace hamiltonian
        """
        if input_target_point is not None:
            self.current_target_point = input_target_point
            self.refresh_sub_ham()

        return self.sub_ham

    def refresh_overlap_matrix(self):
        """ refreshes the current overlap matrix property based on current properties """
        self._overlap_matrix = self.hilbert_space.calc_overlap_matrix()

    def refresh_sub_ham(self):
        """ rereshed the current subspace hamiltonian based on current properties """
        ham_init = self.hilbert_space.HamiltonianInitializer()
        target_ham = ham_init.xxztype_hamiltonian(self.current_target_point,
                                                  self.hilbert_space.num_qubits)
        self._sub_ham = self.hilbert_space.calc_sub_ham(target_ham)

    def solve_gep(self, input_training_points=None, input_target_point=None):
        """ solves the generalized eigencvalue problem for this EC

            :param input_training_points:   used to calculate the current hilbert space's
                                            overlap matrix. If None is passed, will default
                                            to current training_points in the hilbert space
            :param input_target_point:      used to calculate the current hilbert space's
                                            subspace hamiltonian. If None is passed, will default
                                            to current_target_point in this EC
            :returns:                       the evals, evecs calculated
        """

        overlap = self.calc_overlap_matrix(input_training_points)
        subspace = self.calc_sub_ham(input_target_point)


        self._evals, self._evecs = eigh(subspace, overlap)

        return self.evals, self.evecs


# Plotting tools

def plot_xxz_spectrum(bzmin, bzmax, evec_cont: EigenvectorContinuer):
    """ plots the spectrum of eigenvalues for a given EC

        :param bzmin:       the minimum b_z value to plot
        :param bzmax:       the maximum b_z value to plot
        :param evec_cont:   the EC to plot (plots training and target points, and expected energies)

    """

    # determines fine-ness of curves
    data_points = 100


    # initializes plot and axes
    axes = plt.subplots()[1]
    axes.set_xlabel("$B_Z$")
    axes.set_ylabel("Energy")

    # PLOT POINTS FROM INPUT EC
    # sets up hamiltonian initializer to reduce overhead in for loop
    ham_init = evec_cont.hilbert_space.HamiltonianInitializer()

    # "for every training point in the EC, ..."
    for training_point in evec_cont.hilbert_space.training_points:
        # "... calculate its hamiltonian, ..."
        ham = ham_init.xxztype_hamiltonian(param_set=training_point,
                                           n_qubits=evec_cont.hilbert_space.num_qubits)
        # "... get the eigenvalues of that hamiltonian, ..."
        evals = ham_init.calc_eigenpairs(ham)[0]

        # " ... and plot each eigenvalue"
        for current_eval in evals:
            plt.plot(training_point.b_z, current_eval, marker="o", color="blue")


    # gets the evals of the ec to reduce overhead of the for loop
    ec_evals = evec_cont.evals

    # plot each target point
    for ec_eval in ec_evals:
        plt.plot(evec_cont.current_target_point.b_z, ec_eval, marker="o", color="red")

    # PLOT EXPECTED ENERGY CURVES
    # get parameters for expected curves
    j_x = evec_cont.hilbert_space.training_points[0].j_x
    j_z = evec_cont.hilbert_space.training_points[0].j_z
    b_x = evec_cont.hilbert_space.training_points[0].b_x

    # get list of spaced out points
    bzlist = np.linspace(bzmin, bzmax, data_points)

    # plot the lines
    all_evals = np.zeros([len(bzlist), 2**evec_cont.hilbert_space.num_qubits])
    for idx, b_z in enumerate(bzlist):
        param_set = ham_init.ParamSet(j_x, j_z, b_x, b_z)

        ham = ham_init.xxztype_hamiltonian(param_set=param_set,
                                           n_qubits=evec_cont.hilbert_space.num_qubits)
        # if idx == 50:
        # print(ham_init.calc_eigenpairs(ham)[0])
        # print(idx)
        all_evals[idx,:] = ham_init.calc_eigenpairs(ham)[0]

    for idx in range(2**evec_cont.hilbert_space.num_qubits):
        axes.plot(bzlist, all_evals[:,idx], 'k-')
        # print(all_evals[:,idx])

    axes.axvline(1.0, ls = "--", color="blue")  # shows vertical line that represents crossing point

    plt.show()

def main():
    """ generates the plot for user-input values """

    # START Hamiltonian & Eigenvector Initialization

    # useful tuple when dealing with param_sets in this space
    ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")

    # Print conditioning number
    # TRAINING POINTS
    num_qubits = 3
    b_x = 0.1
    j_x = -1
    j_z = 0
    b_zs = np.array([-.1,0.1, 3])  # put your custom input here

    # TARGET POINT
    target_b_x = b_x
    target_j_x = j_x
    target_j_z = j_z

    target_b_z = 2  # put your custom input here
    target_param_set = ParamSet(target_j_x,target_j_z,target_b_x,target_b_z)

    param_sets = [None] * len(b_zs)
    for idx, b_z in enumerate(b_zs):
        param_sets[idx] = ParamSet(j_x,j_z,b_x,b_z)

    # the above set of parameters are used as training points to construct the space
    training_points = param_sets

    # CREATES THE HILBERT SPACE
    hilbert_space = UnitarySpace(training_points, num_qubits)

    eigenvector_continuer = EigenvectorContinuer(hilbert_space,target_param_set)

    evals,evecs = eigenvector_continuer.solve_gep()

    condition_number = np.linalg.cond(eigenvector_continuer.overlap_matrix)
    print("evals:\n", evals, "\nevecs:\n", evecs, "\ncondition number:\n", condition_number)

    plot_xxz_spectrum(-1, 3, eigenvector_continuer)

if __name__ == "__main__":
    main()

#%%
