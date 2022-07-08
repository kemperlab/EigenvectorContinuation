#%%
"""
    Provides tools to showcase the creation and testing of Eigenvector Continuation.
    -----------------------------------------------------------------------------------------------
    INCLUDES:
        HilbertSpaceAbstract:   Abstract class used to outline how concrete Hilbert Spaces should
                                behave. Hilbert Spaces are used to create EigenvectorContinuer
                                objects. (abbr: HSA)

        NumpyArraySpace:        An example concrete implementation of HilbertSpaceAbstract in which
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
from matplotlib import pyplot as plt

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

    @abstractmethod
    def solve_gep(self, a_matrix, b_matrix):
        """ defines behavior to solve a generalized eignevalue problem of the form:
            Ax = rBx
            where A and B are linear transformations, x is an eigenvector, and r is an eigenvalue

            :param a_matrix:    the A matrix
            :param b_matrix:    the B matrix

            :returns:           the eigenvalues, eigenvectors calculated

            should be implemented by concrete class
        """

class NumpyArraySpace(HilbertSpaceAbstract):
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
        """ initializes an instance of a NumpyArraySpace and sets state variables

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
        """ defines the overlap matrix for a NumpyArraySpace

            if points are passed in, these become the new training points of the space
            otherwise, the existing training points are used

            For an overlap matrix S:
            S[i,j] = inner_product(basis_vec_i, basis_vec_j)

            :param points:  points to use as training points (optional)

            :returns:       the calculated overlap matrix
        """

        if points is not None:
            self._basis_vecs = points
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

        return evecs[0]

    def solve_gep(self, a_matrix, b_matrix):
        """ Uses scipy.linalg eigh to solve the generalized eigenvalue problem of the form:
            Ax = rBx
            where A and B are linear transformations, x is an eigenvector, and r is an eigenvalue

            :param a_matrix:    the A matrix
            :param b_matrix:    the B matrix
            :returns:           the eigenvalues, eigenvectors calculated
        """

        return eigh(a_matrix, b_matrix)

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

        TODO in this class:
        - talked to Kemper about design stuff
        - need to make HamInit inside HilbertSpace (b/c generalized vectors need init)      √ HamInit is now an inner class
        - overlap matrix, not interaction matrix                                            √ variables named correctly
        - pretty good picture of IO (treat like python library)                             √ understood
        - main() works, as long as vectors aren't lin dep (b_x must not = 0)                √ ok, don't make vectors linearly dependent
        - hard code pbc                                                                     √ pbc is no longer a required parameter
        - soon, do solve_gep                                                (2)             √ check with Kemper that it works. 
        - make hilbert space the argument for EC __init__                                   √ easy money. Use case: create hilbert_space with training points and num_qubits; create EC with that hilbert_space and some target points
        - plot just in the module, no need for object                       (3)
        - at some point, update UML to have all edits                       (1)             √
        - fix properties so that they have to go through the fancy "gets"   (4)             √
        - update comments                                                   (during ^)      √
        - make num_qubits not in abstract                                                   √
        - fix solve_gep

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
            :param input_target_point:     used to calculate the current hilbert space's
                                            subspace hamiltonian. If None is passed, will default
                                            to current_target_point in this EC
            :returns:                       the evals, evecs calculated
        """

        # calls the hilbert_space's method to solve the gep (which may differ depending on space)

        overlap = self.calc_overlap_matrix(input_training_points)
        subspace = self.calc_sub_ham(input_target_point)


        self._evals, self._evecs = self.hilbert_space.solve_gep(subspace, overlap)

        return self.evals, self.evecs
# vector not numpy




# Plotting tools

def plot_xxz_spectrum(bzmin, bzmax, evec_cont: EigenvectorContinuer):
    DATA_POINTS = 100
    """ determines fineness of curves """

    # initializes plot and axes
    fig, ax = plt.subplots()
    ax.set_xlabel("$B_Z$")
    ax.set_ylabel("Energy")

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
    bzlist = np.linspace(bzmin, bzmax, DATA_POINTS)

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
        ax.plot(bzlist, all_evals[:,idx], 'k-')
        
        # print(all_evals[:,idx])





    ax.axvline(1.0, ls = "--", color="blue")    # shows vertical line that represents [unsure] TODO <--



    
    # for b_z in range(bzlist):
    #     # make a new point of some kind with that b_z value
    #     # make a hilbert space (?) with that point (/points?) 
    #     # make an EC with that Hilbert Space
    #     # do gep
    #     # ask Kemper: should solve_gep be in concrete class?
        
    #     # result should be evals of some sort
    #     ax.plot(bzlist, evals, 'k-')
        
    #     # also plot each training point (either at lowest energy value or all points)
    #     # also plot each target point (either at lowest energy value or all points)
    plt.show()
    





def main():
    """ generates the image, hamiltonian, and overlap matrix """


# NEW
    # START Hamiltonian & Eigenvector Initialization

    # useful tuple when dealing with param_sets in this space
    ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")

    # TRAINING POINTS
    num_qubits = 2
    b_x = 0
    j_x = -1
    j_z = 0
    b_zs = np.array([0,2])  # put your custom input here

    # TARGET POINT
    target_b_x = 0
    target_j_x = -1
    target_j_z = 0
    target_b_z = 1.5  # put your custom input here
    target_param_set = ParamSet(target_j_x,target_j_z,target_b_x,target_b_z)

    param_sets = [None] * len(b_zs)
    for idx, b_z in enumerate(b_zs):
        param_sets[idx] = ParamSet(j_x,j_z,b_x,b_z)

    # the above set of parameters are used as training points to construct the space
    training_points = param_sets

    # CREATES THE HILBERT SPACE
    hilbert_space = NumpyArraySpace(training_points, num_qubits)

    eigenvector_continuer = EigenvectorContinuer(hilbert_space,target_param_set)

    evals,evecs = eigenvector_continuer.solve_gep()

    print(evals, "\n", evecs)

    plot_xxz_spectrum(0, 3, eigenvector_continuer)






if __name__ == "__main__":
    main()

# Random Notes
# shouldn't need n, evals,
# input: evecs, some ham of some osrt


# GARBAGIO BELOWIO
def ignore_this():

# class TrainingPointUtil:
#     DATA_POINTS = 100
#     """ determines fineness of curves """

#     ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
#     """" useful tuple when dealing with param sets in this space """

#     COMP_TOLERANCE = 1e-9
#     """ tolernace when comparing two floats """
    
#     def get_random_training_points(self, bzlist, evals, num_points):
#         """ returns random training points for the system

#             It only accounts for Bz, but later updates will be more robust by exploring more
#             degrees of freedom
#         """
#             # It only accounts for Bz, but later updates will be more robust by exploring more
#             # degrees of freedom

#         # initialize keys to keep track of randomness
#         keys = [0] * len(evals)

#         # named tuple used to organize points
#         Point = namedtuple("Point", "b_z energies")

#         points = [None] * num_points
#         for i in range(num_points):

#             keys[i] = random.randrange(0, len(bzlist) - 1)

#             point = Point(b_z=bzlist[keys[i]], energies=evals[keys[i]])

#             points[i] = point
#             # first index is Bz values; second index is energy values of different states 

#         return points

#     def compare_sets_of_points(self, set_a, set_b):
#         """ return true if any overlap in b_z values"""
#         # test for redundancy
#         for val_a in set_a:
#             for val_b in set_b:
#                 if math.isclose(val_a.b_z,val_b.b_z,rel_tol=self.COMP_TOLERANCE):
#                     return True
#         return False


# class HermitianSpace:
#     def get_overlap_matrix(self, evecs, num_points):
#         """ gets the overlap matrix for a given system """

#         # set up overlap matrix
#         s = np.zeros([num_points, num_points], dtype=complex)
#         # simple case: 2 qubits
#         #       this iterates over all evecs but don't I only want the ground state?
#         for i in range(num_points):
#             for j in range(num_points):
#                 vector1 = np.matrix(evecs[:,i])
#                 vector2 = np.matrix(evecs[:,j])

#                 vector2 = vector2.conj().T
#                 s[i,j] = vector1 @ vector2
#                 # print(s)

#         return s

#     def calc_subspace_ham(self, hams, evecs, num_points): #call evecs basis vector
#         """ calculates the hamiltonian for the subspace of the system """

#         new_hams = [None] * len(hams)
#         # print(num_points)

#         for idx_k, ham_k in enumerate(hams):
#             new_ham = np.zeros([num_points, num_points], dtype=complex)
#             for i in range(num_points):
#                 for j in range(num_points):
#                     vector1 = np.matrix(evecs[:,i])
#                     vector2 = np.matrix(evecs[:,j])

#                     vector2 = vector2.conj().T
#                     new_ham[i,j] = vector1 @ ham_k @ vector2
#             new_hams[idx_k] = new_ham
#             # print(new_ham)

#         return new_hams

#     def diagonalize_ham(self, ham):
#         """ returns the eigenvalues of the diagonalized hamoltonian 
            
#             this is its own function because while this is a very simple return statement here,
#             future abstractions of this funtion will be less simple, but still follow this template
#         """
        
#         return np.linalg.eigvalsh(ham) 

    # type checking code for NumpyArray
    # def check_type_generic(self, value):
    #     """ helper method to verify all data in this implementation is in an np.ndarray """
    #     if not isinstance(value, self.implementation_type):
    #         raise ValueError("data should be of type np.ndarray")

    # def check_basis_vecs_type(self, basis_vecs):
    #     """ checks the type of each basis vector (should be np.ndarray) """
    #     for basis_vec in basis_vecs:
    #         self.check_type_generic(basis_vec)

    # def check_ham_type(self, ham):
    #     """ checks the type of the hamiltonian (should be np.ndarray) """
    #     self.check_type_generic(ham)


# class temp(): GET EVECS FOR SET OF TRAINING POINTS CODE
    
#         # number of points used to construct the Hilbert Space
#         num_points = len(training_points)

#         # initialize hamiltonians
#         hamiltonian_initializer = HamiltonianInitializer()
#         hams = [None] * len(training_points)
#         for idx, training_points in enumerate(training_points):
#             hams[idx] = hamiltonian_initializer.xxztype_hamiltonian(training_points,num_qubits)

#         # calculate evecs for each ham
#         evec_sets = [None] * len(training_points)
#         for idx, ham in enumerate(hams):
#             evec_sets[idx] = hamiltonian_initializer.get_eigenpairs(ham)[1]

    pass

#%%
