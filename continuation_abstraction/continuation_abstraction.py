#%%

"""creates a plot and matrix that correspond to a simplified model of a hamiltonian
    found from eigenvector continuation.

    @author Jack Howard
    North Carolina State University -- Kemper Lab
"""

# import random
# import math
from collections import namedtuple
# import abc
from abc import ABC, abstractmethod
# from typing import Type
import numpy as np
from scipy.linalg import eigh
# from numpy import ndarray
# from matplotlib import pyplot as plt


class HilbertSpaceAbstract(ABC):
    """ defines behavior for objects to have a hamiltonian, inner product,
        and expectation value
    """

    @property 
    def training_points(self): # TODO make private
        """ I'm the current space's set of training points """
        return self._training_points

    @training_points.setter
    def training_points(self, value):
        """ Set the current space's set of training points """
        self._training_points = value

    @property
    def num_qubits(self):
        """ I'm the current space's number of qubits """
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, value):
        """ Set the current space's number of qubits """
        self._num_qubits = value

    @property
    def basis_vecs(self):
        """ I'm the current space's basis vectors """
        return self._basis_vecs

    @basis_vecs.setter
    def basis_vecs(self,value):
        """ Set the current space's basis vectors """
        self._basis_vecs = value

    def __init__(self, training_points, num_qubits):
        """ initializes an instance of a HermitianSpace and sets state variables

            :param points:      the sets of points to use to construct the Hilbert Space
            :param num_qubits:  the number of qubits in the space
        """

        self._training_points = training_points
        self._num_qubits = num_qubits
        self._basis_vecs = None
        self.calc_basis_vecs()

    @abstractmethod
    def calc_basis_vecs(self):
        """ calculates the basis vectors to span the space
            should be implemented by concrete class
        """

    @abstractmethod
    def inner_product(self, vec1, vec2):
        """ defines inner product for space
            should be implemented by concrete class
        """

    @abstractmethod
    def expectation_value(self, vec1, ham, vec2):
        """ defines expectation value calculation for space
            should be implemented by concrete class
        """

    @abstractmethod
    def get_overlap_matrix(self, points=None):
        """ defines the overlap matrix for space given some set of spanning vecs (basis_vecs)
            should be implemented by concrete class

            :param points:  points to use as training points (optional depending on implementation)
        """

    @abstractmethod
    def get_sub_ham(self, ham):
        """ defines a subspace hamiltonian for space given a hamiltonian in the space and
            a set of spanning vectors (basis_vecs)

            NB: ham cannot be constructed using the same points used to get basis_vecs

            should be implemented by concrete class
        """

    @abstractmethod
    def select_vec(self, evecs):
        """ defines which vector to select when chooosing from a set of evecs

            :param evecs:   the set of evecs

            should be implemented by concrete clas
        """

class NumpyArraySpace(HilbertSpaceAbstract):
    """ defines Hermitian Space behavior for numpy arrays

        contains inner class to help construct hamiltonian
    """

    @property
    def implementation_type(self):
        """ I'm the current space's implementation type """
        return np.ndarray

    def calc_basis_vecs(self):
        """ calculates the basis vectors for the given space
            creates a hamiltonian for each point, and determines eigenvecs for each hamiltonian
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
            current_evecs = hamiltonian_initializer.get_eigenpairs(ham)[1]
            evec_set[idx] = self.select_vec(current_evecs)

        self.basis_vecs = evec_set

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

    def get_overlap_matrix(self, points=None):
        """ defines the overlap matrix for a NumpyArraySpace

            if points are passed in, these become the new training points of the space
            otherwise, the existing training points are used

            For an overlap matrix S:
            S[i,j] = inner_product(basis_vec_i, basis_vec_j)

            :param points:  points to use as training points (optional)
        """

        if points is not None:
            self.basis_vecs = points
            self.calc_basis_vecs()

        # dimensions of square matrix will be numbner of basis vectors
        dim = len(self.basis_vecs)
        overlap_s = np.zeros([dim, dim], dtype=complex)

        # S[i,j] = inner_product(basis_vec_i, basis_vec_j)
        for idx_i, vec_i in enumerate(self.basis_vecs):
            for idx_j, vec_j in enumerate(self.basis_vecs):
                overlap_s[idx_i, idx_j] = self.inner_product(vec_i, vec_j)

        return overlap_s

    def get_sub_ham(self, ham):
        """ defines a subspace hamiltonian for space given a hamiltonian in the space and
            a set of spanning vectors (basis_vecs)

            NB: ham cannot be constructed using the same points used to get basis_vecs

            Subspace Ham[i,j] = expectation_value(basis_vec_i, ham, basis_vec_j)
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
        """ returns the lowest engergy evec """

        if len(evecs) == 0:
            pass

        return evecs[0]

    class HamiltonianInitializer:
        """ initializes the hamiltonian """

        PAULIS = {}
        """ defines dict of Paulis to use below """

        ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
        """" useful tuple when dealing with param sets in this space """

        def __init__(self):
            """ initializes class instance and Paulis dict """
            self.PAULIS['X'] = np.array([[0,1],[1,0]], dtype=complex)
            self.PAULIS['Y'] = np.array([[0,-1.j],[1.j,0]], dtype=complex)
            self.PAULIS['Z'] = np.array([[1,0],[0,-1]], dtype=complex)
            self.PAULIS['I'] = np.array([[1,0], [0,1]], dtype=complex)

        def many_kron(self, ops):
            """ produces Kronecker (Tensor) product from list of Pauli charaters """
            result = self.PAULIS[ops[0]]    # set result equal to first pauli given by the param
            if len(ops) == 1:
                return result

            for opj in ops[1:]:             # for all the operations in the parameter
                result = np.kron(result, self.PAULIS[opj])  # tensor product the current matrix with
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

        def get_eigenpairs(self, ham):
            """ gets the eigenpairs for a given param_setinate in a system"""
            evals, evecs = np.linalg.eigh(ham)

            return evals, evecs

class EigenvectorContinuer():
    """ Houses the functionality to create a Hilbert Space of specified type and perform
        Eigenvector Continuation for a given set of training points and target points

        USE CASE:
        1.  Specify a type for your space; must inherit from HilbertSpaceAbstract
        2.  Choose:
                number of qubits
                b_z points to use           # TODO make this more generalized
                periodic boundary condition as either True or False
                target points to use

        OUTPUT:
            Eigenvalues and Eigenvectors from the Generalized Eigenvalue Problem

        TODO in this class:
        - talked to Kemper about design stuff
        - need to make HamInit inside HilbertSpace (b/c generalized vectors need init)      √ HamInit is now an inner class
        - overlap matrix, not interaction matrix                                            √ variables named correctly
        - pretty good picture of IO (treat like python library)                             √ understood
        - main() works, as long as vectors aren't lin dep (b_x must not = 0)                √ ok, don't make vectors linearly dependent
        - hard code pbc                                                                     √ pbc is no longer a required parameter
        - soon, do solve_gep                                                (2)
        - make hilbert space the argument for EC __init__                                   √ easy money. Use case: create hilbert_space with training points and num_qubits; create EC with that hilbert_space and some target points
        - plot just in the module, no need for object                       (3)
        - at some point, update UML to have all edits                       (1)             √
        - fix properties so that they have to go through the fancy "gets"   (4)
        - update comments                                                   (during ^)

    """

    # @property
    # def training_points(self):
    #     return self._training_points

    @property
    def hilbert_space(self):
        return self._hilbert_space

    @property
    def overlap_matrix(self):
        return self._overlap_matrix

    @property
    def sub_ham(self):
        return self._sub_ham

    @property
    def current_target_points(self):
        return self._current_target_points



    # @property
    # def num_qubits(self):
    #     return self._num_qubits

    # @overlap_matrix.setter
    # def overlap_matrix(self, value):
    #     self._overlap_matrix = value

    # @hilbert_space.setter
    # def hilbert_space(self, value):
    #     self._hilbert_space = value

    # @sub_ham.setter
    # def sub_ham(self, value):
    #     self._sub_ham = value

    # @num_qubits.setter
    # def num_qubits(self, value):
    #     self._num_qubits = value


    def __init__(self, hilbert_space, target_points):

        # Validate type of hilbert space
        if not isinstance(hilbert_space, HilbertSpaceAbstract):
            raise ValueError("concrete_type must be a subclass of HilbertSpaceAbstract")

        # Setting properties
        self._hilbert_space = hilbert_space
        self._overlap_matrix = self.hilbert_space.get_overlap_matrix()
        self._current_target_points = target_points
        self.refresh_sub_ham()

    def get_overlap_matrix(self, input_training_points=None):
        """  """

        if input_training_points is not None:
            self.hilbert_space.training_points = input_training_points
            self.refresh_overlap_matrix()

        return self.overlap_matrix

    def get_sub_ham(self, input_target_points=None):

        if input_target_points is not None:
            self.current_target_points = input_target_points
            self.refresh_sub_ham()

        return self.sub_ham

    def refresh_overlap_matrix(self):
        self._overlap_matrix = self.hilbert_space.get_overlap_matrix()

    def refresh_sub_ham(self):
        ham_init = self.hilbert_space.HamiltonianInitializer()
        target_ham = ham_init.xxztype_hamiltonian(self.current_target_points, self.hilbert_space.num_qubits)
        self._sub_ham = self.hilbert_space.get_sub_ham(target_ham)


def main():
    """ generates the image, hamiltonian, and overlap matrix """

# START Hamiltonian & Eigenvector Initialization
    # useful tuple when dealing with param_sets in this space
    ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")

# USER INPUT
    # data for Hilbert Space
    num_qubits = 2
    b_x = .2
    j_x = 1
    j_z = 1
    b_zs = np.array([0,2,3])  # put your custom input here

    # data for target hamiltonian
    target_b_x = .3
    target_j_x = 2
    target_j_z = 1
    target_b_z = 2  # put your custom input here
    target_param_set = ParamSet(target_j_x,target_j_z,target_b_x,target_b_z)


# HILBERT SPACE SETUP

    # setting up hilbert space training points
    # create a param_set for each b_z value
    param_sets = [None] * len(b_zs)
    for idx, b_z in enumerate(b_zs):
        param_sets[idx] = ParamSet(j_x,j_z,b_x,b_z)
    # the above set of parameters are used as training points to construct the space
    training_points = param_sets

# HILBERT SPACE CREATION & USE
    # create new space of a type that implements HilbertSpaceAbstract (chosen by user)
    hilbert_space = NumpyArraySpace(training_points, num_qubits)

    # construct a target hamiltonian for the space to operate on
    init = hilbert_space.HamiltonianInitializer()
    input_ham = init.xxztype_hamiltonian(target_param_set, num_qubits)

    if not isinstance(hilbert_space, HilbertSpaceAbstract):
        print("hilbert_space has incorrect type")

    # initialize the basis vectors (eigenvectors in this case) on the subspace
    hilbert_space.calc_basis_vecs()

    # get the overlap matrix of the space
    overlap_s = hilbert_space.get_overlap_matrix()

    print(overlap_s)

    # calculate the subspace hamitonian for the given target hamiltonian
    target_ham = hilbert_space.get_sub_ham(input_ham)

    print(target_ham)

# GENERALIZED EIGNEVALUE PROBLEM
    # Form of GEP: target_ham @ evec = eval @ overlap_s @ evec
    # use above overlap matrix (overlap_s) and subspace hamiltonian (target_ham) to do GEP
    evals, evecs = eigh(target_ham, overlap_s) # uses scipy.linalg.eigh

    # FINISHED Tuesday  almost all of main method, and some flexibility stuff with the training pts
    #                   UML Diagram (on iPad) (still have some design questions)
    #
    # NEXT Wednesday    Ask Kemper about the weird eigh error (are my training points not good?)
    #                   Ask Kemper about the Questions on the UML Digram (design stuff mostly)


    print(evals, "\n", evecs, "\n\n")
    



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

# class SpectrumPlotter:
#     DATA_POINTS = 100
#     """ determines fineness of curves """

#     ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
#     """" useful tuple when dealing with param sets in this space """

#     COMP_TOLERANCE = 1e-9
#     """ tolernace when comparing two floats """
    
#     def plot_xxz_spectrum(self, bzlist, evals, points, phats, n):
#         """ plots the spectrum along with training points and phats """

#         ax = plt.subplots()[1]    # initializes axes
#         ax.set_xlabel("$B_Z$")
#         ax.set_ylabel("Energy")

#         for idx in range(2**n):                       # prepares plot
#             ax.plot(bzlist, evals[:,idx], 'k-')

#         ax.axvline(1.0, ls = "--", color="blue")    # shows vertical line that represents [unsure] TODO <--

#         # plot training points. i[1][0] corresponds to the lowest energy 
#         for point in points:
#             plt.plot(point.b_z, point.energies[0], marker="o", color="blue")

#         # plot phats
#         for phat in phats:
#             plt.plot(phat.b_z, phat.energies[0], marker="o", color="orange")
#             plt.plot(phat.b_z, phat.energies[1], marker="o", color="orange")
     

#         plt.show()

#     def generate_xxz_type_spectrum(self, param_set, n=2):
#         """ calculates the different hamiltonians, eigenvalues, and overlap matrix for the system
#             and plots the spectrum on a plot"""

#         # j_x = param_set.j_x
#         # j_z = param_set.j_z
#         # b_x = param_set.b_x
#         # bzmin = param_set.bzmin
#         # bzmax = param_set.bzmax

#         # Initialize:
#         # plotting tool: array of evenly spaced numbers between bzmin and bzmax
#         bzlist = np.linspace(param_set.bzmin, param_set.bzmax, self.DATA_POINTS)
#         evals, evecs = self.get_eigenpairs(param_set, bzlist, n)

#         # Training Points:
#         # getting n random training points
#         num_points = n
#         points = self.get_random_training_points(bzlist=bzlist, evals=evals, num_points=num_points)

#         # Phats:
#         # set up new points to diagonalize from: denoted phat
#         # num_phats = n # Instead, make all point selection done in main
#         while True:
#             phats = self.get_random_training_points(bzlist=bzlist, evals=evals, num_points=n)

#             if not self.compare_sets_of_points(points, phats):
#                 break
#         # sort phats in ascending order
#         phats.sort()

#         # Calculate Hams:
#         # list of hamiltonians, one for each phat_k in phats
#         hams = [None] * len(phats)

#         # find the hamiltonian for each phat_k
#         for idx_k, phat_k in enumerate(phats):
#             b_zparam_set = self.ParamSet(j_x=param_set.j_x, j_z=param_set.j_z, b_x=param_set.b_x, b_z=phat_k.b_z)
#             hams[idx_k] = self.xxztype_hamiltonian(param_set=b_zparam_set, n=n)

#         # overlap Matrix:
#         # get the S overlap matrix
#         s = self.get_overlap_matrix(evecs=evecs, num_points=num_points)

#         # Subspace Hams:
#         # get the subspace hamiltonian for each phat_k value
#         new_hams = self.calc_subspace_ham(hams=hams, evecs=evecs, num_points=num_points)
#         # print(new_hams)

#         # Diagonalization:
#         energy_lists = [None] * len(hams) # use np.array
#         # get eigenvals corresponding to each energy level for every hamiltonian
#         for idx_k, new_ham_k in enumerate(new_hams):
#             energy_lists[idx_k] = self.diagonalize_ham(new_ham_k)
#             print("Iteraction",idx_k)
#             print("New ham:")
#             print(new_ham_k)
#             print("Diagonalized:")
#             print(energy_lists[idx_k]) 
#             print()

#             # Check:
#             # checks to see if Inverse_overlap • New_Ham = Eigenvals
#             print("S_inv • New ham (should correspond to Diagonalized value")
#             print(np.linalg.inv(s) @ new_ham_k)

#         self.plot_xxz_spectrum(bzlist, evals, points, phats, n)

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
