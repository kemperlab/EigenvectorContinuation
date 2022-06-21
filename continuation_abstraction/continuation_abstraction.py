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
from typing import Type
import numpy as np
# from numpy import ndarray
# from matplotlib import pyplot as plt


class HamiltonianInitializer:
    """ initializes the hamiltonian """

    PAULIS = {}
    """ defines dict of Paulis to use below """

    DATA_POINTS = 100
    """ determines fineness of curves """

    ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
    """" useful tuple when dealing with param sets in this space """

    COMP_TOLERANCE = 1e-9
    """ tolernace when comparing two floats """

    def __init__(self):
        """ initializes class instance and Paulis dict """
        self.PAULIS['X'] = np.array([[0,1],[1,0]], dtype=complex)
        self.PAULIS['Y'] = np.array([[0,-1.j],[1.j,0]], dtype=complex)
        self.PAULIS['Z'] = np.array([[1,0],[0,-1]], dtype=complex)
        self.PAULIS['I'] = np.array([[1,0], [0,1]], dtype=complex)

    def many_kron(self, ops):
        """ produces Kronecker (Tensor) product from list of Pauli charaters """
        result = self.PAULIS[ops[0]]    # set result equal to the first pauli given by the parameter
        if len(ops) == 1:
            return result

        for opj in ops[1:]:             # for all the operations in the parameter
            result = np.kron(result, self.PAULIS[opj])  # tensor product the current matrix with
                                                        # the next pauli in the parameter list
        return result

    def xxztype_hamiltonian(self, param_set, n_qubits, pbc):
        """ produces the hamiltonian for a system where j_x = j_y and b_x = b_y
            :param param_set:   the set of parameters: j_x, j_z, b_x, b_z
            :param n_qubits:    the number of quibits
            :param pbc:         whether or not to include periodic boundary condition wrap around logic
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
            oplist = ['I']*n_qubits         # makes list of operators (default = identity matrix)
            oplist[isite] = 'Z'             # sets the isite-th entry to Z
            ham += b_z * self.many_kron(oplist)  # applies the operations specified to the ham

            # Apply the Bx information to the hamiltonian matrix
            oplist = ['I']*n_qubits         # makes list of operators (default = identity matrix)
            oplist[isite] = 'X'             # sets the isite-th entry to X
            ham += b_x * self.many_kron(oplist)  # applies the operations specified to the ham

            # checks whether to apply wrap-around rules
            jsite = (isite + 1) % n_qubits
            if (jsite != isite + 1 ) and not pbc:
                continue                            # skips the XX, YY, ZZ

            # Apply the XX information to the hamiltonian
            oplist = ['I']*n_qubits         # makes list of operators (default = identity matrix)
            oplist[isite] = 'X'             # sets the isite-th entry to X
            oplist[jsite] = 'X'             # sets the jsite-th entry to X
            ham += j_x * self.many_kron(oplist)  # applies the operations specified to ham

            # Apply the YY information to the hamiltonian
            oplist = ['I']*n_qubits         # makes list of operators (default = identity matrix)
            oplist[isite] = 'Y'             # sets the isite-th entry to Y
            oplist[jsite] = 'Y'             # sets the jsite-th entry to Y
            ham += j_x * self.many_kron(oplist)  # applies the operations specified to ham

            # Apply the Z information to the hamiltonian
            oplist = ['I']*n_qubits         # makes list of operators (default = identity matrix)
            oplist[isite] = 'Z'             # sets the isite-th entry to Z
            oplist[jsite] = 'Z'             # sets the jsite-th entry to Z
            ham += j_z * self.many_kron(oplist)  # applies the operations specified to ham

        return ham

    def get_eigenpairs(self, ham):
        """ gets the eigenpairs for a given param_setinate in a system"""
        evals, evecs = np.linalg.eigh(ham)

        return evals, evecs

class HermitianSpaceInterface(ABC):
    """ defines behavior for objects to have a hamiltonian, inner product,
        and expectation value
    """

    def __init__(self, basis_vecs, ham):
        """ initializes an instance of a HermitianSpace and sets state variables """

        self.basis_vecs = basis_vecs
        self.ham = ham

    @property
    def ham(self):
        """ I'm the current space's hamiltonian """
        return self._ham

    @ham.setter
    def ham(self,value):
        """ Set the current space's hamiltonian """
        self.check_ham_type(value)
        self._ham = value

    @property
    def basis_vecs(self):
        """ I'm the current space's basis vectors """
        return self._basis_vecs

    @basis_vecs.setter
    def basis_vecs(self,value):
        """ Set the current space's basis vectors """
        self.check_basis_vecs_type(value)
        self._basis_vecs = value

    @abstractmethod
    def check_ham_type(self, ham):
        """ checks the type of the hamiltonian according to concrete class implementation """

    @abstractmethod
    def check_basis_vecs_type(self, basis_vecs):
        """ checks the type of the basis vectors according to concrete class implementation """

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
    def interaction_matrix(self):
        """ defines the interaction matrix for space given some set of spanning vecs (basis_vecs)
            should be implemented by concrete class
        """

    @abstractmethod
    def subsp_ham(self, ham):
        """ defines a subspace hamiltonian for space given a hamiltonian in the space and
            a set of spanning vectors (basis_vecs)

            NB: ham cannot be constructed using the same points used to get basis_vecs

            should be implemented by concrete class
        """
        # TODO check about the "for a space given a hamiltonian in the space" bit

class NumpyArraySpace(HermitianSpaceInterface):
    """ defines Hermitian Space behavior for numpy arrays """

    @property
    def implementation_type(self):
        """ I'm the current space's implementation type """
        return np.ndarray

    def check_type_generic(self, value):
        """ helper method to verify all data in this implementation is in an np.ndarray """
        if not isinstance(value, self.implementation_type):
            raise ValueError("data should be of type np.ndarray")

    def check_basis_vecs_type(self, basis_vecs):
        """ checks the type of each basis vector (should be np.ndarray) """
        for basis_vec in basis_vecs:
            self.check_type_generic(basis_vec)

    def check_ham_type(self, ham):
        """ checks the type of the hamiltonian (should be np.ndarray) """
        self.check_type_generic(ham)

    def inner_product(self, vec1, vec2):
        """ defines inner product for numpy array space

            :param vec1:    the left vector of the inner product
            :param vec2:    the right vector of the inner product
            :returns:       inner product of vec1 & vec2
        """

        # Raises error if argument argument types are not np.ndarray (np.matrix is allowed)
        if (not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray)):
            raise ValueError("both vec1 and vec2 should be of type np.ndarray")

        # TODO ask Kemper about error checking
        # takes the conjugate transpose of vec2, and returns the inner product
        vec2_dagger = vec2.conj().T
        # try:
        return vec1 @ vec2_dagger # except TypeError: # print("Input should be in form (bra, bra)")

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

        # takes the conjugate transpose of vec2, and returns the expectation value
        vec2_dagger = vec2.conj().T # TODO Ask Kemper if this should have conj().
                                    # it means input is kinda funky
        return vec1 @ ham @ vec2_dagger
# NEXT:
# s_inv and sub_ham for abs and concrete
    def interaction_matrix(self):
        """ defines the interaction matrix for a NumpyArraySpace

            For an interaction matrix S:
            S[i,j] = inner_product(basis_vec_i, basis_vec_j)
        """

        # dimensions of square matrix will be numbner of basis vectors
        dim = len(self.basis_vecs)
        intrct = np.array([dim, dim])

        # S[i,j] = inner_product(basis_vec_i, basis_vec_j)
        for idx_i, vec_i in enumerate(self.basis_vecs):
            for idx_j, vec_j in enumerate(self.basis_vecs):
                intrct[idx_i, idx_j] = self.inner_product(vec_i, vec_j)

        return intrct

    def subsp_ham(self, ham):
        """ defines a subspace hamiltonian for space given a hamiltonian in the space and
            a set of spanning vectors (basis_vecs)

            NB: ham cannot be constructed using the same points used to get basis_vecs

            Subspace Ham[i,j] = expectation_value(basis_vec_i, ham, basis_vec_j)
        """

        # dimensions of square matrix will be number of basis vectors
        dim = len(self.basis_vecs)
        subsp_ham = np.array([dim, dim])

        # SubspaceHam[i,j] = expectation_value(basis_vec_i, ham, basis_vec_j)
        for idx_i, vec_i in enumerate(self.basis_vecs):
            for idx_j, vec_j in enumerate(self.basis_vecs):
                subsp_ham[idx_i, idx_j] = self.expectation_value(vec_i, ham, vec_j)

        return subsp_ham


def main():
    """ generates the image, hamiltonian, and overlap matrix """

# START Hamiltonian & Eigenvector Initialization
    # useful tuple when dealing with param_sets in this space
    ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")

    # user input
    n_qubits = 2
    b_x = 0
    j_x = 1
    j_z = 1
    b_zs = np.array([0,2,3])  # put your custom input here
    pbc = False

    # create a param_set for each b_z value
    param_sets = [None] * len(b_zs)
    for idx, b_z in enumerate(b_zs):
        param_sets[idx] = ParamSet(j_x,j_z,b_x,b_z)

    # initialize hamiltonians
    hamiltonian_initializer = HamiltonianInitializer()
    hams = [None] * len(b_zs)
    for idx, param_set in enumerate(param_sets):
        hams[idx] = hamiltonian_initializer.xxztype_hamiltonian(param_set,n_qubits,pbc)

    # calculate evecs for each ham
    evec_sets = [None] * len(b_zs)
    for idx, ham in enumerate(hams):
        evec_sets[idx] = hamiltonian_initializer.get_eigenpairs(ham)[1]
# END Hamiltonian & Eigenvector Initialization

# These instructions are for what main() should do next
# Finished: getting eigenvectors given a few training point b_z values
# Next:     make interface that deals with inner product and expectation value,
#           using the evecs given above
#           and new Bz values
#           unsure: do I take some new Bz values, calculate the ham, then do the sandwich to get
#           the subspace ham?






if __name__ == "__main__":
    main()
    
# shouldn't need n, evals, 
#  input: evecs, some ham of some osrt


# GARBAGIO BELOWIO
# class TrainingPointUtil:
#     DATA_POINTS = 100
#     """ determines fineness of curves """

#     ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
#     """" useful tuple when dealing with param sets in this space """

#     COMP_TOLERANCE = 1e-9
#     """ tolernace when comparing two floats """
    
#     def get_random_training_points(self, bzlist, evals, num_points):
#         """ returns random training points for the system

#             NOTE: This is a simplified case of "get training points".
#             It only accounts for Bz, but later updates will be more robust by exploring more
#             degrees of freedom
#         """
#             # TODO This is a simplified case of "get training points".
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
#     def get_interaction_matrix(self, evecs, num_points):
#         """ gets the interaction matrix for a given system """

#         # set up interaction matrix
#         s = np.zeros([num_points, num_points], dtype=complex)
#         # simple case: 2 qubits

#         # TODO Ask Kemper how to make this work for more qubits
#         # TODO also, is this even right? 
#         #       this iterates over all evecs but don't I only want the ground state?
#         for i in range(num_points):
#             for j in range(num_points):
#                 vector1 = np.matrix(evecs[:,i])
#                 vector2 = np.matrix(evecs[:,j])

#                 vector2 = vector2.conj().T
#                 s[i,j] = vector1 @ vector2
#                 # print(s)

#         return s

#     def calc_subspace_ham(self, hams, evecs, num_points): #call evecs basis vectors TODO
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
        
#         return np.linalg.eigvalsh(ham) # TODO output evecs as well

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

#         ax.axvline(1.0, ls = "--", color="blue")    # shows vertical line that represents [unsure] TODO

#         # plot training points. i[1][0] corresponds to the lowest energy 
#         for point in points:
#             plt.plot(point.b_z, point.energies[0], marker="o", color="blue")

#         # plot phats
#         for phat in phats:
#             plt.plot(phat.b_z, phat.energies[0], marker="o", color="orange")
#             plt.plot(phat.b_z, phat.energies[1], marker="o", color="orange")
#             # TODO only does lowest 2 energy states. can make this more flexible later          

#         plt.show()

#     def generate_xxz_type_spectrum(self, param_set, n=2, pbc=False):
#         """ calculates the different hamiltonians, eigenvalues, and interaction matrix for the system
#             and plots the spectrum on a plot"""

#         # j_x = param_set.j_x
#         # j_z = param_set.j_z
#         # b_x = param_set.b_x
#         # bzmin = param_set.bzmin
#         # bzmax = param_set.bzmax

#         # Initialize:
#         # plotting tool: array of evenly spaced numbers between bzmin and bzmax
#         bzlist = np.linspace(param_set.bzmin, param_set.bzmax, self.DATA_POINTS)
#         evals, evecs = self.get_eigenpairs(param_set, bzlist, n, pbc)

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
#             hams[idx_k] = self.xxztype_hamiltonian(param_set=b_zparam_set, n=n, pbc=pbc)

#         # Interaction Matrix:
#         # get the S interaction matrix
#         s = self.get_interaction_matrix(evecs=evecs, num_points=num_points)

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
#             print(energy_lists[idx_k]) # TODO ask Dr. Kemper what to do with this info
#             print()

#             # Check:
#             # checks to see if Inverse_Interaction • New_Ham = Eigenvals
#             print("S_inv • New ham (should correspond to Diagonalized value")
#             print(np.linalg.inv(s) @ new_ham_k)

#         self.plot_xxz_spectrum(bzlist, evals, points, phats, n)

#%%
