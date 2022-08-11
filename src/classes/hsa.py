""" HilbertSpaceAbstract:   Abstract class used to outline how concrete Hilbert Spaces behave

Hilbert Spaces are used to create EigenvectorContinuer objects. They require an implementation of
an inner product and an expectation value, which are used to calculate the overlap matrix and
subspace hamiltonian of the system. Additionally, they require a method of determinig which vector
to select from a list of eigenvectors (usually these correspond to different energy levels in a
system).
"""


# General Imports
# from collections import namedtuple
from abc import ABC, abstractmethod
# import numpy as np
# from scipy.linalg import eigh
# from scipy.linalg import null_space
# from matplotlib import pyplot as plt

# Local Imports
# import src.util.param_set

__author__ = "Jack H. Howard, Akhil Francis, Alexander F. Kemper"
__citation__ = "" # TODO Arxiv or doi
__copyright__ = "Copyright (c) 2022 Kemper Lab"
__credits__ = ["Jack H. Howard", "Akhil Francis", "Alexander F. Kemper",
               "Anjali A. Agrawal", "Efekan Kökcü"]
__license__ = "BSD-2-Clause-Patent"
__version__ = "1.0.1"
__maintainer__ = "Jack H. Howard"
__email__ = "jhhoward@ncsu.edu"
__status__ = "Development"

class HilbertSpaceAbstract(ABC):
    """ Defines behavior for objects to have a hamiltonian, inner product, expectation value, and
    selected vector

        INPUT:
            a set of training vectors

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
