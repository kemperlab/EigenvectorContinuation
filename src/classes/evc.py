"""
    EigenvectorContinuer:   A class used to take in any type of HSA and perform eigenvector
                            continuation operations using the HSA and some representation of
                            a target point. (abbr: EVC)
"""


# General Imports
# from collections import namedtuple
# from abc import ABC, abstractmethod
# import numpy as np
from scipy.linalg import eigh
# from scipy.linalg import null_space
# from matplotlib import pyplot as plt

# Local Imports
from src.classes.hsa import HilbertSpaceAbstract

__author__ = "Jack Howard"
__copyright__ = "Copyright (c) 2022 Kemper Lab"
__credits__ = "Jack Howard, Akhil Francis, Lex Kemper"

class EigenvectorContinuer():
    """ Houses the functionality to create a Hilbert Space of specified type and perform
        Eigenvector Continuation for a given set of training points and target points

        USE CASE:
        1.  Create an instance of a HilbertSpaceAbstract concrete class/subclass. Requires training
        points (and/or other input depending on implementation)
        2.  Input: a target point to use and calculate the subspace hamiltonian

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
        """ I'm this EVC's hilbert space """
        return self._hilbert_space

    @property
    def overlap_matrix(self):
        """ I'm this EVC's last calculated overlap matrix """
        return self._overlap_matrix

    @property
    def sub_ham(self):
        """ I'm this EVC's last calculated subspace hamiltonian """
        return self._sub_ham

    @property
    def current_target_point(self):
        """ I'm this EVC's current target point. I'm used to create the sub_ham"""
        return self._current_target_point

    @property
    def evals(self):
        """ I'm this EVC's last calculated set of eigenvalues for the diagonalized subspace ham """
        return self._evals

    @property
    def evecs(self):
        """ I'm this EVC's last calculated set of eigenvectors for the diagonalized subspace ham """
        return self._evecs

    def __init__(self, hilbert_space, target_point):
        """ initializes the EVC

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
        """ calculates the subspace hamiltonian based on the EVC's current Hilbert Space and
            target point

            :param input_target_point:  [OPTIONAL] can be used to update the current target
                                        point of the EVC as needed

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
        """ solves the generalized eigencvalue problem for this EVC

            :param input_training_points:   used to calculate the current hilbert space's
                                            overlap matrix. If None is passed, will default
                                            to current training_points in the hilbert space
            :param input_target_point:      used to calculate the current hilbert space's
                                            subspace hamiltonian. If None is passed, will default
                                            to current_target_point in this EVC
            :returns:                       the evals, evecs calculated
        """

        overlap = self.calc_overlap_matrix(input_training_points)
        subspace = self.calc_sub_ham(input_target_point)


        self._evals, self._evecs = eigh(subspace, overlap)

        return self.evals, self.evecs
