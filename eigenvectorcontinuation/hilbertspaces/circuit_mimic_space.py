""" Concrete implementation of HSA that mimics the algebra used in a quantum circuit

CircuitMimicSpace is a simplified linear algebra representation of the computations run on
quantum circuits. For a set of input training points, it defines the behavior for inner products
and expectation values in this space and uses that behavior to calculate the overlap matrix and
subspace hamiltonian for the given space. Does not use Qiskit (Please see CircuitSpace for the
Qiskit implementation).

NOT YET IMPLEMENTED
"""
# TODO Implement CircuitMimicSpace

# # General Imports
# from collections import namedtuple
# from abc import ABC, abstractmethod
# import numpy as np
# from scipy.linalg import eigh
# from scipy.linalg import null_space
# from matplotlib import pyplot as plt

# # Local Imports
# import src.util.param_set

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
