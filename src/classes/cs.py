""" Concrete implementation of HSA that stores and calculates products using the Qiskit library

CircuitSpace uses the Qiskit library to define the behavior for inner products and expectation
values for data represented as quatum circuits. For a set of input training points, it can use
these to calculate the overlap matrix and subspace hamiltonian for the given space.

NOT YET IMPLEMENTED
"""


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
__version__ = "1.0.1"
__maintainer__ = "Jack H. Howard"
__email__ = "jhhoward@ncsu.edu"
__status__ = "Development"