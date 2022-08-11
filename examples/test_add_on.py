#!/usr/bin/env python3
""" Runs EVC on an example concrete implementation add-on of HSA

NOT YET IMPLEMENTED
"""

# from collections import namedtuple
# from abc import ABC, abstractmethod
# import numpy as np
# from scipy.linalg import eigh
# from scipy.linalg import null_space
# from matplotlib import pyplot as plt

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



# # TODO: figure out this import^, make the mimic, and run it

# num_qubits = 2
# b_x = .05
# j_x = -1
# j_z = 0
# b_zs = np.array([0,2])  # put your custom input here

# tbx = b_x
# tjx = j_x
# tjz = j_z
# tbz = 1.5

# ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
# tps = ParamSet(tjx, tjz, tbx, tbz)

# ps = [None] * len(b_zs)
# for idx, b_z in enumerate (b_zs):
#     ps[idx] = ParamSet(j_x,j_z,b_x,b_z)

# tnp = ps

# hilbert_space = ca.NumPyVectorSpace(tnp, num_qubits)

# ec = ca.EigenvectorContinuer(hilbert_space,tps)

# evals,evecs = ec.solve_gep()

# print(evals, "\n", evecs)




