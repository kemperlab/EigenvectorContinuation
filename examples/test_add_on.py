#%%
from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eigh
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from ec_abstract.src import continuation_abstraction
from ..src import continuation_abstraction as ca


# TODO: figure out this import^, make the mimic, and run it

num_qubits = 2
b_x = .05
j_x = -1
j_z = 0
b_zs = np.array([0,2])  # put your custom input here

tbx = b_x
tjx = j_x
tjz = j_z
tbz = 1.5

ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
tps = ParamSet(tjx, tjz, tbx, tbz)

ps = [None] * len(b_zs)
for idx, b_z in enumerate (b_zs):
    ps[idx] = ParamSet(j_x,j_z,b_x,b_z)

tnp = ps

hilbert_space = ca.NumPyVectorSpace(tnp, num_qubits)

ec = ca.EigenvectorContinuer(hilbert_space,tps)

evals,evecs = ec.solve_gep()

print(evals, "\n", evecs)




#%%