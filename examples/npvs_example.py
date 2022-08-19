#!/usr/bin/env python3
""" This script runs through the process of using the EC code with the provided NumPyVectorSpace
concrete implementation of a HilbertSpaceAbstract

"""

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

# Global imports
import numpy as np

# Local Imports
from eigenvectorcontinuation.continuer.eigenvector_continuer import EigenvectorContinuer as evc
from eigenvectorcontinuation.util.param_set import ParamSet
from eigenvectorcontinuation.hilbertspaces.numpy_vector_space import NumPyVectorSpace as npvs
from eigenvectorcontinuation.methods import plot_xxz_spectrum

# Custom Input
# TODO clean up these variable names
num_qubits = 2
b_x = .05
j_x = -1
j_z = 0
b_zs = np.array([0.1, 2])  # put your custom bz inputs here

tbx = b_x
tjx = j_x
tjz = j_z
tbz = 1.5

tps = ParamSet(tjx, tjz, tbx, tbz)

ps = [None] * len(b_zs)

for idx, b_z in enumerate (b_zs):
    ps[idx] = ParamSet(j_x,j_z,b_x,b_z)

tnp = ps

# Create Hilbert Space
hilbert_space = npvs(tnp, num_qubits)

ec = evc(hilbert_space,tps)

evals,evecs = ec.solve_gep()

# Print resulting eigenvalues and eigenvectors
print(evals, "\n", evecs)

# Plot resulting spectrum
plot_xxz_spectrum(0,3,ec)
