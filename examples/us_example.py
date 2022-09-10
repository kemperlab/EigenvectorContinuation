#!/usr/bin/env python3
""" This script runs through the process of using the EC code with the provided UnitarySpace
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
from eigenvectorcontinuation.hilbertspaces.unitary_space import UnitarySpace as us
from eigenvectorcontinuation.methods import plot_xxz_spectrum

# Custom Input
num_qubits = 2

# set training/target point values for J_x, J_z, and B_x
train_bx = .05
train_jx = -1
train_jz = 0

# set training point values for B_z
train_bzs = np.array([0.1, 2])  # put your custom bz inputs here

# set target point B_z value
targ_bz = 1.5

# construct the target point as a tuple
targ_pt = ParamSet(j_x=train_jx, j_z=train_jz, b_x=train_bx, b_z=targ_bz)

# create empty list of training points
train_pts = [None] * len(train_bzs)

# populate training points list
for idx, train_bz in enumerate (train_bzs):
    train_pts[idx] = ParamSet(j_x=train_jx, j_z=train_jz, b_x=train_bx, b_z=train_bz)

# Create Hilbert Space
hilbert_space = us(training_points=train_pts, num_qubits=num_qubits)

# Create EigenvectorContinuer
ec = evc(hilbert_space=hilbert_space, target_point=targ_pt)

# solve the generalized eigenvalue problem for the current EC
evals,evecs = ec.solve_gep()

# Print resulting eigenvalues and eigenvectors
print("Eigenvalues: \n", evals, "\nEigenvectors: \n", evecs, "\n")

# Plot resulting spectrum
plot_xxz_spectrum(bzmin=0,bzmax=3,evec_cont=ec)
