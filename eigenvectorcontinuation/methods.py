"""
    Includes methods helpful to running and plotting the EVC process

    Plotting tools          Multiple target points are used to produce a simple plot. Details
                            may vary depending on implementation specifics
"""


# General Imports
import numpy as np
from matplotlib import pyplot as plt

# Local Imports
from eigenvectorcontinuation.continuer.eigenvector_continuer import EigenvectorContinuer
from eigenvectorcontinuation.util.param_set import ParamSet

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

# Plotting tools

def plot_xxz_spectrum(bzmin, bzmax, evec_cont: EigenvectorContinuer):
    """ plots the spectrum of eigenvalues for a given EVC

        :param bzmin:       the minimum b_z value to plot
        :param bzmax:       the maximum b_z value to plot
        :param evec_cont:   the EVC to plot (plots training/target points and expected energies)

    """

    # determines fine-ness of curves
    data_points = 100


    # initializes plot and axes
    axes = plt.subplots()[1]
    axes.set_xlabel("$B_Z$")
    axes.set_ylabel("Energy")

    # PLOT POINTS FROM INPUT EVC
    # sets up hamiltonian initializer to reduce overhead in for loop
    ham_init = evec_cont.hilbert_space.HamiltonianInitializer()

    # "for every training point in the EVC, ..."
    for training_point in evec_cont.hilbert_space.training_points:
        # "... calculate its hamiltonian, ..."
        ham = ham_init.xxztype_hamiltonian(param_set=training_point,
                                           n_qubits=evec_cont.hilbert_space.num_qubits)
        # "... get the eigenvalues of that hamiltonian, ..."
        evals = ham_init.calc_eigenpairs(ham)[0]

        # " ... and plot the lowest energy eigenvalue"
        # ground_state_eval = EigenvectorContinuer.hilbert_space.select_vec(evals)
        # for current_eval in evals: TODO add support for select_vec
        plt.plot(training_point.b_z, evals[0], marker="o", color="blue")

    # gets the evals of the evc to reduce overhead of the for loop
    evc_evals = evec_cont.evals

    # plot each target point
    for evc_eval in evc_evals:
        plt.plot(evec_cont.current_target_point.b_z, evc_eval, marker="o", color="red")

    # PLOT EXPECTED ENERGY CURVES
    # get parameters for expected curves
    j_x = evec_cont.hilbert_space.training_points[0].j_x
    j_z = evec_cont.hilbert_space.training_points[0].j_z
    b_x = evec_cont.hilbert_space.training_points[0].b_x

    # get list of spaced out points
    bzlist = np.linspace(bzmin, bzmax, data_points)

    # plot the lines
    all_evals = np.zeros([len(bzlist), 2**evec_cont.hilbert_space.num_qubits])
    for idx, b_z in enumerate(bzlist):
        param_set = ParamSet(j_x, j_z, b_x, b_z)

        ham = ham_init.xxztype_hamiltonian(param_set=param_set,
                                           n_qubits=evec_cont.hilbert_space.num_qubits)
        # if idx == 50:
        # print(ham_init.calc_eigenpairs(ham)[0])
        # print(idx)
        all_evals[idx,:] = ham_init.calc_eigenpairs(ham)[0]

    for idx in range(2**evec_cont.hilbert_space.num_qubits):
        axes.plot(bzlist, all_evals[:,idx], 'k-')
        # print(all_evals[:,idx])

    axes.axvline(1.0, ls = "--", color="blue")  # shows vertical line that represents crossing point

    plt.show()
