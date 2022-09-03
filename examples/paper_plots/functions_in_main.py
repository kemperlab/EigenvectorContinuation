from hamiltonian import *
from continuers import *
from h2molecule import *
def XXZfninmain():

    # Bz = 0.01
    Bz=0.0
    Bx = 0.0
    N = 4
    pbc = True
    # Jz = 1
    # Jmin = -4.0
    # Jmax = 4.0
    J = 1
    Jzmin = 0.0
    Jzmax = 2.0


    # fig, ax, Jlist, eval_stor = show_XXZ_spectrumwithJ(Jmin=Jmin,Jmax=Jmax,Jz=Jz,Bx=Bx,Bz=Bz,pbc=pbc,N=N)
    fig, ax, Jzlist, eval_stor = show_XXZ_spectrumwithJz(Jzmin=Jzmin, Jzmax=Jzmax, J=J, Bx=Bx, Bz=Bz, pbc=pbc, N=N)
    # Set up training parameter sets for eigenvector continuer
    # Jlist_training = [0.7, 1.4]
    # training_paramlist = [[J, Jz, Bx, Bz, N, pbc] for J in Jlist_training]
    # Jzlist_training = [0.0, 2.0]
    Jzlist_training = [0.15,0.45 ]
    training_paramlist = [[J, Jz, Bx, Bz, N, pbc] for Jz in Jzlist_training]
    # _qc for passing parameters to the quantum circuit, because I prefer dictionaries than lists
    # training_paramlist_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for J in Jlist_training]

    # if 'ax' in locals():
    #     for b in Jlist_training:
    #         ax.axvline(b)

    if 'ax' in locals():
        for b in Jzlist_training:
            ax.axvline(b)

    # Set up target parameter sets for eigenvector continuer
    # Jlist_target = [ 0.2,0.5,0.8,0.9,1.2,1.5,1.7]
    # Jlist_target = [-0.2, -0.5, -0.8, -0.9, -1.2, -1.5, -1.7]
    # target_paramlist = [[J, Jz, Bx, Bz, N, pbc] for J in Jlist_target]
    Jzlist_target = [0.1, 0.3, 0.5, 0.7,0.9,1.1, 1.3, 1.5, 1.7,1.9]
    target_paramlist = [[J, Jz, Bx, Bz, N, pbc] for Jz in Jzlist_target]
    # target_paramlist_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for J in Jlist_target]
    #################### INPUT parameters over ######################################

    # Object that knows how to deal with the various operations needed
    vectorspace = vector_methods(XXZ_hamiltonian)
    EVcontinuer = vector_continuer(vectorspace,
                                   XXZ_hamiltonian,
                                   Mag_op,
                                   training_paramlist,
                                   target_paramlist,
                                   N)
    EVcontinuer.get_base_eigenvectors()
    EVcontinuer.get_target_eigenvectors(ortho=False)

    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            # ax.plot(Jlist_target, np.real(EVcontinuer.target_evals[:, ip]), 'o', color="b")
            ax.plot(Jzlist_target, np.real(EVcontinuer.target_evals[:, ip]), 'o', color="b")
    plt.show()

    #####################
    ############################################
def H2fninmain():
    # for h2 molecule
    N = 2
    distlist_training = [0.1,1.6]
    # distlist_training = [ 1.6,1.7]
    # distlist_training = [0.71, 0.74]
    training_paramlist = [[dist] for dist in distlist_training]
    distlist_target = [ 0.735,]
    # distlist_target = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    target_paramlist = [[dist] for dist in distlist_target]
    vectorspace = vector_methods(h2_hamiltonian)
    EVcontinuer = vector_continuer(vectorspace,
                                   h2_hamiltonian,
                                   Mag_op,
                                   training_paramlist,
                                   target_paramlist,
                                   N)

    EVcontinuer.get_base_eigenvectors()
    EVcontinuer.get_target_eigenvectors(ortho=False)
    # print("values in object before",EVcontinuer.target_evals)
    # if I don't use the copy function the object pointer will be passed and it messes up lets say when we add nuclear energies
    EVC_total_energies = np.copy(EVcontinuer.target_evals)
    nuclear_repulsion_energies=[]
    for ir, r in enumerate(distlist_target):
        h2temp = h2molecule(dist=r)
        h2temp.make_ham_matrix()
        h2temp.solve_for_energies()
        nuclear_repulsion_energies.append(h2temp.nuclear_repulsion_energy)
        EVC_total_energies[ir, :] = EVC_total_energies[ir, :] + h2temp.nuclear_repulsion_energy
    # print("values in object after", EVcontinuer.target_evals)
    # print("nuclear_repulsion_energies in main ",nuclear_repulsion_energies)
    fig, ax, distlist, eval_stor, eval_stor_tot = show_h2_spectrum(distzmin=0.08, distmax=2.0,npoints=10)

    for b in distlist_training:
        ax[0].axvline(b)
        ax[1].axvline(b)
    for ip in range(len(training_paramlist)):
        ax[0].plot(distlist_target, np.real(EVcontinuer.target_evals[:, ip]) , 'o', color="b")
        ax[1].plot(distlist_target, np.real(EVC_total_energies[:, ip]), 'o', color="k")
        # ax[1].plot(distlist_target, np.real(EVcontinuer.target_evals[:, ip]) + nuclear_repulsion_energies[1], 'o', color="k")
    # for i in range(1):
    #     bottom_side = ax[i].spines["bottom"]

    # tag = "hydrogen_distlist_training"+ str(distlist_training) + "distlist_target" + str(distlist_target)
    # fignamepdf = "plots/hydrogen/" + tag + ".pdf"
    # fig.savefig(fignamepdf)
    # fignamepng = "plots/hydrogen/" + tag + ".png"
    # fig.savefig(fignamepng)
    plt.show()

def XYinQCinmain():
    ## XY model

    J = -1
    # Bx = 0.5
    Bx = 0.1
    N = 2
    pbc = False

    fig, ax, Bzlist, eval_stor = show_XY_spectrum(N=N, Bzmin=0, Bzmax=2, Bx=Bx, J=J, pbc=pbc)

    # Set up training parameter sets for eigenvector continuer
    # Bzlist = [0,0.2,0.75]
    # Bzlist_training = [0,1.9]
    # Bzlist_training = [0.5, 1.9]
    Bzlist_training = [0.1, 1.3]
    # Bzlist_training = [0.1, 1.9]
    # Bzlist_training = [0, 0.1,0.2]
    # Bzlist_training = [0, 0.8]
    training_paramlist = [[J, Bx, Bz, N, pbc] for Bz in Bzlist_training]
    # _qc for passing parameters to the quantum circuit, because I prefer dictionaries than lists
    training_paramlist_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for Bz in Bzlist_training]

    if 'ax' in locals():
        for b in Bzlist_training:
            ax.axvline(b)

    # Set up target parameter sets for eigenvector continuer
    # Bzlist = np.linspace(0,2,20)
    # Bzlist_target = [0.7,1.8]

    # Bzlist_target = [1.5,1.7,1.9]
    # Bzlist_target = [0.3, 0.5, 0.7, 1.1, 1.3, 1.5, 1.7]
    Bzlist_target = [0.3, 0.5, 0.7, 1.1, 1.5, 1.7, 1.9]

    target_paramlist = [[J, Bx, Bz, N, pbc] for Bz in Bzlist_target]
    target_paramlist_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for Bz in Bzlist_target]
    #################### INPUT parameters over ######################################

    # Object that knows how to deal with the various operations needed
    vectorspace = vector_methods(XY_hamiltonian)

    # Reference vector is internal for now
    # vectorspace = unitary_methods(N, XY_hamiltonian)

    # vectorspace = circuit_methods(N,XY_hamiltonian_Qiskit)

    EVcontinuer = vector_continuer(vectorspace,
                                   XY_hamiltonian,
                                   Mag_op,
                                   training_paramlist,
                                   target_paramlist,
                                   N)

    EVcontinuer.get_base_eigenvectors()
    # EVcontinuer.form_orthogonal_basis()

    # added_evals = EVcontinuer.get_target_eigenvectors(ortho=True)
    # added_evals = EVcontinuer.get_target_eigenvectors(ortho=False)
    EVcontinuer.get_target_eigenvectors(ortho=False)
    # print("Eigen values_continuer: ",added_evals)
    # print("Eigen values_continuer: ",EVcontinuer.target_evals)
    # print("basis_exact: ", basis_exact)
    # print("basis_exact: ", basis_exact_from_class)
    # print("basis_exact inside class: ", EVcontinuer.base_vecs)
    # print("Mag outside class: ", Mag_evals)
    # print("basis_exact inside class: ", EVcontinuer.target_magnetization)

    # exit()
    # mimic_evals = get_evals_targetlist_mimic(training_paramlist=training_paramlist_qc,target_paramlist=target_paramlist_qc)
    # mimic_evals = get_evals_targetlist_mimic(training_paramlist=training_paramlist_qc,target_paramlist=target_paramlist_qc,\
    #                                          basis_vecs=basis_exact,Basis_exact_flag=True)
    #####################
    ######################## INPUT backend information of the quatnum machine ######################
    ##########
    # backend_name = "qasm_simulator"
    backend_name = "ibmq_qasm_simulator"
    # backend_name = "ibmq_bogota"
    # layout = [0,1,2]
    # backend_name = "ibm_lagos"
    # layout = [3,5,6]
    # backend_name = "ibmq_manila"
    # layout = [2,3,4]
    # backend_name = "ibmq_montreal"
    # layout = [0, 1, 2]

    # layout = [5, 3, 4]
    # shots=8192
    ##########
    # 3site
    # backend_name = "qasm_simulator"
    #     backend_name = "ibmq_bogota"
    #     layout = [1, 2, 3]
    # layout = [0, 1, 2]
    layout = [0, 1, 2]
    # shots = 8192
    shots = 20000
    date_flag = True
    # date_flag=False
    ###################################### INPUT backend over #######################################

    # qasm_circuit_evals_separate =  get_evals_targetlist_qasmcirc_sepaate\
    #     ( training_paramlist=training_paramlist_qc, target_paramlist=target_paramlist_qc,backend_name=backend_name,layout=layout,shots=shots)

    # # # uncomment for machine
    # # qsearch_circuit_evals_bundle = get_evals_targetlist_qsearchcirc\
    # #     (training_paramlist=training_paramlist_qc, target_paramlist=target_paramlist_qc,backend_name=backend_name,layout=layout,shots=shots)
    # qsearch_circuit_evals_bundle = get_evals_targetlist_qsearchcirc(training_paramlist=training_paramlist_qc, target_paramlist=target_paramlist_qc,\
    #             backend_name=backend_name,layout=layout,shots=shots, basis_vecs=EVcontinuer.base_vecs,Basis_exact_flag=True,date_flag=date_flag)

    # qasm_circuit_evals_bundle = get_evals_targetlist_qasmcirc(
    #     training_paramlist=training_paramlist_qc, target_paramlist=target_paramlist_qc,backend_name=backend_name,layout=layout,shots=shots)

    ################# This is to reconstruct and plot from the saved matrix values(results)
    # reconstruct_qsearch_circuit_evals_bundle = evals_from_reading(target_paramlist=target_paramlist_qc, training_paramlist=training_paramlist_qc,target_paramlist_new = target_paramlist_qc, optimizationlevel=3, backend_name="ibmq_manila",
    #                    layout=[2,3,4])
    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            ax.plot(Bzlist_target, np.real(EVcontinuer.target_evals[:, ip]), 'o', color="b")
    #         # ax.plot(Bzlist_target,np.real(mimic_evals[:,ip]),'*',color="r")
    #         ax.plot(Bzlist_target, np.real(reconstruct_qsearch_circuit_evals_bundle[:, ip]), '^', color="g")
    #         # ax.plot(Bzlist_target, np.real(qasm_circuit_evals_separate[:, ip]), 's', color="orange")
    #         # ax.plot(Bzlist_target, np.real(qasm_circuit_evals_bundle[:, ip]), '^', color="k")
    #         ax.plot(Bzlist_target, np.real(qsearch_circuit_evals_bundle[:, ip]), '^', color="k")
    ####################

    ###########################

    if (date_flag):
        import datetime
        x = datetime.datetime.now()
        date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J) + "pbc=" + str(pbc) + "Bztrain" + str(
            Bzlist_training) + \
              "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(
            layout) + "shots=" + str(shots) + "date" + date + ".pdf"
    else:
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J) + "pbc=" + str(pbc) + "Bztrain" + str(
            Bzlist_training) + \
              "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(
            layout) + "shots=" + str(shots) + ".pdf"
    fignamepdf = "plots/EVC/evc" + tag
    # fig.savefig(fignamepdf)
    #
    # fignamepdf = "plots/reconstructing_" + "Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(Bzlist_target) \
    #       + "backend_name=" + "ibmq_manila" + "layout=" + str([2,3,4]) +"25_11_2021"+ ".pdf"
    # fig.savefig(fignamepdf)

    plt.show()

def XYallinmain():
    ## XY model

    J = -1
    # Bx = 0.5
    Bx = 0.1
    N = 2
    pbc = False

    fig, ax, Bzlist, eval_stor = show_XY_spectrum(N=N, Bzmin=0, Bzmax=2, Bx=Bx, J=J, pbc=pbc)

    # Set up training parameter sets for eigenvector continuer
    # Bzlist = [0,0.2,0.75]
    # Bzlist_training = [0,1.9]
    # Bzlist_training = [0.5, 1.9]
    Bzlist_training = [0.1, 1.3]
    # Bzlist_training = [0.1, 1.9]
    # Bzlist_training = [0, 0.1,0.2]
    # Bzlist_training = [0, 0.8]
    training_paramlist = [[J, Bx, Bz, N, pbc] for Bz in Bzlist_training]
    # _qc for passing parameters to the quantum circuit, because I prefer dictionaries than lists
    training_paramlist_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for Bz in Bzlist_training]

    if 'ax' in locals():
        for b in Bzlist_training:
            ax.axvline(b)

    # Set up target parameter sets for eigenvector continuer
    # Bzlist = np.linspace(0,2,20)
    # Bzlist_target = [0.7,1.8]

    # Bzlist_target = [1.5,1.7,1.9]
    # Bzlist_target = [0.3, 0.5, 0.7, 1.1, 1.3, 1.5, 1.7]
    Bzlist_target = [0.3, 0.5, 0.7, 1.1, 1.5, 1.7, 1.9]

    target_paramlist = [[J, Bx, Bz, N, pbc] for Bz in Bzlist_target]
    target_paramlist_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for Bz in Bzlist_target]
    #################### INPUT parameters over ######################################

    # Object that knows how to deal with the various operations needed
    vectorspace = vector_methods(XY_hamiltonian)

    # Reference vector is internal for now
    # vectorspace = unitary_methods(N, XY_hamiltonian)

    # vectorspace = circuit_methods(N,XY_hamiltonian_Qiskit)

    EVcontinuer = vector_continuer(vectorspace,
                                   XY_hamiltonian,
                                   Mag_op,
                                   training_paramlist,
                                   target_paramlist,
                                   N)

    EVcontinuer.get_base_eigenvectors()
    # EVcontinuer.form_orthogonal_basis()

    # added_evals = EVcontinuer.get_target_eigenvectors(ortho=True)
    # added_evals = EVcontinuer.get_target_eigenvectors(ortho=False)
    EVcontinuer.get_target_eigenvectors(ortho=False)
    # print("Eigen values_continuer: ",added_evals)
    # print("Eigen values_continuer: ",EVcontinuer.target_evals)
    # print("basis_exact: ", basis_exact)
    # print("basis_exact: ", basis_exact_from_class)
    # print("basis_exact inside class: ", EVcontinuer.base_vecs)
    # print("Mag outside class: ", Mag_evals)
    # print("basis_exact inside class: ", EVcontinuer.target_magnetization)

    # exit()
    # mimic_evals = get_evals_targetlist_mimic(training_paramlist=training_paramlist_qc,target_paramlist=target_paramlist_qc)
    # mimic_evals = get_evals_targetlist_mimic(training_paramlist=training_paramlist_qc,target_paramlist=target_paramlist_qc,\
    #                                          basis_vecs=basis_exact,Basis_exact_flag=True)
    #####################
    ######################## INPUT backend information of the quatnum machine ######################
    ##########
    # backend_name = "qasm_simulator"
    backend_name = "ibmq_qasm_simulator"
    # backend_name = "ibmq_bogota"
    # layout = [0,1,2]
    # backend_name = "ibm_lagos"
    # layout = [3,5,6]
    # backend_name = "ibmq_manila"
    # layout = [2,3,4]
    # backend_name = "ibmq_montreal"
    # layout = [0, 1, 2]

    # layout = [5, 3, 4]
    # shots=8192
    ##########
    # 3site
    # backend_name = "qasm_simulator"
    #     backend_name = "ibmq_bogota"
    #     layout = [1, 2, 3]
    # layout = [0, 1, 2]
    layout = [0, 1, 2]
    # shots = 8192
    shots = 20000
    date_flag = True
    # date_flag=False
    ###################################### INPUT backend over #######################################

    # qasm_circuit_evals_separate =  get_evals_targetlist_qasmcirc_sepaate\
    #     ( training_paramlist=training_paramlist_qc, target_paramlist=target_paramlist_qc,backend_name=backend_name,layout=layout,shots=shots)

    # # # uncomment for machine
    # # qsearch_circuit_evals_bundle = get_evals_targetlist_qsearchcirc\
    # #     (training_paramlist=training_paramlist_qc, target_paramlist=target_paramlist_qc,backend_name=backend_name,layout=layout,shots=shots)
    # qsearch_circuit_evals_bundle = get_evals_targetlist_qsearchcirc(training_paramlist=training_paramlist_qc, target_paramlist=target_paramlist_qc,\
    #             backend_name=backend_name,layout=layout,shots=shots, basis_vecs=EVcontinuer.base_vecs,Basis_exact_flag=True,date_flag=date_flag)

    # qasm_circuit_evals_bundle = get_evals_targetlist_qasmcirc(
    #     training_paramlist=training_paramlist_qc, target_paramlist=target_paramlist_qc,backend_name=backend_name,layout=layout,shots=shots)

    ################# This is to reconstruct and plot from the saved matrix values(results)
    # reconstruct_qsearch_circuit_evals_bundle = evals_from_reading(target_paramlist=target_paramlist_qc, training_paramlist=training_paramlist_qc,target_paramlist_new = target_paramlist_qc, optimizationlevel=3, backend_name="ibmq_manila",
    #                    layout=[2,3,4])
    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            ax.plot(Bzlist_target, np.real(EVcontinuer.target_evals[:, ip]), 'o', color="b")
    #         # ax.plot(Bzlist_target,np.real(mimic_evals[:,ip]),'*',color="r")
    #         ax.plot(Bzlist_target, np.real(reconstruct_qsearch_circuit_evals_bundle[:, ip]), '^', color="g")
    #         # ax.plot(Bzlist_target, np.real(qasm_circuit_evals_separate[:, ip]), 's', color="orange")
    #         # ax.plot(Bzlist_target, np.real(qasm_circuit_evals_bundle[:, ip]), '^', color="k")
    #         ax.plot(Bzlist_target, np.real(qsearch_circuit_evals_bundle[:, ip]), '^', color="k")
    ####################

    ###########################

    if (date_flag):
        import datetime
        x = datetime.datetime.now()
        date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J) + "pbc=" + str(pbc) + "Bztrain" + str(
            Bzlist_training) + \
              "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(
            layout) + "shots=" + str(shots) + "date" + date + ".pdf"
    else:
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J) + "pbc=" + str(pbc) + "Bztrain" + str(
            Bzlist_training) + \
              "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(
            layout) + "shots=" + str(shots) + ".pdf"
    fignamepdf = "plots/EVC/evc" + tag
    # fig.savefig(fignamepdf)
    #
    # fignamepdf = "plots/reconstructing_" + "Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(Bzlist_target) \
    #       + "backend_name=" + "ibmq_manila" + "layout=" + str([2,3,4]) +"25_11_2021"+ ".pdf"
    # fig.savefig(fignamepdf)

    plt.show()
    # exit()

    # #############construct second figure : more reconstruction points##############
    # Bzlist_target_new = [0.5, 1.0, 1.5, 1.7, 1.9]
    # fig, ax = show_XY_spectrum(N=N, Bzmin=0, Bzmax=2, Bx=Bx)
    # if 'ax' in locals():
    #     for b in Bzlist_training:
    #         ax.axvline(b)
    # target_paramlist_new_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for Bz in Bzlist_target_new]
    # reconstruct_qsearch_circuit_evals_bundle = evals_from_reading(target_paramlist=target_paramlist_qc,
    #                                                               training_paramlist=training_paramlist_qc,target_paramlist_new = target_paramlist_new_qc,
    #                                                               optimizationlevel=3, backend_name="ibmq_manila",
    #                                                               layout=[2, 3, 4])
    # if 'ax' in locals():
    #     for ip in range(len(training_paramlist)):
    #         ax.plot(Bzlist_target,np.real(added_evals[:,ip]),'o',color="b")
    #         ax.plot(Bzlist_target_new,np.real(reconstruct_qsearch_circuit_evals_bundle[:,ip]),'*',color="r")
    #
    # plt.show()

    # ############ construct third figure : for magnetization##############
    #
    # fig, ax = show_XY_spectrum(N=N, Bzmin=0, Bzmax=2, Bx=Bx)
    # if 'ax' in locals():
    #     for b in Bzlist_training:
    #         ax.axvline(b)
    # # target_paramlist_new_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for Bz in Bzlist_target_new]
    # # reconstruct_qsearch_circuit_evals_bundle = evals_from_reading(target_paramlist=target_paramlist_qc,
    # #                                                               training_paramlist=training_paramlist_qc,target_paramlist_new = target_paramlist_new_qc,
    # #                                                               optimizationlevel=3, backend_name="ibmq_manila",
    # #                                                               layout=[2, 3, 4])
    # if 'ax' in locals():
    #     for ip in range(len(training_paramlist)):
    #         ax.plot(Bzlist_target,np.real(Mag_evals[:,ip]),'o',color="b")
    #         # ax.plot(Bzlist_target_new,np.real(reconstruct_qsearch_circuit_evals_bundle[:,ip]),'*',color="r")
    #
    # plt.show()
    # # ##############################

    ################ This is for LCU ###########

    # Bzlist_target = [0.7]
    # Bzlist_target = [1.5,1.6,1.8]
    target_paramlist = [[J, Bx, Bz, N, pbc] for Bz in Bzlist_target]
    target_paramlist_qc = [{"J": J, "Bx": Bx, "Bz": Bz, "N": N, "pbc": pbc} for Bz in Bzlist_target]

    # LCU_gs_list_mimic_exact = get_LCU_gs_list_mimic_exact(training_paramlist=training_paramlist_qc,
    #                                        target_paramlist=target_paramlist_qc, \
    #                                        basis_vecs=EVcontinuer.base_vecs, Basis_exact_flag=True)

    #
    # LCU_gs_list_mimic,LCU_coeffs_gs_list_mimic = make_target_LCU_mimic(training_paramlist=training_paramlist_qc,
    #                                        target_paramlist=target_paramlist_qc, \
    #                                        basis_vecs=EVcontinuer.base_vecs, Basis_exact_flag=True)
    # print("LCU_gs_list_mimic_exact")
    # print(LCU_gs_list_mimic_exact)
    # print("LCU_gs_list_mimic")
    # print(LCU_gs_list_mimic)
    # testcoeffs_list = [np.array([0.99994899, 0.01010049]), np.array([0.99994899, 0.01010049])]
    # LCU_m_list_qsearch = get_mag_qsearch(training_paramlist=training_paramlist_qc,target_paramlist=target_paramlist_qc,\
    #     LCU_coeffs_gs_list = EVcontinuer.LCU_coeffs_list, basis_vecs=EVcontinuer.base_vecs, backend_name=backend_name,layout=layout,shots=shots,Basis_exact_flag=True)
    #
    # print("LCU_mvals")
    # print(LCU_m_list_qsearch)
    # print("Mag_evals")
    # print(EVcontinuer.target_magnetization)
    # print(LCU_coeffs_gs_list_mimic)
    # testcoeffs_list = [np.array([0.99994899, 0.01010049]), np.array([0.99994899, 0.01010049])]

    LCU_E_list_qsearch, LCU_m_list_qsearch = get_energy_qsearch(training_paramlist=training_paramlist_qc,
                                                                target_paramlist=target_paramlist_qc,
                                                                LCU_coeffs_gs_list=EVcontinuer.LCU_coeffs_list,
                                                                basis_vecs=EVcontinuer.base_vecs,
                                                                backend_name=backend_name, layout=layout, shots=shots,
                                                                Basis_exact_flag=True, date_flag=date_flag)
    # print("LCU_Evals")
    # print(LCU_E_list_qsearch)
    fig, ax = plt.subplots()
    for b in Bzlist_training:
        ax.axvline(b)
    for j in range(2 ** N):
        ax.plot(Bzlist, eval_stor[:, j], 'k-')
    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            ax.plot(Bzlist_target, np.real(EVcontinuer.target_evals[:, ip]), 'o', color="b")
    if 'ax' in locals():
        ax.plot(Bzlist_target, np.real(LCU_E_list_qsearch), 's', color="red")
        # ax.plot(Bzlist_target, np.real(qsearch_circuit_evals_bundle[:, ip]), '^', color="k")
    ax.set_xlabel("$B_z$")
    ax.set_ylabel("Energy")
    if (date_flag):
        import datetime

        x = datetime.datetime.now()
        date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
        tagpdf = "LCU_energy Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
            Bzlist_training) + "Bztarget" + str(
            Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(
            shots) + "date_" + date + " .pdf"
    else:
        tagpdf = "LCU_energy Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
            Bzlist_training) + "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(
            layout) + "shots=" + str(shots) + " .pdf"
    # fignamepdf = "plots/testing/LCU_energy Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(Bzlist_target) \
    #       + "backend_name="+backend_name+" .pdf"
    fignamepdf = "plots/LCU/" + tagpdf
    # fig.savefig(fignamepdf)
    plt.show()

    fig, ax, Bzlist, mag_stor = show_XY_magnetization(N=N, Bzmin=0, Bzmax=2, Bx=Bx, J=J, pbc=pbc)
    for b in Bzlist_training:
        ax.axvline(b)
    for j in range(2 ** N):
        ax.plot(Bzlist, mag_stor[:, j], 'k-')
    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            ax.plot(Bzlist_target, np.real(EVcontinuer.target_magnetization[:, ip]), 'o', color="b")
    if 'ax' in locals():
        ax.plot(Bzlist_target, np.real(LCU_m_list_qsearch), 's', color="red")
    tagpdf = tagpdf.replace("LCU_energy", "LCU_mag")
    fignamepdf = "plots/LCU/" + tagpdf
    fig.savefig(fignamepdf)
    plt.show()
#     print(fignamepdf)
