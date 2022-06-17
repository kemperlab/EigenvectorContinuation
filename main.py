import numpy as np
import matplotlib.pyplot as plt
from continuers import *
from qiskit.opflow import I, X, Y, Z

from quantum_circuit_mimic import *
# from qasm_bundle_circuit import *
from qsearch_bundle_circuit import *
from qasm_bundle_circuit import *
from qsearch_bundle_circuit import *
# from qasm_separate_cricuit import *
from hamiltonian import *
# from reconstruct import *
from qsearch_LCU import *
from h2molecule import *
from functions_in_main import *
# paulis = {}
# paulis['X'] = np.array([[0,1],[1,0]],dtype=complex)
# paulis['Y'] = np.array([[0,-1.j],[1.j,0]],dtype=complex)
# paulis['Z'] = np.array([[1,0],[0,-1]],dtype=complex)
# paulis['I'] = np.array([[1,0],[0,1]],dtype=complex)

# def many_kron(ops):
#     # Takes an array of Pauli characters and produces the tensor product
#     op = paulis[ops[0]]
#     if len(ops) == 1:
#         return op
#
#     for opj in ops[1:]:
#         op = np.kron(op,paulis[opj])
#
#     return op


# def XY_hamiltonian(J, Bx, Bz, N, pbc):


#     ham = np.zeros([2**N,2**N],dtype=complex)

#     # Build hamiltonian matrix
#     for isite in range(N):

#         # BZ
#         oplist = ['I']*N
#         oplist[isite] = 'Z'
#         #print("".join(oplist))
#         ham += Bz*many_kron(oplist)

#         # BX
#         oplist = ['I']*N
#         oplist[isite] = 'X'
#         #print("".join(oplist))
#         ham += Bx*many_kron(oplist)

#         jsite = (isite + 1) % N
#         if not(jsite == isite+1) and not pbc:
#             continue

#         # XX
#         oplist = ['I']*N
#         oplist[isite] = 'X'
#         oplist[jsite] = 'X'
#         #print("".join(oplist))
#         ham += J*many_kron(oplist)

#         # YY
#         oplist = ['I']*N
#         oplist[isite] = 'Y'
#         oplist[jsite] = 'Y'
#         #print("".join(oplist))
#         ham += J*many_kron(oplist)

#     return ham

# # def XY_hamiltonian_Qiskit(J, Bx, Bz, N, pbc):
# #     assert(N==2)
# #     hamiltonian = J*((X^X) + (Y^Y)) + Bz*((I^Z) + (Z^I)) + Bx*((I^X) + (X^I))
# #     return hamiltonian

# def show_XY_spectrum(N,Bzmin,Bzmax,Bx,J,pbc):

#     Bzlist = np.linspace(Bzmin,Bzmax,100)
#     eval_stor = np.zeros([len(Bzlist),2**N])
#     for iBz, Bz in enumerate(Bzlist):
#         ham = XY_hamiltonian(J=J,Bz=Bz,Bx=Bx,N=N,pbc=pbc)
#         eval_stor[iBz,:] = np.linalg.eigvalsh(ham)

#     fig, ax = plt.subplots()
#     ax.set_xlabel("$B_z$")
#     ax.set_ylabel("Energy")
#     for j in range(2**N):
#         ax.plot(Bzlist,eval_stor[:,j],'k-')
#     return fig, ax

# def show_XY_magnetization(N,Bzmin,Bzmax,Bx,J,pbc):

#     Bzlist = np.linspace(Bzmin,Bzmax,100)
#     mag_stor = np.zeros([len(Bzlist),2**N])
#     # mztot_oplist = ['Z'] * N
#     # mztot_op = many_kron(mztot_oplist)
#     mztot_op =  Mag_op(N=N)
#     for iBz, Bz in enumerate(Bzlist):
#         ham = XY_hamiltonian(J=J,Bz=Bz,Bx=Bx,N=N,pbc=pbc)
#         evals, evecs = np.linalg.eigh(ham)
#         mzarray = np.zeros(2**N)
#         for i in range(2**N):
#             mz = np.real(np.conj(np.transpose(evecs[:,i])) @ mztot_op @ evecs[:,i])
#             mzarray[i] = mz
#         mag_stor[iBz,:] = mzarray

#     fig, ax = plt.subplots()
#     ax.set_xlabel("$B_z$")
#     ax.set_ylabel("Magnetization")
#     for j in range(2**N):
#         ax.plot(Bzlist,mag_stor[:,j],'k-')
#     return fig, ax
# #
# # def dot_vectors(A,B):
# #     return np.conjugate(np.transpose(A)) @ B
# #
# # def evaluate_op_vectors(A,B,C):
# #     return np.conjugate(np.transpose(A)) @ B @ C
# #
# def Mag_op(N):

#     Mag = np.zeros([2**N,2**N],dtype=complex)

#     # Build hamiltonian matrix
#     for isite in range(N):

#         # BZ
#         oplist = ['I']*N
#         oplist[isite] = 'Z'
#         #print("".join(oplist))
#         Mag +=many_kron(oplist)

#     return Mag

# def characterize_eigenspectrum(J=-1,Bz=0,Bx=0,N=8,pbc=True):

#     ham = XY_hamiltonian(J,Bz,Bx,N,False)

#     bcterm = ['Z']*N
#     bcterm1 = bcterm
#     bcterm1[0] = 'X'
#     bcterm1[-1] = 'X'
#     print(bcterm1)

#     bcterm2 = bcterm
#     bcterm1[0] = 'Y'
#     bcterm1[-1] = 'Y'
#     print(bcterm2)

#     ham += many_kron(bcterm1)
#     ham += many_kron(bcterm2)

#     evals, evecs = np.linalg.eigh(ham)

#     mztot_oplist = ['Z']*N
#     mztot_op = many_kron(mztot_oplist)

#     for k in range(2**N):
#         energy = evals[k]
#         evec = evecs[:,k]
#         mz = np.real( np.conj(np.transpose(evec)) @ mztot_op @ evec )
#         if abs(mz) < 0.01:
#             print(mz,energy)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # for XXZ model

    # XXZfninmain()
    # exit()
    #####################
    ############################################3
    # for h2 molecule

    # H2fninmain()
    # exit()

    XYinQCinmain()
    exit()
    ####################### INPUT parameters ############################
    ## XY model

    J = -1
    # Bx = 0.5
    Bx = 0.1
    N = 2
    pbc = False

    fig, ax, Bzlist, eval_stor = show_XY_spectrum(N=N, Bzmin=0, Bzmax=2, Bx=Bx,J=J,pbc=pbc)

    # Set up training parameter sets for eigenvector continuer
    # Bzlist = [0,0.2,0.75]
    # Bzlist_training = [0,1.9]
    # Bzlist_training = [0.5, 1.9]
    Bzlist_training = [0.1, 1.3]
    # Bzlist_training = [0.1, 1.9]
    # Bzlist_training = [0, 0.1,0.2]
    # Bzlist_training = [0, 0.8]
    training_paramlist = [[J,Bx,Bz,N,pbc] for Bz in Bzlist_training]
    # _qc for passing parameters to the quantum circuit, because I prefer dictionaries than lists
    training_paramlist_qc = [{"J":J,"Bx":Bx,"Bz":Bz,"N":N,"pbc":pbc} for Bz in Bzlist_training]
    
    if 'ax' in locals():
        for b in Bzlist_training:
            ax.axvline(b)

    # Set up target parameter sets for eigenvector continuer
    # Bzlist = np.linspace(0,2,20)
    # Bzlist_target = [0.7,1.8]

    # Bzlist_target = [1.5,1.7,1.9]
    # Bzlist_target = [0.3, 0.5, 0.7, 1.1, 1.3, 1.5, 1.7]
    Bzlist_target = [0.3, 0.5, 0.7, 1.1, 1.5, 1.7, 1.9]

    target_paramlist = [[J,Bx,Bz,N,pbc] for Bz in Bzlist_target]
    target_paramlist_qc = [{"J":J,"Bx":Bx,"Bz":Bz,"N":N,"pbc":pbc} for Bz in Bzlist_target]
    #################### INPUT parameters over ######################################

    # Object that knows how to deal with the various operations needed
    vectorspace = vector_methods(XY_hamiltonian)

    # Reference vector is internal for now
    #vectorspace = unitary_methods(N, XY_hamiltonian)

    #vectorspace = circuit_methods(N,XY_hamiltonian_Qiskit)

    EVcontinuer = vector_continuer(vectorspace,
                                   XY_hamiltonian,
                                   Mag_op,
                                   training_paramlist,
                                   target_paramlist,
                                   N)

    EVcontinuer.get_base_eigenvectors()
    #EVcontinuer.form_orthogonal_basis()

    #added_evals = EVcontinuer.get_target_eigenvectors(ortho=True)
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
            ax.plot(Bzlist_target,np.real(EVcontinuer.target_evals[:,ip]),'o',color="b")
    #         # ax.plot(Bzlist_target,np.real(mimic_evals[:,ip]),'*',color="r")
    #         ax.plot(Bzlist_target, np.real(reconstruct_qsearch_circuit_evals_bundle[:, ip]), '^', color="g")
    #         # ax.plot(Bzlist_target, np.real(qasm_circuit_evals_separate[:, ip]), 's', color="orange")
    #         # ax.plot(Bzlist_target, np.real(qasm_circuit_evals_bundle[:, ip]), '^', color="k")
    #         ax.plot(Bzlist_target, np.real(qsearch_circuit_evals_bundle[:, ip]), '^', color="k")
        ####################

    ###########################
    
    if(date_flag):
        import datetime
        x = datetime.datetime.now()
        date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J)+"pbc=" + str(pbc) + "Bztrain" + str(Bzlist_training) + \
              "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) + "date" + date + ".pdf"
    else:
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J)+"pbc=" + str(pbc) + "Bztrain" + str(Bzlist_training) + \
              "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) + ".pdf"
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

    LCU_E_list_qsearch,LCU_m_list_qsearch = get_energy_qsearch(training_paramlist = training_paramlist_qc, target_paramlist = target_paramlist_qc, LCU_coeffs_gs_list = EVcontinuer.LCU_coeffs_list, basis_vecs = EVcontinuer.base_vecs, backend_name = backend_name, layout = layout, shots = shots, Basis_exact_flag=True, date_flag = date_flag)
    # print("LCU_Evals")
    # print(LCU_E_list_qsearch)
    fig,ax = plt.subplots()
    for b in Bzlist_training:
        ax.axvline(b)
    for j in range(2**N):
        ax.plot(Bzlist,eval_stor[:,j],'k-')
    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            ax.plot(Bzlist_target,np.real(EVcontinuer.target_evals[:,ip]),'o',color="b")
    if 'ax' in locals():
        ax.plot(Bzlist_target,np.real(LCU_E_list_qsearch),'s',color="red")
            # ax.plot(Bzlist_target, np.real(qsearch_circuit_evals_bundle[:, ip]), '^', color="k")
    ax.set_xlabel("$B_z$")
    ax.set_ylabel("Energy")
    if (date_flag):
        import datetime

        x = datetime.datetime.now()
        date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
        tagpdf = "LCU_energy Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
            Bzlist_training) + "Bztarget" + str(
            Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) + "date_" + date + " .pdf"
    else:
        tagpdf = "LCU_energy Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
            Bzlist_training) + "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) + " .pdf"
    # fignamepdf = "plots/testing/LCU_energy Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(Bzlist_target) \
    #       + "backend_name="+backend_name+" .pdf"
    fignamepdf = "plots/LCU/" + tagpdf
    # fig.savefig(fignamepdf)
    plt.show()

    fig, ax, Bzlist, mag_stor = show_XY_magnetization(N=N, Bzmin=0, Bzmax=2, Bx=Bx, J=J, pbc=pbc)
    for b in Bzlist_training:
            ax.axvline(b)
    for j in range(2**N):
        ax.plot(Bzlist,mag_stor[:,j],'k-')
    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            ax.plot(Bzlist_target,np.real(EVcontinuer.target_magnetization[:,ip]),'o',color="b")
    if 'ax' in locals():
        ax.plot(Bzlist_target,np.real(LCU_m_list_qsearch),'s',color="red")
    tagpdf = tagpdf.replace( "LCU_energy","LCU_mag")
    fignamepdf = "plots/LCU/" + tagpdf
    fig.savefig(fignamepdf)
    plt.show()
#     print(fignamepdf)
