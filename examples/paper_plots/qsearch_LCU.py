# The aim of this notebook is to handle making a target state using LCU and measure its energy given the information
# of traininng sets and target coefficient after diagonalising smaller Hmailtonian.

import pickle

import numpy as np
from qiskit import Aer, execute, QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, SLSQP
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.opflow import I, X, Y, Z

from hamiltonian import *
from scipy import linalg
from scipy.linalg import null_space
from qiskit.extensions import UnitaryGate
import qiskit as qk
from qiskit import IBMQ
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter

# lets assume at the moment that training vectors are given as Uilist and target coefficients are passed
# given the two basis circuits this returns the dot product circuit with control qubit

def convert_U_to_qsearch_circuit(U,circname="circ",project_dir = "LCU_project_dir"):
    # print("Ufeeded in")
    # print(U)
    U = np.array(U)
    # print("Ufeeded in")
    # print(U)
    import search_compiler as sc
    project = sc.Project(project_dir)
    project.add_compilation( circname, U)
    project.run()
    circ_str = project.assemble( circname)
    # print(U)
    # print(circ_str)
    # die()
    circ = QuantumCircuit.from_qasm_str(circ_str)
    # print("Ufeeded out:")
    # backend = Aer.get_backend('unitary_simulator')
    # qc = QuantumCircuit(3)
    # qc.compose(circ, inplace=True)
    # # print(qc.decompose().draw())
    # job = execute(qc, backend=backend)
    # result = job.result()
    #
    # Uout = result.get_unitary(qc)
    # print(Uout)

    return circ
#
# def get_evals_gs_target_ham(Uilist,paramn):
#     J = paramn["J"]
#     Bx = paramn["Bx"]
#     Bz = paramn["Bz"]
#     N = paramn["N"]
#     pbc = paramn["pbc"]
#     print(paramn)
#     overlap_matrix = create_overlap_matrix_fromQC(Uilist=Uilist,N=N)
#     print("Overlap matrix mimic:\n" , overlap_matrix)
#     smaller_ham = make_target_hamiltonian_fromQC(Uilist,paramn)
#     #print(smaller_ham)
#     print("Hamiltonian mimic:\n" ,smaller_ham)
#     evals, evecs = linalg.eigh(smaller_ham,overlap_matrix)
#     print("Evals circuit mimic: ",evals)
#     #print("GS circuit mimic: ", evecs[:,0])
#     return evals,evecs[:, 0]
###############################
#
# def get_LCU_gs_list_mimic_exact(training_paramlist,target_paramlist, basis_vecs, Basis_exact_flag=True):
#     gs_LCU = np.zeros([len(target_paramlist),2**2],dtype=complex)
#
#     if(Basis_exact_flag==True):
#         Uilist = get_training_vectors_exact(basis_vecs = basis_vecs)
#     else:
#         Uilist = get_training_vectors(training_paramlist = training_paramlist)
#     # Uilist = get_training_vectors(training_paramlist)
#     print("printing Uilist from mimic")
#     for u in Uilist:
#         print(u[:,0])
#     for ip,paramn in enumerate(target_paramlist):
#         evals,gsc = get_evals_gs_target_ham(Uilist,paramn)
#         print("gsc\n",gsc)
#         gs_temp = np.zeros(2 ** 2, dtype=complex)
#         for k in range(len(training_paramlist)):
#             Ui = Uilist[k]
#             print(np.shape(Ui))
#             gs_temp += gsc[k]*Ui[:,0]
#         gs_LCU[ip,:] = gs_temp
#
#     return gs_LCU
#####################
def make_cntrlU(U):
    L = len(U)
    cntrlU = np.identity(2*L,dtype=complex)
    for i in range(L):
        for j in range(L):
            cntrlU[L+i,L+j] = U[i,j]
    return cntrlU
################
def make_open_cntrlU(U):
    L = len(U)
    opncntrlU = np.identity(2*L,dtype=complex)
    for i in range(L):
        for j in range(L):
            opncntrlU[i,j] = U[i,j]
    return opncntrlU
#####################
def get_LCU_unitary(gs,Uilist,N=2):
    import cmath
    rs = np.zeros(len(gs))
    phases = np.zeros(len(gs))
    for i in range(len(gs)):
        rs[i] = abs(gs[i])
        phases[i]  = cmath.phase(gs[i])

    k = rs[0]/rs[1]
    Vk = (1/np.sqrt(k+1))*np.matrix([[np.sqrt(k), -1],[1,np.sqrt(k)]],dtype = "complex")
    # psi_an = np.zeros([2], dtype="complex")
    # psi_an[0] = 1.0
    #
    # psi_sys = np.zeros([2 ** (N)], dtype="complex")
    # psi_sys[0] = 1.0

    #     step 0
    # psi = np.kron(psi_an, psi_sys)
    Ua = np.exp(1.j* phases[0])*Uilist[0]
    Ub = np.exp(1.j* phases[1])*Uilist[1]
    Ustep0 = np.kron(Vk,np.identity(2**N))
    Ustep1 = make_open_cntrlU(U = Ua)
    Ustep2 = make_cntrlU(U = Ub)
    Ustep3 = np.kron(np.conjugate(np.transpose(Vk)),np.identity(N**2))

    # print(np.shape(Ustep0))
    # print(np.shape(psi))
    Utot = np.matmul(Ustep1,Ustep0)
    Utot = np.matmul(Ustep2, Utot)
    Utot = np.matmul(Ustep3,Utot)


    return Utot
#############################
def get_backend(backend_name="qasm_simulator"):
    if(backend_name=="qasm_simulator"):
        backend = Aer.get_backend('qasm_simulator')
    else:
        if IBMQ.active_account() is None:
            provider = IBMQ.load_account()
#         provider  = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        provider  = IBMQ.get_provider(hub='ibm-q-ncsu', group='nc-state', project='physics-of-spin-')
        backend = provider.get_backend(backend_name)

    return backend
####################################
def make_LCU_circuit(target_gs_coeffs, Uilist):
    r0 = abs(target_gs_coeffs[0])
    r1 = abs(target_gs_coeffs[1])

    if(r0 < 10**-8):
        print("Please give target points different from training points")
    elif (r1 < 10 ** -8):
        print("Please give target points different from training points")
    else:
        LCU_qc = get_LCU_unitary(gs=target_gs_coeffs,Uilist=Uilist,N=2)

    return LCU_qc
#######################
def get_training_vectors_exact(basis_vecs):
    Uilist=[]
    for basis_v in basis_vecs:
        Ui = makeUnitaryfromvec(vec=basis_v)
        Uilist.append(Ui)
    return Uilist
###############################
def makeUnitaryfromvec(vec):
    length = len(vec)
    ket0 = np.zeros([length], dtype="complex")
    ket0[0] = 1.0
    ket0 = np.reshape(ket0, (length, 1))
    bravec = np.reshape(vec, (1, length))
    Unitaryvec = np.matmul(ket0, bravec)
    nullspace = null_space(Unitaryvec)
    U_vec = np.zeros((length, length), dtype="complex")
    U_vec[:, 0] = vec
    for i in range(length - 1):
        U_vec[:, i + 1] = nullspace[:, i]
    return U_vec
#####################################
def make_measure_inbasis_circ(circ, N=2, basis="X"):
    q = QuantumRegister(N+1)
    c = ClassicalRegister(N+1)
    qc = QuantumCircuit(q, c)
    # print(circ)
    qc.compose(circ, inplace=True)
    # qc.compose(circ, inplace=False)
    if (basis == "X"):
        for i in range(N):
            qc.h(q[i+1])
    elif (basis == "Y"):
        for i in range(N):
            qc.rx(-np.pi / 2, q[i+1])

    for i in range(N+1):
        qc.measure(q[i], c[i])
    return qc
def make_target_LCU_qc(target_paramlist,LCU_coeffs_gs_list, basis_vecs, project_dir, Basis_exact_flag=True):
    if (Basis_exact_flag == True):
        Uilist = get_training_vectors_exact(basis_vecs=basis_vecs)
    else:
        print("will update later")
        # Uilist = get_training_vectors(training_paramlist=training_paramlist)
    # print("Uilist = \n",Uilist)
    LCU_qc_list = []
    N=2
    for ip,paramn in enumerate(target_paramlist):
        gs =  LCU_coeffs_gs_list[ip]
        # print("gs coefficients for target: ",gs)
        rs = np.zeros(len(gs))
        for i in range(len(gs)):
            rs[i] = abs(gs[i])
        LCU_U = make_LCU_circuit(target_gs_coeffs = gs, Uilist=Uilist)
        # print("LCU U shape: ",np.shape(LCU_U))
        project_dir_wtarg = project_dir +"Bztarg" + str(paramn["Bz"])
        LCU_qc = convert_U_to_qsearch_circuit(U = LCU_U, circname="LCUcircZbasis", project_dir=project_dir_wtarg)

        LCU_qc = make_measure_inbasis_circ(circ = LCU_qc, N=2, basis="Z")
        LCU_qc_list.append(LCU_qc)

    return LCU_qc_list

def get_mval_from_counts(counts):
    # print(counts)
    keys = list(counts.keys())
    # print(keys)
    L=2
    mval = 0.0
    mshots = 0.0
    for key in keys:
        if(key[L]=='0'):
            # print(key)
            mi=0.0
            for i in range(L):
                # t = key[i+1]
                t = key[i]
                mi += int(t)
            mi = -2 * mi + L
            # print(mi)
            mval+= counts[key]*mi
            mshots += counts[key]

    mval = mval/mshots

    return mval
###########################

def get_expectation_fromPauli(counts, oplist):
    #     note there is an extra ancilla qubit in counts
    keys = list(counts.keys())
    # print(keys)
    L = 2
    Expval = 0.0
    Expshots = 0.0
    for key in keys:
        if (key[L] == '0'):
            # print(key)
            valperkey = 1.0
            for i in range(L):
                val = 1
                if (oplist[i] != 'I'):
                    if (key[i] == '1'):
                        val = -1
                # print(val)
                valperkey *= val

            # print(counts[key])
            # print(valperkey)
            Expval += counts[key] * valperkey
            Expshots += counts[key]

    Expval = Expval / Expshots

    return Expval

def get_Energy_fromcounts(countsallbasis,Bx,Bz,J):
    countsZ = countsallbasis[0]
    countsX = countsallbasis[1]
    countsY = countsallbasis[2]

    Jxxcomp = get_expectation_fromPauli(counts=countsX, oplist=['X', 'X'])
    Jyycomp = get_expectation_fromPauli(counts=countsY, oplist=['Y', 'Y'])
    Bxcomp =  get_expectation_fromPauli(counts=countsX, oplist=['X', 'I']) + get_expectation_fromPauli(counts=countsX, oplist=['I', 'X'])
    Bzcomp = get_expectation_fromPauli(counts=countsZ, oplist=['Z', 'I']) + get_expectation_fromPauli(counts=countsZ, oplist=['I', 'Z'])

    # print("Jxx = ",Jxxcomp)
    # print("Jyy = ", Jyycomp)
    # print("Jxx = ", Jxxcomp)
    # print("Jyy = ", Jyycomp)
    # print("Bx = ", Bxcomp)
    # print("Bz = ", Bzcomp)

    E = J*(Jxxcomp+Jyycomp) + Bx*Bxcomp + Bz*Bzcomp
    return E
#############################
def get_mag_qsearch(training_paramlist,target_paramlist, LCU_coeffs_gs_list, basis_vecs,backend_name,layout,shots, Basis_exact_flag=True):
    paramn = target_paramlist[0]
    N = paramn["N"]
    pbc = paramn["pbc"]
    Bx = paramn["Bx"]
    Bzlist_target = []
    for paramn in target_paramlist:
        Bzlist_target.append(paramn["Bz"])
    #
    Bzlist_training = []
    for paramn in training_paramlist:
        Bzlist_training.append(paramn["Bz"])

    project_dir = "LCU_project_dir/Zmeas_" + str(N) + "site" + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training)
    LCU_qc_list = make_target_LCU_qc(target_paramlist=target_paramlist, LCU_coeffs_gs_list = LCU_coeffs_gs_list,\
                                     basis_vecs=basis_vecs, project_dir=project_dir, Basis_exact_flag=Basis_exact_flag)

    # filename = "circuits/LCU/sc_LCU_trans" + "Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
    #     Bzlist_training) + "Bztarget" + str(Bzlist_target) \
    #            + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel) + ".p"
    backend = get_backend(backend_name=backend_name)
    optimizationlevel = 3
    transpiled_bundlelist = qk.transpile(circuits=LCU_qc_list, backend=backend, initial_layout=layout,
                                         optimization_level=optimizationlevel)

    # pickle.dump(transpiled_bundlelist, open(filename, "wb"))
    # print( transpiled_bundlelist)
    # print( transpiled_bundlelist[0])

    qobj = qk.assemble(transpiled_bundlelist, backend=backend, shots=shots)
    # print("qobjects created")
    job = backend.run(qobj)
    results = job.result()
    # print(results)
    # allresults = results.results
    allcounts = results.get_counts()
    # print("Does it run??")
    # print(allcounts)
    m_vals = []
    for i in range(len(transpiled_bundlelist)):
        if(len(transpiled_bundlelist)==1):
            mval = get_mval_from_counts(counts=allcounts)
        else:
            mval = get_mval_from_counts(counts = allcounts[i])
        # p = get_p_from_counts(counts=allcounts[i])
        m_vals.append(mval)
    # m_vals = np.zeros_like(len(target_paramlist))
    return m_vals
#######################

def make_target_LCU_withrot_qc(paramn,LCU_coeffs_gs ,basis_vecs, project_dir, meas_basis="Z",Basis_exact_flag=True):
    if (Basis_exact_flag == True):
        Uilist = get_training_vectors_exact(basis_vecs=basis_vecs)
    else:
        print("will update later")
        # Uilist = get_training_vectors(training_paramlist=training_paramlist)
    # print("Uilist = \n",Uilist)
    # LCU_qc_list = []
    basis_filename = project_dir
    basis_filename = basis_filename.replace("qsearch_dir/LCU_project_dir/"+ meas_basis ,"results/LCU/Basis/basis_uniatris")
    print(basis_filename)
    pickle.dump(Uilist, open(basis_filename, "wb"))
    N=2

    gs =  LCU_coeffs_gs
        # print("gs coefficients for target: ",gs)
    rs = np.zeros(len(gs))
    for i in range(len(gs)):
        rs[i] = abs(gs[i])
    LCU_U = make_LCU_circuit(target_gs_coeffs = gs, Uilist=Uilist)
    # print("LCU U shape: ",np.shape(LCU_U))
    project_dir_wtarg = project_dir +"Bztarg" + str(paramn["Bz"])
    circname = "LCUcirc" + meas_basis + "basis"
    LCU_qc = convert_U_to_qsearch_circuit(U = LCU_U, circname = circname, project_dir=project_dir_wtarg)

    LCU_qc = make_measure_inbasis_circ(circ = LCU_qc, N=2, basis = meas_basis)
    # LCU_qc_list.append(LCU_qc)

    return LCU_qc

##################
def get_energy_qsearch(training_paramlist,target_paramlist, LCU_coeffs_gs_list, basis_vecs,backend_name,layout,shots, Basis_exact_flag=True,date_flag = True):
    ###########        initialization      ##############

    paramn = target_paramlist[0]
    N = paramn["N"]
    pbc = paramn["pbc"]
    J = paramn["J"]
    Bx = paramn["Bx"]
    Bzlist_target = []
    for paramn in target_paramlist:
        Bzlist_target.append(paramn["Bz"])
    #
    Bzlist_training = []
    for paramn in training_paramlist:
        Bzlist_training.append(paramn["Bz"])

    backend = get_backend(backend_name=backend_name)
    optimizationlevel = 3
    metadata = [N,J,pbc,Bx,Bzlist_training,Bzlist_target ]
    ############ setting up search compiler ############
    LCU_qc_list = []
    tag = str(N) + "site" + "Bx=" + str(Bx) +"J=" + str(J)+"pbc=" + str(pbc) + "Bztrain" + str(Bzlist_training)

    project_dirZ = "qsearch_dir/LCU_project_dir/Zmeas_" + tag
    project_dirX = "qsearch_dir/LCU_project_dir/Xmeas_" + tag
    project_dirY = "qsearch_dir/LCU_project_dir/Ymeas_" + tag
    #
    # project_dirZ = "LCU_project_dir/Zmeas_" + str(N) + "site" + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training)
    # project_dirX = "LCU_project_dir/Xmeas_" + str(N) + "site" + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training)
    # project_dirY = "LCU_project_dir/Ymeas_" + str(N) + "site" + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training)

    for i in range(len(target_paramlist)):


        LCU_qc_Z = make_target_LCU_withrot_qc(paramn=target_paramlist[i], LCU_coeffs_gs = LCU_coeffs_gs_list[i],\
                                         basis_vecs=basis_vecs, project_dir=project_dirZ,meas_basis="Z", Basis_exact_flag=Basis_exact_flag)
        LCU_qc_list.append(LCU_qc_Z)
        LCU_qc_X = make_target_LCU_withrot_qc(paramn=target_paramlist[i], LCU_coeffs_gs = LCU_coeffs_gs_list[i], \
                                         basis_vecs=basis_vecs, project_dir=project_dirX,meas_basis="X", Basis_exact_flag=Basis_exact_flag)
        LCU_qc_list.append(LCU_qc_X)
        LCU_qc_Y = make_target_LCU_withrot_qc(paramn=target_paramlist[i],  LCU_coeffs_gs = LCU_coeffs_gs_list[i], \
                                         basis_vecs=basis_vecs, project_dir=project_dirY, meas_basis="Y",
                                         Basis_exact_flag=Basis_exact_flag)
        LCU_qc_list.append(LCU_qc_Y)

    ######## setting up the machine run ###############

    if(date_flag):
        import datetime
        x = datetime.datetime.now()
        date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J)+"pbc=" + str(pbc) + "Bztrain" + str(Bzlist_training) + \
              "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) +"opt=" + \
              str(optimizationlevel) + "date" + date + ".p"
    else:
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J)+"pbc=" + str(pbc) + "Bztrain" + str(
            Bzlist_training) + "Bztarget" + str(Bzlist_target) \
                   + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) + "opt=" + str(optimizationlevel) + ".p"
    filename_notrans = "circuits/LCU/sc_LCU_notrans" + tag
    filename_trans = "circuits/LCU/trans/sc_LCU_trans" + tag
    pickle.dump(LCU_qc_list, open(filename_notrans, "wb"))
    print("Transpiling ", backend)
    transpiled_bundlelist = qk.transpile(circuits=LCU_qc_list, backend=backend, initial_layout=layout,
                                         optimization_level=optimizationlevel)
    print("Transpiledlcu")
    pickle.dump(transpiled_bundlelist, open(filename_trans, "wb"))
    # print( transpiled_bundlelist)
    print( transpiled_bundlelist[0])

    qobj = qk.assemble(transpiled_bundlelist, backend=backend, shots=shots)
    # print("qobjects created")
    job = backend.run(qobj)
    results = job.result()
    print("jobid lcu: ", job.job_id())
    # print(results)
    # allresults = results.results
    allcounts = results.get_counts()
   ############measurement error correction##########
    qr = QuantumRegister(3)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    # make sure it is layout[0]
    t_qc_calibs = qk.transpile(meas_calibs, backend=backend, initial_layout=layout)
    qobj_calibs = qk.assemble(t_qc_calibs, backend=backend, shots=shots)
    job_calibs = backend.run(qobj_calibs)
    calib_results = job_calibs.result()
    print("jobid lcu calib: ", job_calibs.job_id())
    meas_fitter = CompleteMeasFitter(calib_results, state_labels, circlabel='mcal')
    meas_filter = meas_fitter.filter
    print("calibration matrix : ")
    print(meas_fitter.cal_matrix)
    mitigated_results = meas_filter.apply(results)
    mitigated_counts = mitigated_results.get_counts()
    ################# Lets save the results and counts ##################
    filename_results_together = "results/LCU/results_counts" + tag
    pickle.dump([results, calib_results, mitigated_results, allcounts, meas_fitter.cal_matrix, mitigated_counts,metadata],
                open(filename_results_together, "wb"))
    filename_results = "results/LCU/results" + tag
    pickle.dump([results, calib_results, mitigated_results,metadata],
                open(filename_results, "wb"))
    filename_counts = "results/LCU/counts" + tag
    pickle.dump([allcounts, meas_fitter.cal_matrix, mitigated_counts,metadata],
                open(filename_counts, "wb"))
    # if (date_flag):
    #     import datetime
    #     x = datetime.datetime.now()
    #     date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
    #     filename = "results/LCU/results_counts" + "Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
    #         Bzlist_training) + "Bztarget" + str(Bzlist_target) \
    #                + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(
    #         optimizationlevel) + "date" + date + ".p"
    # else:
    #     filename = "results/LCU/results_counts" + "Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
    #         Bzlist_training) + "Bztarget" + str(Bzlist_target) \
    #                + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel) + ".p"
    # pickle.dump([results,calib_results,mitigated_results,allcounts,meas_fitter.cal_matrix,mitigated_counts,], open(filename, "wb"))
    # now lets get the Energy components from measurement

    Energylist_raw = []
    maglist_raw = []
    for i in range(len(target_paramlist)):
        countsthree = allcounts[i*3:i*3+3]
        E = get_Energy_fromcounts(countsallbasis=countsthree,Bx=Bx,Bz=Bzlist_target[i],J=J)
        Energylist_raw.append(E)
        m = get_mval_from_counts(counts = allcounts[i*3])
        maglist_raw.append(m)
    Energylist = []
    maglist = []
    for i in range(len(target_paramlist)):
        countsthree = mitigated_counts[i*3:i*3+3]
        E = get_Energy_fromcounts(countsallbasis=countsthree, Bx=Bx, Bz=Bzlist_target[i], J=J)
        Energylist.append(E)
        m = get_mval_from_counts(counts=allcounts[i * 3])
        maglist.append(m)

    tag = tag.replace(".p", ".txt")
    filename_summary = "summary/LCU_eigen_values_"+tag
    # filename = "summary/LCU_eigen_values_Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
    #     Bzlist_training) + "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel)".txt"
    f = open(filename_summary, "w")
    f.write("Energylist raw : "+str(Energylist_raw))
    f.write("\n Energylist mitigated : " + str(Energylist))
    f.write("\n Magnetizationlist raw : " + str(maglist_raw))
    f.write("\n Magnetizationlist mitigated : " + str(maglist))
    f.close()

    return Energylist, maglist
