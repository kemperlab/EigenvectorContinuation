import pickle

from qiskit import Aer, execute, QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, SLSQP
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.opflow import I, X, Y, Z

from hamiltonian import *
from scipy import linalg

from qiskit.extensions import UnitaryGate
import qiskit as qk
from qiskit import IBMQ
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter

###################################
# given the parameters this function gets the VQE circuit and Unitary
def get_circuit_unitary_fromVQE(param):
    
    J = param["J"]
    Bx = param["Bx"]
    Bz= param["Bz"]
    pbc =param["pbc"]
    N = param["N"]
    
    # this hamiltonian needs to be written in using pauli strings later
    ham = XY_hamiltonian(J, Bx, Bz, N, pbc)
    hamiltonian = J*((X^X) + (Y^Y)) + Bz*((I^Z) + (Z^I)) + Bx*((I^X) + (X^I))
    # print(ham)
    # print(hamiltonian)
    # print(type(hamiltonian))
    initial = [ 4.13850621,  4.68105177,  0.10459295,  5.19511699, -3.60948918, 2.08480452, -3.83679895,  1.9264198 ]
    #initial = None
    
    seed = 50
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

    ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
    slsqp = SLSQP(maxiter=1000)
    spsa = SPSA(maxiter=1000)
    vqe = VQE(ansatz, optimizer=spsa, quantum_instance=qi, initial_point=initial)
    VQEresult = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    # print("VQE found eigenvalue:",VQEresult.eigenvalue)
    evals, evecs = linalg.eigh(ham)
    # print("I expected:",evals[0])

    cc = vqe.get_optimal_circuit()
    
    backend = Aer.get_backend('unitary_simulator')
    qc = QuantumCircuit(2)
    qc.compose(cc,inplace=True)
    #print(qc.decompose().draw())    
    job = execute(qc, backend=backend)
    result = job.result()
    
    U = result.get_unitary(qc)

    return cc,U
###############################
# given the training parameter list this function returns the basis circuit list from vqe
def get_circuit_list(training_paramlist):
    basis_circuits_list = []
    for param in training_paramlist:
        circuiti,Unitaryi = get_circuit_unitary_fromVQE(param=param)
        basis_circuits_list.append(circuiti)

    return basis_circuits_list

##############################
# This function gets the simulator either Aer qasm or ibmq ones
def get_backend(backend_name="qasm_simulator"):
    if(backend_name=="qasm_simulator"):
        backend = Aer.get_backend('qasm_simulator')

    else:
        if IBMQ.active_account() is None:
            provider = IBMQ.load_account()
        provider  = IBMQ.get_provider(hub='ibm-q-ncsu', group='nc-state', project='physics-of-spin-')
        backend = provider.get_backend(backend_name)

    return backend


######################################################
# given the circ this func attaches the basis rotation gate and returns the circuit
def make_measure_inbasis_circ(circ, N=2, basis="X"):
    q = QuantumRegister(N)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    # print(circ)
    qc.compose(circ, inplace=True)
    if (basis == "X"):
        qc.h(q[0])
    elif (basis == "Y"):
        qc.rx(-np.pi / 2, q[0])
    qc.measure(q[0], c[0])
    return qc
#############################################
# # given the result it returns the average m value
# def get_p_from_result(result):
#
#     counts = result.get_counts()
#
#     try:
#         p = (counts['0'] - counts['1']) / (counts['0'] + counts['1'])
#     except:
#         try:
#             p = counts['0'] / counts['0']
#         except:
#             p = -1
#
#     return p
########################################
#
# given the counts it returns the average m value
def get_p_from_counts(counts):

    try:
        p = (counts['0'] - counts['1']) / (counts['0'] + counts['1'])
    except:
        try:
            p = counts['0'] / counts['0']
        except:
            p = -1

    return p
###########################################
# given the two basis circuits this returns the dot product circuit with control qubit
def phi_ij(circi,circj,N):
    # q = QuantumRegister(N)
    # c = ClassicalRegister(1)
    # qc = deepcopy(circi)
    qc = QuantumCircuit(N)
    # qc.append(circi,[0,1])
    # qc = circi
    # print("circi\n")
    # print(qc.decompose().draw())
    gatei = circi.to_gate()
    qc.append(gatei,[0,1])
    gatej = circj.to_gate()
    gatejinv = gatej.inverse()
    
    # qc.compose(gatejinv,inplace=True)
    qc.append(gatejinv,[0,1])
    # print("circj\n")
    # print(qc.decompose().draw())
    q = QuantumRegister(N+1)
    # c = ClassicalRegister(1)
    qccont= QuantumCircuit(q) 
    # qccont= QuantumCircuit(q,c) 
     
    layout = []
    for i in range(N):
        layout.append(i)
    layout.append(N)
    # print(layout)
    
    qccont.h(q[0])
    gate = qc.to_gate()
    cgate = gate.control()
    qccont.append(cgate,layout)
    # print(qccont.decompose().draw())
    return qccont
########################################
# This takes in the the Bz,Bx,J matrix paramn and returns target hamiltonian
def make_target_hamiltonian_fromQC(Bzmatrix,Bxmatrix,Jmatrix,paramn):
    
    J = paramn["J"]
    Bx = paramn["Bx"]
    Bz = paramn["Bz"]
    N = paramn["N"]
    pbc = paramn["pbc"]
    
    base_length = len(Bxmatrix)
    
    ham_target = np.identity(base_length,dtype="complex")
    for i in range(base_length):
        for j in range(i,base_length):
            hamij = Bz*Bzmatrix[i,j] + Bx*Bxmatrix[i,j] + J*Jmatrix[i,j]
            ham_target[i,j] = hamij
            if(i!=j):
                ham_target[j,i] = np.conjugate(hamij)        
 
    return ham_target
##############################################
# given the overlap_matrix,Bzmatrix,Bxmatrix,Jmatrix ,paramn this sets up the hamiltonian matrix and returns eigvals and vecs
def get_evals_of_target_ham_from_matrices(overlap_matrix,Bzmatrix,Bxmatrix,Jmatrix ,paramn):
    J =paramn["J"]
    Bx = paramn["Bx"]
    Bz = paramn["Bz"]
    N = paramn["N"]
    pbc = paramn["pbc"]
  
    smaller_ham = make_target_hamiltonian_fromQC(Bzmatrix,Bxmatrix,Jmatrix,paramn)
    print("Hamiltonian qasm :\n", smaller_ham)
    evals, evecs = linalg.eigh(smaller_ham,overlap_matrix)
    # evals, evecs = linalg.eigh(smaller_ham,overlap_matrix,driver="gv")
    print("Evals qasm: ",evals)
    return evals,evecs
############################################################
#####################
# given the circi, circj, paulisop this makes the circuit for Paulidot product
def make_Pauli_ij_circuit(circi, circj, paulisop, N=2 ):
    # circuiti,Ui = get_circuit_unitary_fromVQE(param=parami,N=N)
    # circuitj,Uj = get_circuit_unitary_fromVQE(param=paramj,N=N)

    H = many_kron(ops=paulisop)
    # print(  "H Pauli = ",H)

    circ_H = QuantumCircuit(N)
    circ_H.append(UnitaryGate(H), [0, 1])
    gateH = circ_H.to_gate()
    cgateH = gateH.control()

    gatei = circi.to_gate()
    cgatei = gatei.control()

    # gatej = circj.to_gate()
    gatej = circj.to_gate()
    gatejinv = gatej.inverse()
    cgatejinv = gatejinv.control()

    q = QuantumRegister(N + 1)
    circ_Pij = QuantumCircuit(q)
    circ_Pij.h(q[0])
    circ_Pij.append(cgatei, [0, 1, 2])
    circ_Pij.append(cgateH, [0, 1, 2])
    circ_Pij.append(cgatejinv, [0, 1, 2])

    return circ_Pij
#####################################
# given the circi, circj, this makes the circuit for Paulidot product and bundles all of them
def make_Paulibundledcircuits_ij(N, pbc, circi, circj):
    bundled_Pauliij_circuitlist = []
    # Build hamiltonian matrix
    for isite in range(N):

        # BZ
        oplist = ['I'] * N
        oplist[isite] = 'Z'
        # print("".join(oplist))
        circ_Pij = make_Pauli_ij_circuit(circi=circi, circj=circj, paulisop=oplist, N=N )
        circ_Pijx = make_measure_inbasis_circ(circ= circ_Pij, N=N+1, basis="X")
        bundled_Pauliij_circuitlist.append(circ_Pijx)
        circ_Pijy = make_measure_inbasis_circ(circ= circ_Pij, N=N+1, basis="Y")
        bundled_Pauliij_circuitlist.append(circ_Pijy)

        # BX
        oplist = ['I'] * N
        oplist[isite] = 'X'
        # print("".join(oplist))
        circ_Pij = make_Pauli_ij_circuit(circi=circi, circj=circj, paulisop=oplist, N=N)
        circ_Pijx = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="X")
        bundled_Pauliij_circuitlist.append(circ_Pijx)
        circ_Pijy = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="Y")
        bundled_Pauliij_circuitlist.append(circ_Pijy)

        jsite = (isite + 1) % N
        if not (jsite == isite + 1) and not pbc:
            continue

        # XX
        oplist = ['I'] * N
        oplist[isite] = 'X'
        oplist[jsite] = 'X'
        # print("".join(oplist))
        circ_Pij = make_Pauli_ij_circuit(circi=circi, circj=circj, paulisop=oplist, N=N)
        circ_Pijx = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="X")
        bundled_Pauliij_circuitlist.append(circ_Pijx)
        circ_Pijy = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="Y")
        bundled_Pauliij_circuitlist.append(circ_Pijy)
        # YY
        oplist = ['I'] * N
        oplist[isite] = 'Y'
        oplist[jsite] = 'Y'
        # print("".join(oplist))
        circ_Pij = make_Pauli_ij_circuit(circi=circi, circj=circj, paulisop=oplist, N=N)
        circ_Pijx = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="X")
        bundled_Pauliij_circuitlist.append(circ_Pijx)
        circ_Pijy = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="Y")
        bundled_Pauliij_circuitlist.append(circ_Pijy)

    return bundled_Pauliij_circuitlist
#########################################
# given the basis circuit list this bundles circuits for all ij
def make_hamiltonian_circuit_bundlelist(basis_circuits_list, N, pbc):
    base_length = len(basis_circuits_list)

    bundled_Paulifull_circuitlist = []

    for i in range(base_length):
        for j in range(i, base_length):
            bundled_Paulij_circuitlist = make_Paulibundledcircuits_ij(N=N, pbc=pbc, circi=basis_circuits_list[i],circj=basis_circuits_list[j])
            bundled_Paulifull_circuitlist.extend(bundled_Paulij_circuitlist)

    return bundled_Paulifull_circuitlist
##########################
# this unbundles plist to give Bx,Bz,J matrices
def get_hamiltonian_contributions_from_bundled_Paulifull( base_length, N, pbc, parray_Paulifull):

    Bzmatrix = np.zeros((base_length, base_length), dtype="complex")
    Bxmatrix = np.zeros((base_length, base_length), dtype="complex")
    Jmatrix = np.zeros((base_length, base_length), dtype="complex")
    if(pbc):
        length_in_eachij = 2*4*N
    else:
        length_in_eachij = 2 * (2 * N + 2*(N-1))
    index=0
    for i in range(base_length):
        for j in range(i, base_length):
            parray = np.zeros(length_in_eachij,dtype="complex")
            for k in range(length_in_eachij):
                parray[k] = parray_Paulifull[index*length_in_eachij + k]
            index += 1
            [hamBz, hamBx, hamXX, hamYY] = get_ham_pauli_ijcomponents_from_plist(parray,N,pbc)
            Bxmatrix[i, j] = hamBx
            Bzmatrix[i, j] = hamBz
            Jmatrix[i, j] = hamXX + hamYY
            if (i != 0):
                Bxmatrix[j, i] = np.conjugate(Bxmatrix[i, j])
                Bzmatrix[j, i] = np.conjugate(Bzmatrix[i, j])
                Jmatrix[j, i] = np.conjugate(Jmatrix[i, j])

    return Bzmatrix, Bxmatrix, Jmatrix
###################################
# given parray this gives the [hamBz, hamBx, hamXX, hamYY]
def get_ham_pauli_ijcomponents_from_plist(parray,N,pbc):
    hamBz = 0.0 + 0.0 * 1.j
    hamBx = 0.0 + 0.0 * 1.j
    hamXX = 0.0 + 0.0 * 1.j
    hamYY = 0.0 + 0.0 * 1.j

    # Build hamiltonian matrix
    for isite in range(N):

        # BZ
        oplist = ['I'] * N
        oplist[isite] = 'Z'
        # print("".join(oplist))
        hamij = parray[isite*8 +0] + 1.j*parray[isite*8 +1]
        hamBz += hamij

        # BX
        oplist = ['I'] * N
        oplist[isite] = 'X'
        # print("".join(oplist))
        hamij = parray[isite * 8 + 2] + 1.j * parray[isite * 8 + 3]

        hamBx += hamij

        jsite = (isite + 1) % N
        if not (jsite == isite + 1) and not pbc:
            continue

        # XX
        oplist = ['I'] * N
        oplist[isite] = 'X'
        oplist[jsite] = 'X'
        # print("".join(oplist))
        hamij = parray[isite * 8 + 4] + 1.j * parray[isite * 8 + 5]

        hamXX += hamij
        # YY
        oplist = ['I'] * N
        oplist[isite] = 'Y'
        oplist[jsite] = 'Y'
        # print("".join(oplist))
        hamij = parray[isite * 8 + 6] + 1.j * parray[isite * 8 + 7]
        hamYY += hamij
    return [hamBz, hamBx, hamXX, hamYY]
################################
# given the basiscircuitlist this wraps everything for final bundle list
def make_circuit_bundle_forQC(basis_circuits_list , N ,pbc):
#     lets bundle the overlap circuits
    basis_length = len(basis_circuits_list)
    qc_bundlelist=[]
    for i in range (basis_length-1):
        for j in range(i+1,basis_length):
            circij = phi_ij(circi=basis_circuits_list[i],circj=basis_circuits_list[j],N=N)
            phiijx = make_measure_inbasis_circ(circ=circij,N=N+1,basis="X")
            qc_bundlelist.append(phiijx)
            phiijy = make_measure_inbasis_circ(circ=circij, N=N+1, basis="Y")
            qc_bundlelist.append(phiijy)
    # print(qc_bundlelist)

#     lets bundle the pauli circuits
    ham_bundlelist = make_hamiltonian_circuit_bundlelist(basis_circuits_list=basis_circuits_list, N=N, pbc=pbc)
    # print(ham_bundlelist)
    qc_bundlelist.extend(ham_bundlelist)
    # print(qc_bundlelist)
    return qc_bundlelist
#####################################
# this unpacks plist to get all the matrices
def unpackplist(plist,N,pbc,basis_length):
    length_overlap = (basis_length**2) - basis_length
    poverlaplist = plist[:length_overlap]
    overlap_matrix = np.identity(basis_length,dtype="complex")
    index=0
    for i in range(basis_length - 1):
        for j in range(i + 1, basis_length):
            phij = poverlaplist[index*basis_length + 0] + 1.j* poverlaplist[index*basis_length + 1]
            overlap_matrix[i, j] = phij
            overlap_matrix[j, i] = np.conjugate(phij)
    #         lets get the hamiltoninan matrix parts
    phamlist = plist[length_overlap:]
    phamarray = np.array(phamlist)
    Bzmatrix, Bxmatrix, Jmatrix = get_hamiltonian_contributions_from_bundled_Paulifull( base_length=basis_length, N=N, pbc=pbc, parray_Paulifull=phamarray)
    return overlap_matrix,Bzmatrix, Bxmatrix, Jmatrix
#############################################

def get_evals_targetlist_qasmcirc(training_paramlist,target_paramlist,backend_name,layout,shots=8192 ):
    ####################   input QC ##################
    # backend_name = "qasm_simulator"
    # layout = [0,1,2]

    # backend_name = "ibmq_jakarta"
    # layout = [5,3,4]
    # backend_name = "ibmq_manila"
    # layout = [2, 3, 4]
    optimizationlevel = 3
    # shots = 8192
    ########################### input over #######################

    ######################## getting basis circuit list  ########################
    # evals_qc = np.zeros([len(target_paramlist),len(training_paramlist)],dtype=complex)
    basis_circuits_list = get_circuit_list(training_paramlist)

    paramn = target_paramlist[0]
    N = paramn["N"]
    pbc = paramn["pbc"]
    Bx = paramn["Bx"]
    Bzlist_target=[]
    for paramn in target_paramlist:
        Bzlist_target.append(paramn["Bz"])
        
    Bzlist_training = []
    for paramn in training_paramlist:
        Bzlist_training.append(paramn["Bz"] )
    # tag ="Bx="+str(Bx)+"Bztrain"+str(Bzlist_training)+"Bztarget"+str(Bzlist_target)\
    #      +"backend_name="+backend_name +"layout=" + str(layout)
    #
    # pickle.dump(basis_circuits_list, open("matrix_data/basis_circuits_list"+tag+".p", "wb"))
    filename = "matrix_data/basis_circuits_list" + "Bx=" + str(Bx) + "Bztrain" + str(
        Bzlist_training) + "Bztarget" + str(Bzlist_target) + "from_qiskit_bundle.p"
    pickle.dump(basis_circuits_list, open(filename, "wb"))

    ############## Lets try the bundling method here ##################

    qc_bundlelist = make_circuit_bundle_forQC(basis_circuits_list = basis_circuits_list, N = N,pbc=pbc)

    # print the bundle list
    # print( qc_bundlelist)

    # filename = "circuits/transpiled_test"+"opt="+str(optimizationlevel)+".p"
    # filename = "circuits/qiskit_bundle_" +tag+  "opt=" + str(optimizationlevel) + ".p"
    filename = "circuits/sc_trans" + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(
        Bzlist_target) \
               + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel) + ".p"
    backend = get_backend(backend_name=backend_name)
    transpiled_bundlelist = qk.transpile(circuits=qc_bundlelist,backend=backend,initial_layout=layout,optimization_level=optimizationlevel)

    pickle.dump(transpiled_bundlelist, open(filename , "wb" ) )

    qobj = qk.assemble(transpiled_bundlelist, backend=backend, shots=shots)
    # print("qobjects created")
    job = backend.run(qobj)
    results = job.result()
    # allresults = results.results
    allcounts = results.get_counts()
    # read out error mitigation
    qr = QuantumRegister(1)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    # make sure it is layout[0]
    t_qc_calibs = qk.transpile(meas_calibs, backend=backend,initial_layout=[layout[0]])
    qobj_calibs = qk.assemble(t_qc_calibs,backend=backend, shots=shots)
    job_calibs = backend.run(qobj_calibs)
    calib_results = job_calibs.result()

    meas_fitter = CompleteMeasFitter(calib_results, state_labels, circlabel='mcal')
    meas_filter = meas_fitter.filter
    print("calibration matrix : ")
    print(meas_fitter.cal_matrix)
    mitigated_results = meas_filter.apply(results)
    mitigated_counts = mitigated_results.get_counts()


    plist = []
    for i in range(len(transpiled_bundlelist)):
        p = get_p_from_counts(counts=mitigated_counts[i])
        # p = get_p_from_counts(counts=allcounts[i])
        plist.append(p)

    # p_overlap = plist[0] + 1.j * plist[1]
    # print("overlap element from p_overlap",p_overlap)
    #####
    basis_length = len(training_paramlist)
    overlap_matrix_bundle,Bzmatrix_bundle,Bxmatrix_bundle,Jmatrix_bundle = \
        unpackplist(plist=plist, N=N, pbc=pbc, basis_length=basis_length)

    tag = "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(Bzlist_target) + \
          "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel)
    print("Overlap matrix_bundle qasm:\n", overlap_matrix_bundle)
    np.savetxt('matrix_data/overlap_matrix_bundle_qiskit' + tag + '.txt', overlap_matrix_bundle)

    np.savetxt('matrix_data/Bzmatrix_bundle_qiskit' + tag + '.txt', Bzmatrix_bundle)
    np.savetxt('matrix_data/Bxmatrix_bundle_qiskit' + tag + '.txt', overlap_matrix_bundle)
    np.savetxt('matrix_data/Jmatrix_bundle_qiskit' + tag + '.txt', Jmatrix_bundle)
    print("Bzmatrix_bundle qasm:\n", Bzmatrix_bundle)
    print("Bxmatrix_bundle qasm:\n", Bxmatrix_bundle)
    print("Jmatrix_bundle qasm:\n", Jmatrix_bundle)

    # ### getting evals
    evals_qc_bundle = np.zeros([len(target_paramlist),len(training_paramlist)],dtype=complex)
    # Uilist = get_training_vectors(training_paramlist)
    for ip, paramn in enumerate(target_paramlist):
        evals_bundle,evecs_bundle = get_evals_of_target_ham_from_matrices \
            (overlap_matrix=overlap_matrix_bundle, Bzmatrix=Bzmatrix_bundle, Bxmatrix=Bxmatrix_bundle,
             Jmatrix=Jmatrix_bundle, paramn=paramn)
        for k in range(len(training_paramlist)):
            evals_qc_bundle[ip, k] = evals_bundle[k]

    ##################### bundling over ################
    # return evals_qc,evals_qc_bundle
    return evals_qc_bundle


