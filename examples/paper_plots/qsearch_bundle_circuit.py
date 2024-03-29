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

from numpy.linalg import cond
#############################################################

# given the parameters this gets the vqe circuit and the unitary
def get_circuit_unitary_fromVQE(param):
    J = param["J"]
    Bx = param["Bx"]
    Bz = param["Bz"]
    pbc = param["pbc"]
    N = param["N"]

    # this hamiltonian needs to be written in using pauli strings later
    ham = XY_hamiltonian(J, Bx, Bz, N, pbc)
    hamiltonian = J * ((X ^ X) + (Y ^ Y)) + Bz * ((I ^ Z) + (Z ^ I)) + Bx * ((I ^ X) + (X ^ I))
    # print(ham)
    # print(hamiltonian)
    # print(type(hamiltonian))
    initial = [4.13850621, 4.68105177, 0.10459295, 5.19511699, -3.60948918, 2.08480452, -3.83679895, 1.9264198]
    # initial = None

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
    qc.compose(cc, inplace=True)
    # print(qc.decompose().draw())
    job = execute(qc, backend=backend)
    result = job.result()

    U = result.get_unitary(qc)

    return cc, U
##############################
# given the training parameter list this function returns the basis circuit,unitaries list from vqe
def get_basis_list(training_paramlist):
    basis_circuits_list = []
    basis_unitaries_list = []
    for param in training_paramlist:
        circuiti,Unitaryi = get_circuit_unitary_fromVQE(param=param)
        basis_circuits_list.append(circuiti)
        basis_unitaries_list.append(Unitaryi)

    return basis_circuits_list,basis_unitaries_list
######################################################

# To not use any VQE, pass the vector this will give the corresponding unitary.
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

# pass the vectors list this will give the unitaries list
def get_training_vectors_exact(basis_vecs):
    Uilist=[]
    for basis_v in basis_vecs:
        Ui = makeUnitaryfromvec(vec=basis_v)
        Uilist.append(Ui)
    return Uilist
#############################
# This function gets the simulator either Aer qasm or ibmq ones
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
#########################################################
# given the circ this func attaches the basis rotation gate and returns the circuit
def make_measure_inbasis_circ(circ, N=2, basis="X"):
    q = QuantumRegister(N)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    # print(circ)
    qc.compose(circ, inplace=True)
    # qc.compose(circ, inplace=False)
    if (basis == "X"):
        qc.h(q[0])
    elif (basis == "Y"):
        qc.rx(-np.pi / 2, q[0])
    qc.measure(q[0], c[0])
    return qc
##################################################################
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
###########################

# given the two basis circuits this returns the dot product circuit with control qubit
def convert_U_to_qsearch_circuit(U,circname="circ",project_dir = "project_dir"):
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

def make_cntrlU(U):
    L = len(U)
    cntrlU = np.identity(2*L,dtype=complex)
    for i in range(L):
        for j in range(L):
            cntrlU[L+i,L+j] = U[i,j]
    return cntrlU

def phi_ij(Ui,Uj,N,circname,project_dir):
    Ujd = np.conjugate(np.transpose(Uj))
    U = np.matmul(Ujd, Ui)
    had = (1/np.sqrt(2))*np.array([[1,1],[1,-1]],dtype=complex)
    Uini = np.kron(had,np.identity(2**N,dtype=complex))
    print(np.shape(Uini))
    print(Uini)
    U_ansys = make_cntrlU(U=U)
    U_ansys = np.matmul(U_ansys,Uini)
    print(np.shape(U_ansys))
    circij = convert_U_to_qsearch_circuit(U=U_ansys,circname=circname,project_dir=project_dir)
    return circij
########################################
# given the basis circuitlist this function gets the overlap matrix by running the circuits
# this is not used in bundling
# def create_overlap_matrix_fromQC(basis_unitaries_list,N=2,backend_name="qasm_simulator",layout=[0,1,2],shots=8192):
#     basis_length = len(basis_unitaries_list)
#     # params= []
#     overlap_matrix = np.identity(basis_length,dtype="complex")
#     # print(overlap_matrix)
#     for i in range(basis_length-1):
#         for j in range(i+1,basis_length):
#             # print(i,j)
#             circij = phi_ij(Ui=basis_unitaries_list[i],Uj=basis_unitaries_list[j],N=N)
#             # print(circij.decompose().draw())
#             phij = measure_real_imaginary_from_circ(circ=circij,N=N+1,backend_name=backend_name,layout=layout,shots=shots)
#             # phij = get_phii_phij_fromcircuit(Ui=Uilist[i],Uj = Uilist[j],N=N)
#             overlap_matrix[i,j] = phij
#             overlap_matrix[j,i] = np.conjugate(phij)
#
#     return overlap_matrix
##################################################

# This takes in the basis circuit elements and a single Pauli String to return the average m value
# not used while bundling
# def get_HPauli_ij_fromcircuit(circi,circj,paulisop,N=2,backend_name="qasm_simulator",layout=[0,1,2],shots=8192):
#     # circuiti,Ui = get_circuit_unitary_fromVQE(param=parami,N=N)
#     # circuitj,Uj = get_circuit_unitary_fromVQE(param=paramj,N=N)
#
#     H = many_kron(ops=paulisop)
#     # print(  "H Pauli = ",H)
#
#     circ_H = QuantumCircuit(N)
#     circ_H.append(UnitaryGate(H),[0,1])
#     gateH = circ_H.to_gate()
#     cgateH  = gateH.control()
#
#     gatei = circi.to_gate()
#     cgatei = gatei.control()
#
#     # gatej = circj.to_gate()
#     gatej = circj.to_gate()
#     gatejinv = gatej.inverse()
#     cgatejinv = gatejinv.control()
#
#     q = QuantumRegister(N+1)
#     circ_Pij = QuantumCircuit(q)
#     circ_Pij.h(q[0])
#     circ_Pij.append(cgatei,[0,1,2])
#     circ_Pij.append(cgateH,[0,1,2])
#     circ_Pij.append(cgatejinv,[0,1,2])
#     # print(circ_Pij.decompose().draw())
#     # return circ_Pij
#
#     p = measure_real_imaginary_from_circ(circ=circ_Pij,N=N+1,backend_name=backend_name,layout=layout,shots=shots)
#
#     return p

# This takes in the basis circuit elements and makes all Pauli String to return the Bz,Bx,J matrix parts
# not used while bundling
# def get_HXY_bare_ij_fromcircuit( N, pbc,circi,circj,backend_name,layout,shots):
#
#     hamBz=0.0+0.0*1.j
#     hamBx=0.0+0.0*1.j
#     hamXX=0.0+0.0*1.j
#     hamYY=0.0+0.0*1.j
#
#     # Build hamiltonian matrix
#     for isite in range(N):
#
#         # BZ
#         oplist = ['I']*N
#         oplist[isite] = 'Z'
#         #print("".join(oplist))
#         hamij = get_HPauli_ij_fromcircuit(circi=circi,circj=circj,paulisop=oplist,N=N,backend_name=backend_name,layout=layout,shots=shots)
#         hamBz += hamij
#
#         # BX
#         oplist = ['I']*N
#         oplist[isite] = 'X'
#         #print("".join(oplist))
#         hamij = get_HPauli_ij_fromcircuit(circi=circi,circj=circj,paulisop=oplist,N=N,backend_name=backend_name,layout=layout,shots=shots)
#         hamBx += hamij
#
#         jsite = (isite + 1) % N
#         if not(jsite == isite+1) and not pbc:
#             continue
#
#         # XX
#         oplist = ['I']*N
#         oplist[isite] = 'X'
#         oplist[jsite] = 'X'
#         #print("".join(oplist))
#         hamij = get_HPauli_ij_fromcircuit(circi=circi,circj=circj,paulisop=oplist,N=N,backend_name=backend_name,layout=layout,shots=shots)
#         hamXX += hamij
#         # YY
#         oplist = ['I']*N
#         oplist[isite] = 'Y'
#         oplist[jsite] = 'Y'
#         #print("".join(oplist))
#         hamij = get_HPauli_ij_fromcircuit(circi=circi,circj=circj,paulisop=oplist,N=N,backend_name=backend_name,layout=layout,shots=shots)
#         hamYY += hamij
#
#     return [hamBz,hamBx,hamXX,hamYY]
##########################
# This takes in the basis circuit list and returns the Bz,Bx,J matrix
# # not used while bundling
# def make_hamiltonian_contributions_fromQC(basis_circuits_list,N,pbc,backend_name,layout,shots):
#     base_length = len(basis_circuits_list)
#
#     Bzmatrix = np.zeros((base_length,base_length),dtype="complex")
#     Bxmatrix = np.zeros((base_length,base_length),dtype="complex")
#     Jmatrix = np.zeros((base_length,base_length),dtype="complex")
#
#     for i in range(base_length):
#         for j in range(i,base_length):
#             [hamBz,hamBx,hamXX,hamYY] = get_HXY_bare_ij_fromcircuit(N, pbc,circi=basis_circuits_list[i],circj=basis_circuits_list[j],backend_name=backend_name,layout=layout,shots=shots)
#             Bxmatrix[i,j] = hamBx
#             Bzmatrix[i,j] = hamBz
#             Jmatrix[i,j] = hamXX+hamYY
#             if(i!= 0):
#                 Bxmatrix[j,i] = np.conjugate(Bxmatrix[i,j])
#                 Bzmatrix[j,i] = np.conjugate(Bzmatrix[i,j])
#                 Jmatrix[j,i] = np.conjugate(Jmatrix[i,j])
#
#     return Bzmatrix,Bxmatrix,Jmatrix
# ###################################
###################################
# def get_HXY_together_ij_fromcircuit(paramn,circi,circj):
#     J =paramn["J"]
#     Bx = paramn["Bx"]
#     Bz = paramn["Bz"]
#     N = paramn["N"]
#     pbc = paramn["pbc"]
    
#     [hamBz,hamBx,hamXX,hamYY] = get_HXY_bare_ij_fromcircuit( N, pbc,circi,circj)
#     # prntlst = [hamBz,hamBx,hamXX,hamYY] 
#     # print(prntlst)
#     hamij = Bz*hamBz + Bx*hamBx + J*(hamXX+hamYY)
    
#     return hamij

# def make_target_hamiltonian_fromQC(basis_circuits_list,paramn):
#     base_length = len(basis_circuits_list)
#     ham_target = np.identity(base_length,dtype="complex")
#     for i in range(base_length):
#         for j in range(i,base_length):
#             hamij = get_HXY_together_ij_fromcircuit(paramn=paramn,circi=basis_circuits_list[i],circj=basis_circuits_list[j])
#             ham_target[i,j] = hamij
#             ham_target[j,i] = np.conjugate(hamij)        
 
#     return ham_target
########################################
# This takes in the the Bz,Bx,J matrix paramn and returns target hamiltonian
def make_target_hamiltonian_fromQC(Bzmatrix,Bxmatrix,Jmatrix,paramn):
    
    J = paramn["J"]
    Bx = paramn["Bx"]
    Bz = paramn["Bz"]

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

    smaller_ham = make_target_hamiltonian_fromQC(Bzmatrix,Bxmatrix,Jmatrix,paramn)
    print("Hamiltonian qsearch:\n", smaller_ham)
    # This is to solve smaller ham explicitly using inverse
    #mham = linalg.inv(overlap_matrix) @ smaller_ham
    #evals, evecs = linalg.eigh(mham)
    
    evals, evecs = linalg.eigh(smaller_ham,overlap_matrix)
    
    # evals, evecs = linalg.eigh(smaller_ham,overlap_matrix,driver="gv")
    print("Evals qsearch: ",evals)
    return evals,evecs
############################################################

# def get_evals_target_ham_qasmcirc(basis_circuits_list,paramn):
    
#     J =paramn["J"]
#     Bx = paramn["Bx"]
#     Bz = paramn["Bz"]
#     N = paramn["N"]
#     pbc = paramn["pbc"]
   
#     overlap_matrix = create_overlap_matrix_fromQC(basis_circuits_list=basis_circuits_list,N=N)
    
# #     overlap_matrix[0,1] *= -1
# #     overlap_matrix[1,0] *= -1
# #     overlap_matrix[1,2] *= -1
# #     overlap_matrix[2,1] *= -1
    
#     # print("Overlap:")
#     # print(overlap_matrix)
    
#     #print(linalg.lu(overlap_matrix))
    
#     smaller_ham = make_target_hamiltonian_fromQC(basis_circuits_list=basis_circuits_list,paramn=paramn)
    
    
# #     smaller_ham[0,1] *= -1
# #     smaller_ham[1,0] *= -1
# #     smaller_ham[1,2] *= -1
# #     smaller_ham[2,1] *= -1
    
#     # print("Hamiltonian:")
#     # print(smaller_ham)
    
#     evals, evecs = linalg.eigh(smaller_ham,overlap_matrix)
#     # evals, evecs = linalg.eigh(smaller_ham,overlap_matrix,driver="gv")
#     # print("Evals: ",evals)
#     return evals

####################################
# def get_evals_targetlist_qasmcirc(training_paramlist,target_paramlist ):
#     evals_qc = np.zeros([len(target_paramlist),len(training_paramlist)],dtype=complex)
#     basis_circuits_list = get_circuit_list(training_paramlist)
#     # Uilist = get_training_vectors(training_paramlist)
#     for ip,paramn in enumerate(target_paramlist):
#         evals = get_evals_target_ham_qasmcirc(basis_circuits_list,paramn)
#         for k in range(len(training_paramlist)):
#                 evals_qc[ip,k] = evals[k]
#     return evals_qc
#############################
#####################
# given the circi, circj, paulisop this makes the circuit for Paulidot product
def make_Pauli_ij_circuit(Ui, Uj, paulisop,N, circname,project_dir):
    # circuiti,Ui = get_circuit_unitary_fromVQE(param=parami,N=N)
    # circuitj,Uj = get_circuit_unitary_fromVQE(param=paramj,N=N)

    H = many_kron(ops=paulisop)
    # print(  "H Pauli = ",H)

    Ujd = np.conjugate(np.transpose(Uj))
    U = np.matmul(H, Ui)
    U = np.matmul(Ujd, U)
    # U_ansys = make_cntrlU(U=U)

    had = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    Uini = np.kron(had, np.identity(2**N, dtype=complex))
    # print(np.shape(Uini))
    # print(Uini)
    U_ansys = make_cntrlU(U=U)
    U_ansys = np.matmul(U_ansys, Uini)

    circ_Pij = convert_U_to_qsearch_circuit(U=U_ansys,circname=circname,project_dir=project_dir)

    return circ_Pij
#####################################
# given the circi, circj, this makes the circuit for Paulidot product and bundles all of them
def make_Paulibundledcircuits_ij(N, pbc, Ui, Uj,circname_ij,project_dir):
    bundled_Pauliij_circuitlist = []
    # Build hamiltonian matrix
    for isite in range(N):

        # BZ
        oplist = ['I'] * N
        oplist[isite] = 'Z'
        circname = "Pauli"+'Z'+str(isite) + circname_ij
        # print("".join(oplist))
        circ_Pij = make_Pauli_ij_circuit(Ui=Ui, Uj=Uj, paulisop=oplist, N=N,circname=circname,project_dir=project_dir )
        circ_Pijx = make_measure_inbasis_circ(circ= circ_Pij, N=N+1, basis="X")
        bundled_Pauliij_circuitlist.append(circ_Pijx)
        circ_Pijy = make_measure_inbasis_circ(circ= circ_Pij, N=N+1, basis="Y")
        bundled_Pauliij_circuitlist.append(circ_Pijy)

        # BX
        oplist = ['I'] * N
        oplist[isite] = 'X'
        # print("".join(oplist))
        circname = "Pauli" + 'X' + str(isite) + circname_ij
        circ_Pij = make_Pauli_ij_circuit(Ui=Ui, Uj=Uj, paulisop=oplist, N=N,circname=circname,project_dir=project_dir )
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
        circname = "Pauli" + 'X' + str(isite)+  'X' + str(jsite) + circname_ij
        circ_Pij = make_Pauli_ij_circuit(Ui=Ui, Uj=Uj, paulisop=oplist, N=N,circname=circname,project_dir=project_dir)
        circ_Pijx = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="X")
        bundled_Pauliij_circuitlist.append(circ_Pijx)
        circ_Pijy = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="Y")
        bundled_Pauliij_circuitlist.append(circ_Pijy)
        # YY
        oplist = ['I'] * N
        oplist[isite] = 'Y'
        oplist[jsite] = 'Y'
        # print("".join(oplist))
        circname = "Pauli" + 'Y' + str(isite) +  'Y' + str(jsite) + circname_ij
        circ_Pij = make_Pauli_ij_circuit(Ui=Ui, Uj=Uj, paulisop=oplist,N=N, circname=circname,project_dir=project_dir)
        circ_Pijx = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="X")
        bundled_Pauliij_circuitlist.append(circ_Pijx)
        circ_Pijy = make_measure_inbasis_circ(circ=circ_Pij, N=N+1, basis="Y")
        bundled_Pauliij_circuitlist.append(circ_Pijy)

    return bundled_Pauliij_circuitlist
#########################################
# given the basis circuit list this bundles circuits for all ij
def make_hamiltonian_circuit_bundlelist(basis_unitaries_list, N, pbc,project_dir):

    base_length = len(basis_unitaries_list)

    bundled_Paulifull_circuitlist = []

    for i in range(base_length):
        for j in range(i, base_length):
            bundled_Paulij_circuitlist = make_Paulibundledcircuits_ij(N=N, pbc=pbc, Ui=basis_unitaries_list[i],Uj=basis_unitaries_list[j],  circname_ij = str(i)+str(j),project_dir=project_dir)
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
def make_circuit_bundle_forQC(basis_unitaries_list , N ,pbc,project_dir):
#     lets bundle the overlap circuits
    basis_length = len(basis_unitaries_list)
    qc_bundlelist=[]
    for i in range (basis_length-1):
        for j in range(i+1,basis_length):
            circname = "Phi"+str(i)+str(j)
            circij = phi_ij(Ui=basis_unitaries_list[i],Uj=basis_unitaries_list[j],N=N,circname=circname, project_dir=project_dir)
            phiijx = make_measure_inbasis_circ(circ=circij,N=N+1,basis="X")
            qc_bundlelist.append(phiijx)
            phiijy = make_measure_inbasis_circ(circ=circij, N=N+1, basis="Y")
            qc_bundlelist.append(phiijy)
    # print(qc_bundlelist)

#     lets bundle the pauli circuits
    ham_bundlelist = make_hamiltonian_circuit_bundlelist(basis_unitaries_list=basis_unitaries_list, N=N, pbc=pbc,project_dir=project_dir)
    qc_bundlelist.extend(ham_bundlelist)
    # print(qc_bundlelist)
    print("len(ham_bundlelist)", len(ham_bundlelist))
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
            phij = poverlaplist[index*2 + 0] + 1.j* poverlaplist[index*2 + 1]
            # phij = poverlaplist[index*basis_length + 0] + 1.j* poverlaplist[index*basis_length + 1]
            overlap_matrix[i, j] = phij
            overlap_matrix[j, i] = np.conjugate(phij)
            index=index+1
    #         lets get the hamiltoninan matrix parts
    phamlist = plist[length_overlap:]
    phamarray = np.array(phamlist)
    Bzmatrix, Bxmatrix, Jmatrix = get_hamiltonian_contributions_from_bundled_Paulifull( base_length=basis_length, N=N, pbc=pbc, parray_Paulifull=phamarray)
    # return overlap_matrix
    return overlap_matrix,Bzmatrix, Bxmatrix, Jmatrix
#############################################
# this is the main function where things are put together
def get_evals_targetlist_qsearchcirc(training_paramlist,target_paramlist,backend_name,layout,shots ,basis_vecs, Basis_exact_flag=True,date_flag = True ):
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

    ################ getting basis circuit list  #######################
    evals_qc = np.zeros([len(target_paramlist),len(training_paramlist)],dtype=complex)

    # basis_circuits_list,basis_unitaries_list = get_basis_list(training_paramlist = training_paramlist)

    if (Basis_exact_flag == True):
        basis_unitaries_list = get_training_vectors_exact(basis_vecs=basis_vecs)
    else:
        basis_circuits_list,basis_unitaries_list = get_basis_list(training_paramlist = training_paramlist)
    #
    # basis_unitaries_list = Uilist
    # print("printing Uilist from the qsearch")
    # for u in basis_unitaries_list:
    #     print(u)
    paramn = target_paramlist[0]
    N = paramn["N"]
    J = paramn["J"]
    pbc = paramn["pbc"]
    Bx = paramn["Bx"]
    Bzlist_target=[]
    for paramn in target_paramlist:
        Bzlist_target.append(paramn["Bz"])
    #
    Bzlist_training = []
    for paramn in training_paramlist:
        Bzlist_training.append(paramn["Bz"] )
    metadata = [N, J, pbc, Bx, Bzlist_training, Bzlist_target]
    # tag ="Bx="+str(Bx)+"Bztrain"+str(Bzlist_training)+"Bztarget"+str(Bzlist_target)\
    #      +"backend_name="+backend_name +"layout=" + str(layout)
    #
    # pickle.dump(basis_circuits_list, open("matrix_data/basis_circuits_list"+tag+".p", "wb"))
    import datetime
    x = datetime.datetime.now()
    date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
    if (Basis_exact_flag):
        tag = "base_unitaries_Nsite=" + str(N) + "Bx="+str(Bx) + "J=" + str(J)+"pbc=" + str(pbc) + "Bztrain"+str(Bzlist_training) + "Bztarget"+str(Bzlist_target) \
                + "Basis_exact_flag= " + str(Basis_exact_flag)
        if(date_flag):
            tag = tag + "date_" + date
        tag += ".p"
        filename = "results/EVC/Basis/" + tag
        pickle.dump(basis_unitaries_list, open(filename, "wb"))
    else:
        tag = "base_circuits_Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J) + "pbc=" + str(pbc) + "Bztrain" + str(
            Bzlist_training) + "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) + "opt=" +\
              str(optimizationlevel) + "Basis_exact_flag= " + str(Basis_exact_flag)
        if (date_flag):
            tag = tag + "date_" + date
        tag += ".p"
        filename = "results/EVC/Basis/" + tag
        pickle.dump(basis_circuits_list, open(filename, "wb"))
    print("len(basis_unitaries_list)",len(basis_unitaries_list))
    # filename = "matrix_data/basis_unitaries_list" + "Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(Bzlist_target) \
    #            +"Basis_exact_flag= "+str(Basis_exact_flag)+ "from_sc_bundle.p"
    # filename = "results/EVC/Basis/"+tag
    # pickle.dump(basis_unitaries_list, open(filename, "wb"))

    ############## Lets try the bundling method here ##################

    project_dir = "qsearch_dir/EVC_project_dir/Z_meas" + str(N) + "site" + "Bx="+ str(Bx) + "Bztrain" + str(Bzlist_training)
    # print("basis_unitaries_list before circuits",basis_unitaries_list)
    qc_bundlelist = make_circuit_bundle_forQC(basis_unitaries_list = basis_unitaries_list, N = N,pbc=pbc,project_dir=project_dir)
    # print("basis_unitaries_list after circuits", basis_unitaries_list)
    # print the bundle list
    # print( qc_bundlelist)

    # filename = "circuits/sc_trans" + "Nsite=" + str(N) + "Bx="+str(Bx)+"Bztrain"+str(Bzlist_training)+"Bztarget"+str(Bzlist_target)\
    #      +"backend_name="+backend_name +"layout=" + str(layout)+"opt="+str(optimizationlevel)+".p"
    backend = get_backend(backend_name=backend_name)
    print("Transpiling ", backend)
    transpiled_bundlelist = qk.transpile(circuits=qc_bundlelist,backend=backend,initial_layout=layout,optimization_level=optimizationlevel)
    print("Transpiled")
    print("len (qc_bundlelist) ",len(qc_bundlelist))
    # pickle.dump(transpiled_bundlelist, open(filename , "wb" ) )

    if (date_flag):
       tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J) + "pbc=" + str(pbc) + "Bztrain" + str(
            Bzlist_training) + \
              "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(
            layout) + "shots=" + str(shots) + "opt=" + \
              str(optimizationlevel) + "date" + date + ".p"
    else:
        tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J) + "pbc=" + str(pbc) + "Bztrain" + str(
            Bzlist_training) + "Bztarget" + str(Bzlist_target) \
              + "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) + "opt=" + str(
            optimizationlevel) + ".p"
    filename_notrans = "circuits/EVC/sc_EVC_notrans" + tag
    filename_trans = "circuits/EVC/trans/sc_EVC_trans" + tag
    pickle.dump(qc_bundlelist, open(filename_notrans, "wb"))
    pickle.dump(transpiled_bundlelist, open(filename_trans, "wb"))

    qobj = qk.assemble(transpiled_bundlelist, backend=backend, shots=shots)
    # print("qobjects created")
    job = backend.run(qobj)

    results = job.result()
    print("jobid evc: ",job.job_id())
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
    print("jobid evc calibs: ", job_calibs.job_id())
    meas_fitter = CompleteMeasFitter(calib_results, state_labels, circlabel='mcal')
    meas_filter = meas_fitter.filter
    print("calibration matrix : ")
    print(meas_fitter.cal_matrix)
    mitigated_results = meas_filter.apply(results)
    mitigated_counts = mitigated_results.get_counts()
    ########## saving results  ##############
    # filename = "circuits/sc_results"  + "Nsite=" + str(N) +  "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(
    #     Bzlist_target) \
    #            + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel) + ".p"

    # filename = "circuits/sc_results" + "Nsite=" + str(N) +"J=" + str(J) + "Bx=" + str(Bx) + "Bztrain" + str( Bzlist_training) \
    #            + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel) + ".p"

    # pickle.dump(mitigated_results, open(filename, "wb"))
    #
    # filename = "circuits/sc_counts"  + "Nsite=" + str(N) +  "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(
    #     Bzlist_target) \
    #            + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel) + ".p"
    #
    # filename = "circuits/sc_counts" + "Nsite=" + str(N) + "J=" + str(J) + "Bx=" + str(Bx) + "Bztrain" + str(
    #     Bzlist_training) \
    #            + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel) + ".p"

    # pickle.dump(mitigated_counts , open(filename, "wb"))


##################### saving over ################
    plist = []
    for i in range(len(transpiled_bundlelist)):
        p = get_p_from_counts(counts=mitigated_counts[i])
        # p = get_p_from_counts(counts=allcounts[i])
        plist.append(p)
        ################# Lets save the results and counts ##################
        filename_results_together = "results/EVC/results_counts" + tag
        pickle.dump(
            [results, calib_results, mitigated_results, allcounts, meas_fitter.cal_matrix, mitigated_counts, metadata],
            open(filename_results_together, "wb"))
        filename_results = "results/EVC/results" + tag
        pickle.dump([results, calib_results, mitigated_results, metadata],
                    open(filename_results, "wb"))
        filename_counts = "results/EVC/counts" + tag
        pickle.dump([allcounts, meas_fitter.cal_matrix, mitigated_counts, plist, metadata],
                    open(filename_counts, "wb"))
    # p_overlap = plist[0] + 1.j * plist[1]
    # print("overlap element from p_overlap",p_overlap)
    #####
    basis_length = len(training_paramlist)
    # overlap_matrix_bundle = \
    #     unpackplist(plist=plist, N=N, pbc=pbc, basis_length=basis_length)
    overlap_matrix_bundle, Bzmatrix_bundle, Bxmatrix_bundle, Jmatrix_bundle = \
        unpackplist(plist=plist, N=N, pbc=pbc, basis_length=basis_length)

    # import datetime
    # x = datetime.datetime.now()
    # date = str(x.day) + "_" + str(x.month) + "_" + str(x.year)
    tag = "Nsite=" + str(N) + "Bx=" + str(Bx) + "J=" + str(J) + "pbc=" + str(pbc) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(Bzlist_target)+ \
          "backend_name=" + backend_name + "layout=" + str(layout) + "shots=" + str(shots) + "opt=" + str(optimizationlevel)
    if(date_flag):
        if (date_flag):
            tag = tag + "date_" + date
    np.savetxt('matrix_data/overlap_matrix_bundle_sc' + tag + '.txt', overlap_matrix_bundle)
    np.savetxt('matrix_data/Bxmatrix_bundle_sc' + tag + '.txt', Bxmatrix_bundle)
    np.savetxt('matrix_data/Bzmatrix_bundle_sc' + tag + '.txt', Bzmatrix_bundle)
    np.savetxt('matrix_data/Jmatrix_bundle_sc' + tag + '.txt', Jmatrix_bundle)
    print("Overlap matrix_bundle qsearch:\n", overlap_matrix_bundle)
    print("Overlap matrix_bundle qsearch cond:\t", cond(overlap_matrix_bundle))
    print("Bzmatrix_bundle qsearch:\n", Bzmatrix_bundle)
    print("Bxmatrix_bundle qsearch:\n", Bxmatrix_bundle)
    print("Jmatrix_bundle qsearch:\n", Jmatrix_bundle)

    # ### getting evals
    evals_qc_bundle = np.zeros([len(target_paramlist),len(training_paramlist)],dtype=complex)
    for ip, paramn in enumerate(target_paramlist):
        evals_bundle,evecs_bundle = get_evals_of_target_ham_from_matrices \
            (overlap_matrix=overlap_matrix_bundle, Bzmatrix=Bzmatrix_bundle, Bxmatrix=Bxmatrix_bundle,
             Jmatrix=Jmatrix_bundle, paramn=paramn)
        for k in range(len(training_paramlist)):
            evals_qc_bundle[ip, k] = evals_bundle[k]

    ##################### bundling over ################tag = tag.replace(".p", ".txt")
    filename_summary = "summary/EVC_eigen_values_"+tag
    # filename = "summary/LCU_eigen_values_Nsite=" + str(N) + "Bx=" + str(Bx) + "Bztrain" + str(
    #     Bzlist_training) + "Bztarget" + str(Bzlist_target) + "backend_name=" + backend_name + "layout=" + str(layout) + "opt=" + str(optimizationlevel)".txt"
    f = open(filename_summary, "w")
    f.write("Energylist EVC : "+str(evals_qc_bundle))
    # f.write("\n Energylist mitigated : " + str(Energylist))
    # f.write("\n Magnetizationlist raw : " + str(maglist_raw))
    # f.write("\n Magnetizationlist mitigated : " + str(maglist))
    f.close()


    # return evals_qc,evals_qc_bundle
    return evals_qc_bundle

#
# def get_evals_targetlist_qasmcirc(training_paramlist, target_paramlist):
#     ####################
#     backend_name = "qasm_simulator"
#     layout = [3, 7, 5]
#     ###########################
#     evals_qc = np.zeros([len(target_paramlist), len(training_paramlist)], dtype=complex)
#     basis_circuits_list = get_circuit_list(training_paramlist)
#
#     paramn = target_paramlist[0]
#     N = paramn["N"]
#     pbc = paramn["pbc"]
#     Bx = paramn["Bx"]
#     Bzlist_target = []
#     for paramn in target_paramlist:
#         Bzlist_target.append(paramn["Bz"])
#
#     Bzlist_training = []
#     for paramn in training_paramlist:
#         Bzlist_training.append(paramn["Bz"])
#     tag = "Bx=" + str(Bx) + "Bztrain" + str(Bzlist_training) + "Bztarget" + str(Bzlist_target) \
#           + "backend_name=" + backend_name + "layout=" + str(layout)
#
#     pickle.dump(basis_circuits_list, open("matrix_data/basis_circuits_list" + tag + ".p", "wb"))
#
#     overlap_matrix = create_overlap_matrix_fromQC(basis_circuits_list=basis_circuits_list, N=N,
#                                                   backend_name=backend_name, layout=layout)
#     print("Overlap matrix qasm:\n", overlap_matrix)
#     np.savetxt('matrix_data/overlap_matrix' + tag + '.txt', overlap_matrix)
#     Bzmatrix, Bxmatrix, Jmatrix = make_hamiltonian_contributions_fromQC(basis_circuits_list=basis_circuits_list, N=N,
#                                                                         pbc=pbc, backend_name=backend_name,
#                                                                         layout=layout)
#     np.savetxt('matrix_data/Bzmatrix' + tag + '.txt', Bzmatrix)
#     np.savetxt('matrix_data/Bxmatrix' + tag + '.txt', overlap_matrix)
#     np.savetxt('matrix_data/Jmatrix' + tag + '.txt', Jmatrix)
#
#     # Uilist = get_training_vectors(training_paramlist)
#     for ip, paramn in enumerate(target_paramlist):
#         # evals = get_evals_target_ham_qasmcirc(basis_circuits_list,paramn)
#         # evals = get_evals_target_ham(basis_circuits_list,paramn)
#         evals = get_evals_of_target_ham_from_matrices(overlap_matrix, Bzmatrix, Bxmatrix, Jmatrix, paramn)
#         for k in range(len(training_paramlist)):
#             evals_qc[ip, k] = evals[k]
#     return evals_qc
#
