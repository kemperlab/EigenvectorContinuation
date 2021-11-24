from qiskit import Aer, execute, QuantumCircuit
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, SLSQP
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.opflow import I, X, Y, Z

import numpy as np
from scipy import linalg

paulis = {}
paulis['X'] = np.array([[0,1],[1,0]],dtype=complex)
paulis['Y'] = np.array([[0,-1.j],[1.j,0]],dtype=complex)
paulis['Z'] = np.array([[1,0],[0,-1]],dtype=complex)
paulis['I'] = np.array([[1,0],[0,1]],dtype=complex)

####################

def many_kron(ops):
    # Takes an array of Pauli characters and produces the tensor product
    op = paulis[ops[0]]
    if len(ops) == 1:
        return op

    for opj in ops[1:]:
        op = np.kron(op,paulis[opj])

    return op

###############################

def XY_hamiltonian(J, Bx, Bz, N, pbc):


    ham = np.zeros([2**N,2**N],dtype=complex)

    # Build hamiltonian matrix
    for isite in range(N):

        # BZ
        oplist = ['I']*N
        oplist[isite] = 'Z'
        #print("".join(oplist))
        ham += Bz*many_kron(oplist)

        # BX
        oplist = ['I']*N
        oplist[isite] = 'X'
        #print("".join(oplist))
        ham += Bx*many_kron(oplist)

        jsite = (isite + 1) % N
        if not(jsite == isite+1) and not pbc:
            continue

        # XX
        oplist = ['I']*N
        oplist[isite] = 'X'
        oplist[jsite] = 'X'
        #print("".join(oplist))
        ham += J*many_kron(oplist)

        # YY
        oplist = ['I']*N
        oplist[isite] = 'Y'
        oplist[jsite] = 'Y'
        #print("".join(oplist))
        ham += J*many_kron(oplist)

    return ham
##############################

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

###########################################################
    
def make_cntrlU(U):
    L = len(U)
    cntrlU = np.identity(2*L,dtype=complex)
    for i in range(L):
        for j in range(L):
            cntrlU[L+i,L+j] = U[i,j]
    return cntrlU

def reducerho(wf,lA,lB):
    
    slicedwf=np.reshape(wf,(lA,lB))
    slicestar=np.transpose(slicedwf)
    slicestar=np.conj(slicestar)
    rhoA=np.matmul(slicedwf,slicestar)
    
    return rhoA
#############################
# for the moment lets take the unitaries later we take the circuits as well
def get_training_vectors(paramslist):
    Uilist = []
    for parami in paramslist:
        circuiti,Ui = get_circuit_unitary_fromVQE(param=parami)
        Uilist.append(Ui)
    return Uilist
#####################

def get_phii_phij_fromcircuit(Ui,Uj,N=2):
    # circuiti,Ui = get_circuit_unitary_fromVQE(param=parami,N=N)
    # circuitj,Uj = get_circuit_unitary_fromVQE(param=paramj,N=N)
    
    psi_0_an  = np.zeros([2],dtype="complex")
    psi_0_an[0] = 1.0
    psi_0_an[1] = 1.0
    psi_0_an = psi_0_an/linalg.norm(psi_0_an)
    
    psi_0_sys = np.zeros([2**(N)],dtype="complex")    
    psi_0_sys[0] = 1.0
    
    #     step 0
    psi_0 = np.kron(psi_0_an,psi_0_sys)
    
    #   step 1
    Ujd = np.conjugate(np.transpose(Uj))
    U = np.matmul(Ujd,Ui)
    U_ansys = make_cntrlU(U=U)
    psi_1 = np.matmul(U_ansys,psi_0)
    
    # step 3
    rho_an = reducerho(wf = psi_1,lA=2,lB=2**N)
    
    return 2*rho_an[0,1]

def create_overlap_matrix_fromQC(Uilist,N=2):
    basis_length = len(Uilist)
    # params= []
    overlap_matrix = np.identity(basis_length,dtype="complex")
    for i in range(basis_length-1):
        for j in range(i+1,basis_length):
            # print(i,j)
            phij = get_phii_phij_fromcircuit(Ui=Uilist[i],Uj = Uilist[j],N=N)
            overlap_matrix[i,j] = phij
            overlap_matrix[j,i] = np.conjugate(phij)
     
    return overlap_matrix
##################################
def get_HPauli_ij_fromcircuit(Ui,Uj,paulisop,N=2):
    # circuiti,Ui = get_circuit_unitary_fromVQE(param=parami,N=N)
    # circuitj,Uj = get_circuit_unitary_fromVQE(param=paramj,N=N)
    
    H = many_kron(ops=paulisop)
    psi_0_an  = np.zeros([2],dtype="complex")
    psi_0_an[0] = 1.0
    psi_0_an[1] = 1.0
    psi_0_an = psi_0_an/linalg.norm(psi_0_an)
    
    psi_0_sys = np.zeros([2**(N)],dtype="complex")    
    psi_0_sys[0] = 1.0
    
    #     step 0
    psi_0 = np.kron(psi_0_an,psi_0_sys)
    
    #   step 1
    Ujd = np.conjugate(np.transpose(Uj))
    U = np.matmul(H,Ui)
    U = np.matmul(Ujd,U)
    U_ansys = make_cntrlU(U=U)
    psi_1 = np.matmul(U_ansys,psi_0)
    
    # step 3
    rho_an = reducerho(wf = psi_1,lA=2,lB=2**N)
    
    return 2*rho_an[0,1]

def get_HXY_bare_ij_fromcircuit( N, pbc,Ui,Uj):
    
    hamBz=0.0+0.0*1.j
    hamBx=0.0+0.0*1.j
    hamXX=0.0+0.0*1.j
    hamYY=0.0+0.0*1.j
    # Build hamiltonian matrix
    for isite in range(N):

        # BZ
        oplist = ['I']*N
        oplist[isite] = 'Z'
        #print("".join(oplist))
        hamij = get_HPauli_ij_fromcircuit(Ui=Ui,Uj=Uj,paulisop=oplist,N=N)
        hamBz += hamij

        # BX
        oplist = ['I']*N
        oplist[isite] = 'X'
        #print("".join(oplist))
        hamij = get_HPauli_ij_fromcircuit(Ui=Ui,Uj=Uj,paulisop=oplist,N=N)
        hamBx += hamij

        jsite = (isite + 1) % N
        if not(jsite == isite+1) and not pbc:
            continue

        # XX
        oplist = ['I']*N
        oplist[isite] = 'X'
        oplist[jsite] = 'X'
        #print("".join(oplist))
        hamij = get_HPauli_ij_fromcircuit(Ui=Ui,Uj=Uj,paulisop=oplist,N=N)
        hamXX += hamij
        # YY
        oplist = ['I']*N
        oplist[isite] = 'Y'
        oplist[jsite] = 'Y'
        #print("".join(oplist))
        hamij = get_HPauli_ij_fromcircuit(Ui=Ui,Uj=Uj,paulisop=oplist,N=N)
        hamYY += hamij

    return [hamBz,hamBx,hamXX,hamYY]

def get_HXY_together_ij_fromcircuit(paramn,Ui,Uj):
    J =paramn["J"]
    Bx = paramn["Bx"]
    Bz = paramn["Bz"]
    N = paramn["N"]
    pbc = paramn["pbc"]
    
    [hamBz,hamBx,hamXX,hamYY] = get_HXY_bare_ij_fromcircuit( N, pbc,Ui,Uj)
    
    hamij = Bz*hamBz + Bx*hamBx + J*(hamXX+hamYY)
    
    return hamij

def make_target_hamiltonian_fromQC(Uilist,paramn):
    base_length = len(Uilist)
    ham_target = np.identity(base_length,dtype="complex")
    for i in range(base_length):
        for j in range(i,base_length):
            hamij = get_HXY_together_ij_fromcircuit(paramn,Ui=Uilist[i],Uj=Uilist[j])
            ham_target[i,j] = hamij
            ham_target[j,i] = np.conjugate(hamij)        
 
    return ham_target
    
    
def get_evals_target_ham(Uilist,paramn):
    J =paramn["J"]
    Bx = paramn["Bx"]
    Bz = paramn["Bz"]
    N = paramn["N"]
    pbc = paramn["pbc"]
    overlap_matrix = create_overlap_matrix_fromQC(Uilist=Uilist,N=N)
    print("Overlap matrix mimic:\n" , overlap_matrix)
    smaller_ham = make_target_hamiltonian_fromQC(Uilist,paramn)
    print("Hamiltonian mimic:\n" ,smaller_ham)
    evals, evecs = linalg.eigh(smaller_ham,overlap_matrix)
    print("Evals circuit mimic: ",evals)
    return evals
    

def get_evals_targetlist_mimic(training_paramlist,target_paramlist ):
    evals_qc = np.zeros([len(target_paramlist),len(training_paramlist)],dtype=complex)
    Uilist = get_training_vectors(training_paramlist)
    # print("printing Uilist from mimic")
    # for u in Uilist:
    #     print(u)
    for ip,paramn in enumerate(target_paramlist):
        evals = get_evals_target_ham(Uilist,paramn)
        for k in range(len(training_paramlist)):
                evals_qc[ip,k] = evals[k]
    return evals_qc

