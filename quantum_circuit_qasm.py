from qiskit import Aer, execute, QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, SLSQP
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.opflow import I, X, Y, Z

from hamiltonian import *
from scipy import linalg

from qiskit.extensions import UnitaryGate


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
    
def get_circuit_list(training_paramlist):
    basis_circuits_list = []
    for param in training_paramlist:
        circuiti,Unitaryi = get_circuit_unitary_fromVQE(param=param)
        basis_circuits_list.append(circuiti)
    return basis_circuits_list

def measure_inbasis_circ(circ,N=2,basis="X"):
    q = QuantumRegister(N)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q,c)
    # print(circ)
    qc.compose(circ,inplace=True)
    
    if(basis=="X"):
        qc.h(q[0])
    elif(basis=="Y"):
        qc.rx(-np.pi/2,q[0])
        
    qc.measure(q[0],c[0])
    # print(qc)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend=backend,shots=8192)
    result = job.result()
    counts = result.get_counts()
    # qobj = execute(qc)
    
    
    try:
        p = (counts['0'] -counts['1'])/(counts['0'] + counts['1'])
    except:
        try:
            p = counts['0']/counts['0']
        except:
            p = -1
        
    return p
    
def measure_real_imaginary_from_circ(circ,N=2):
    px = measure_inbasis_circ(circ=circ,N=N,basis="X")
    py = measure_inbasis_circ(circ=circ,N=N,basis="Y")
    # py=0.0
    
    return px+1.j*py

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

def create_overlap_matrix_fromQC(basis_circuits_list,N=2):
    basis_length = len(basis_circuits_list)
    # params= []
    overlap_matrix = np.identity(basis_length,dtype="complex")
    # print(overlap_matrix)
    for i in range(basis_length-1):
        for j in range(i+1,basis_length):
            # print(i,j)
            circij = phi_ij(circi=basis_circuits_list[i],circj=basis_circuits_list[j],N=N)
            # print(circij.decompose().draw())
            phij = measure_real_imaginary_from_circ(circ=circij,N=N+1)
            # phij = get_phii_phij_fromcircuit(Ui=Uilist[i],Uj = Uilist[j],N=N)
            overlap_matrix[i,j] = phij
            overlap_matrix[j,i] = np.conjugate(phij)
            
            # break
     
    return overlap_matrix


# This takes in the basis circuit elements and a single Pauli String to return the contrlled circuits

def get_HPauli_ij_fromcircuit(circi,circj,paulisop,N=2):
    # circuiti,Ui = get_circuit_unitary_fromVQE(param=parami,N=N)
    # circuitj,Uj = get_circuit_unitary_fromVQE(param=paramj,N=N)
    
    H = many_kron(ops=paulisop)
    # print(  "H Pauli = ",H)
    
    circ_H = QuantumCircuit(N)
    circ_H.append(UnitaryGate(H),[0,1])
    gateH = circ_H.to_gate()
    cgateH  = gateH.control()
    
    gatei = circi.to_gate()
    cgatei = gatei.control()

    # gatej = circj.to_gate()
    gatej = circj.to_gate()
    gatejinv = gatej.inverse()
    cgatejinv = gatejinv.control()
    
    q = QuantumRegister(N+1)
    circ_Pij = QuantumCircuit(q)
    circ_Pij.h(q[0])
    circ_Pij.append(cgatei,[0,1,2])
    circ_Pij.append(cgateH,[0,1,2])
    circ_Pij.append(cgatejinv,[0,1,2])
    # print(circ_Pij.decompose().draw())
    # return circ_Pij
    
    p = measure_real_imaginary_from_circ(circ=circ_Pij,N=N+1)
    
    return p

def get_HXY_bare_ij_fromcircuit( N, pbc,circi,circj):
    
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
        hamij = get_HPauli_ij_fromcircuit(circi=circi,circj=circj,paulisop=oplist,N=N)
        hamBz += hamij

        # BX
        oplist = ['I']*N
        oplist[isite] = 'X'
        #print("".join(oplist))
        hamij = get_HPauli_ij_fromcircuit(circi=circi,circj=circj,paulisop=oplist,N=N)
        hamBx += hamij

        jsite = (isite + 1) % N
        if not(jsite == isite+1) and not pbc:
            continue

        # XX
        oplist = ['I']*N
        oplist[isite] = 'X'
        oplist[jsite] = 'X'
        #print("".join(oplist))
        hamij = get_HPauli_ij_fromcircuit(circi=circi,circj=circj,paulisop=oplist,N=N)
        hamXX += hamij
        # YY
        oplist = ['I']*N
        oplist[isite] = 'Y'
        oplist[jsite] = 'Y'
        #print("".join(oplist))
        hamij = get_HPauli_ij_fromcircuit(circi=circi,circj=circj,paulisop=oplist,N=N)
        hamYY += hamij

    return [hamBz,hamBx,hamXX,hamYY]

def get_HXY_together_ij_fromcircuit(paramn,circi,circj):
    J =paramn["J"]
    Bx = paramn["Bx"]
    Bz = paramn["Bz"]
    N = paramn["N"]
    pbc = paramn["pbc"]
    
    [hamBz,hamBx,hamXX,hamYY] = get_HXY_bare_ij_fromcircuit( N, pbc,circi,circj)
    # prntlst = [hamBz,hamBx,hamXX,hamYY] 
    # print(prntlst)
    hamij = Bz*hamBz + Bx*hamBx + J*(hamXX+hamYY)
    
    return hamij

def make_target_hamiltonian_fromQC(basis_circuits_list,paramn):
    base_length = len(basis_circuits_list)
    ham_target = np.identity(base_length,dtype="complex")
    for i in range(base_length):
        for j in range(i,base_length):
            hamij = get_HXY_together_ij_fromcircuit(paramn=paramn,circi=basis_circuits_list[i],circj=basis_circuits_list[j])
            ham_target[i,j] = hamij
            ham_target[j,i] = np.conjugate(hamij)        
 
    return ham_target

def get_evals_target_ham_qasmcirc(basis_circuits_list,paramn):
    
    J =paramn["J"]
    Bx = paramn["Bx"]
    Bz = paramn["Bz"]
    N = paramn["N"]
    pbc = paramn["pbc"]
   
    overlap_matrix = create_overlap_matrix_fromQC(basis_circuits_list=basis_circuits_list,N=N)
    
#     overlap_matrix[0,1] *= -1
#     overlap_matrix[1,0] *= -1
#     overlap_matrix[1,2] *= -1
#     overlap_matrix[2,1] *= -1
    
    # print("Overlap:")
    # print(overlap_matrix)
    
    #print(linalg.lu(overlap_matrix))
    
    smaller_ham = make_target_hamiltonian_fromQC(basis_circuits_list=basis_circuits_list,paramn=paramn)
    
    
#     smaller_ham[0,1] *= -1
#     smaller_ham[1,0] *= -1
#     smaller_ham[1,2] *= -1
#     smaller_ham[2,1] *= -1
    
    # print("Hamiltonian:")
    # print(smaller_ham)
    
    evals, evecs = linalg.eigh(smaller_ham,overlap_matrix)
    # evals, evecs = linalg.eigh(smaller_ham,overlap_matrix,driver="gv")
    # print("Evals: ",evals)
    return evals

def get_evals_targetlist_qasmcirc(training_paramlist,target_paramlist ):
    evals_qc = np.zeros([len(target_paramlist),len(training_paramlist)],dtype=complex)
    basis_circuits_list = get_circuit_list(training_paramlist)
    # Uilist = get_training_vectors(training_paramlist)
    for ip,paramn in enumerate(target_paramlist):
        evals = get_evals_target_ham_qasmcirc(basis_circuits_list,paramn)
        for k in range(len(training_paramlist)):
                evals_qc[ip,k] = evals[k]
    return evals_qc

