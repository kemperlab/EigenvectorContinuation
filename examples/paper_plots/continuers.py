import numpy as np
import scipy
import qiskit as qk

from qiskit import Aer, execute, QuantumCircuit
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, SLSQP
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.opflow import I, X, Y, Z

class vector_methods():

    def __init__(self,hamiltonian_function):
        self.hamiltonian_function = hamiltonian_function
        pass

    def generate_training_state(self,params):

        ham = self.hamiltonian_function(*params)
        evals, evecs = np.linalg.eigh(ham)
        repr = self.representation_from_vector(evecs[:,0])
        return repr


    def dot(self,A,B):
        return np.conjugate(np.transpose(A)) @ B

    def evaluate_operator(self,A,op,B):
        return np.conjugate(np.transpose(A)) @ op @ B

    def representation_from_vector(self,A):
        return A



class unitary_methods():


    def __init__(self,Nqubits, hamiltonian_function):
        refvec = np.zeros(2**Nqubits,dtype=complex)
        refvec[0] = 1.

        self.hamiltonian_function = hamiltonian_function
        self.Nqubits = Nqubits
        self.reference_vector = refvec
        self.backend = qk.Aer.get_backend('unitary_simulator')


    def generate_training_state(self,params):
        ham = self.hamiltonian_function(*params)
        evals, evecs = np.linalg.eigh(ham)
        repr = self.representation_from_vector(evecs[:,0])
        return repr

    def dot(self,A,B):
        return np.conjugate(np.transpose(A @ self.reference_vector)) @ \
               (B @ self.reference_vector)

    def evaluate_operator(self,A,op,B):
        return np.conjugate(np.transpose(A @ self.reference_vector)) @ \
               op @ \
               (B @ self.reference_vector)

    def representation_from_vector(self,A):
        circuit = qk.QuantumCircuit(self.Nqubits)
        circuit.initialize(A, circuit.qubits)
        job = qk.execute(circuit, self.backend)
        result = job.result()
        return result.get_unitary(circuit, decimals=8)


class circuit_methods():
    def __init__(self,N,hamiltonian_function):
        self.hamiltonian_function = hamiltonian_function
        self.nqubits = N
        pass

    def generate_training_state(self,params):
        """ Given a set of parameters (and the internal representation
        of the Hamiltonian, produce the internal representation of the
        resulting eigenstate

        In this case, the internal rep is a Qiskit circuit "GATE" structure
        """
        self.get_unitary(self.hamiltonian_function(*params))

    def get_unitary(self,hamiltonian):
        print(hamiltonian)
        initial = [ 4.13850621,  4.68105177,  0.10459295,  5.19511699, -3.60948918, 2.08480452, -3.83679895,  1.9264198 ]
        #initial = None

        seed = 50
        algorithm_globals.random_seed = seed
        qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

        ansatz = TwoLocal(self.nqubits,rotation_blocks='ry', entanglement_blocks='cz')

        slsqp = SLSQP(maxiter=1000)
        spsa = SPSA(maxiter=1000)
        vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi, initial_point=initial)
        VQEresult = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
        print("VQE found eigenvalue:",VQEresult.eigenvalue)
        #print("I expected:",exact(J,Bz,Bx))

        cc = vqe.get_optimal_circuit()
        return cc

    def dot(self,A,B):
        raise NotImplementedError

    def evaluate_operator(self,A,op,B):

        # Stuff to do:
        # Implement ctrl-UA, ctrl-H, ctrl-UBinv

        raise NotImplementedError




class vector_continuer:

    def __init__(self,
                 vectorspace=None,
                 hamiltonian_function=None,
                 Mag_op = None,
                 training_paramsets=[],
                 target_paramsets=[],
                 Nsites=1):

        self.hamiltonian_function = hamiltonian_function
        self.vectorspace = vectorspace
        self.dot = vectorspace.dot
        self.operator_evaluator = vectorspace.evaluate_operator
        self.Nsites = Nsites
        self.training_paramsets = training_paramsets
        self.target_paramsets = target_paramsets
        self.base_vecs = []
        self.Mag_op = Mag_op(N=Nsites)
        self.target_evals = []
        self.target_magnetization = []
        self.LCU_coeffs_list = []
        self.target_full_evecs = []

    def get_base_eigenvectors(self):


        for params in self.training_paramsets:

            print(params)
            repr = self.vectorspace.generate_training_state(params)

            self.base_vecs.append(repr)
            print("Adding vector for parameter set",params)

        print("")
        # return self.base_vecs
    def form_orthogonal_basis(self):
        print("Orthogonalizing basis")
        if len(self.base_vecs) == 0:
            print("No base vectors to start from! Call get_base_eigenvectors first")

        # Use dumb Gram-Schmidt
        self.ortho_vecs = []
        self.ortho_vecs.append(self.base_vecs[0])

        nvecs = len(self.base_vecs)
        for i in range(1,nvecs):

            newvec = self.base_vecs[i]

            for j in range(0,i):
                uj = self.ortho_vecs[j]
                newvec -= self.dot(newvec,uj)/self.dot(uj,uj) * uj

            self.ortho_vecs.append(newvec)

        # Normalize all vectors
        for v in self.ortho_vecs:
            vnorm = np.real(self.dot(v,v))
            assert(np.sqrt(vnorm) > 1e-8)
            v /= np.sqrt(vnorm)

        # Check orthogonality
        for i in range(nvecs):
            u = self.ortho_vecs[i]
            for v in self.ortho_vecs[i+1:]:
                dotproduct = self.dot(u,v)
                assert abs(dotproduct) < 1e-12

        print("")

    def do_continuation(self,ham,ortho):

        nvecs = len(self.base_vecs)

        smaller_ham = np.zeros([nvecs,nvecs],dtype=complex)
        overlap_matrix = None
        if not ortho:
            basis = self.base_vecs
            overlap_matrix = np.zeros_like(smaller_ham)
        else:
            basis = self.ortho_vecs

        for i in range(nvecs):
            ui = basis[i]
            for j in range(i,nvecs):
                uj = basis[j]
                smaller_ham[i,j] = self.operator_evaluator(ui,ham,uj)

                if not i == j:
                    smaller_ham[j,i] = np.conjugate(smaller_ham[i,j])

                if not ortho:
                    overlap_matrix[i,j] = self.dot(ui,uj)

                    if not i == j:
                        overlap_matrix[j,i] = np.conjugate(overlap_matrix[i,j])
        # # uncomment
        # print("overlap_matrix_continuer:\n ", overlap_matrix)
        # print("Hamiltonian_continuer:\n ", smaller_ham)
        if ortho:   # Solve the straight up eigenvalue problem
            evals, evecs = np.linalg.eigh(smaller_ham)
        else:       # Solve the generalized eigenvalue problem
            evals, evecs = scipy.linalg.eigh(smaller_ham,overlap_matrix)
        
        # print("overlap_matrix_continuer:\n ",overlap_matrix)
        # print("Hamiltonian_continuer:\n ",smaller_ham)
        # return evecs
        return evals,evecs

    def get_target_eigenvectors(self,ortho):

        nvec = 0
        basis = None
        if ortho:
            nvec = len(self.ortho_vecs)
            basis = self.ortho_vecs
        else:
            nvec = len(self.base_vecs)
            basis = self.base_vecs

# # uncomment
#         print("basis continuer:")
#         for i in range(len(basis)):
#             print(basis[i])

        new_evals = np.zeros([len(self.target_paramsets),nvec],dtype=complex)
        mag_evals =  np.zeros([len(self.target_paramsets),nvec],dtype=complex)
        LCU_coeff_list = []
        for ip,param in enumerate(self.target_paramsets):
            ham = self.hamiltonian_function(*param)
            # evecs = self.do_continuation(ham,ortho=ortho)
            evals,evecs = self.do_continuation(ham,ortho=ortho)
            LCU_coeff_list.append(evecs[:,0])
            for k in range(nvec):
                fullvec = np.zeros_like(basis[0])
                for l in range(nvec):
                    fullvec += evecs[l,k] * basis[l]
                self.target_full_evecs.append(fullvec)
                # # uncomment
                # if(k==0):
                #     print("fullvec")
                #     print(fullvec)
                energy = self.operator_evaluator(fullvec,ham,fullvec)
                # new_evals[ip,k] = energy
                new_evals[ip, k] = evals[k]
                mag = self.operator_evaluator(fullvec, self.Mag_op, fullvec)
                mag_evals[ip,k] = mag
        self.target_evals = new_evals
        self.target_magnetization = mag_evals
        self.LCU_coeffs_list = LCU_coeff_list

        # return new_evals,basis,mag_evals









