import numpy as np
import scipy
import qiskit as qk

class vector_methods():

    def __init__(self):
        pass

    def dot(self,A,B):
        return np.conjugate(np.transpose(A)) @ B

    def evaluate_operator(self,A,op,B):
        return np.conjugate(np.transpose(A)) @ op @ B

    def representation_from_vector(self,A):
        return A



class unitary_methods():


    def __init__(self,Nqubits):
        refvec = np.zeros(2**Nqubits,dtype=complex)
        refvec[0] = 1.

        self.Nqubits = Nqubits
        self.reference_vector = refvec
        self.backend = qk.Aer.get_backend('unitary_simulator')

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


class vector_continuer:

    def __init__(self,
                 vectorspace=None,
                 hamiltonian_function=None,
                 training_paramsets=[],
                 target_paramsets=[],
                 Nsites=1):

        self.vectorspace = vectorspace
        self.dot = vectorspace.dot
        self.operator_evaluator = vectorspace.evaluate_operator
        self.hamiltonian_function = hamiltonian_function
        self.Nsites = Nsites
        self.training_paramsets = training_paramsets
        self.target_paramsets = target_paramsets
        self.base_vecs = []

    def get_base_eigenvectors(self):


        for params in self.training_paramsets:

            ham = self.hamiltonian_function(*params)
            evals, evecs = np.linalg.eigh(ham)

            repr = self.vectorspace.representation_from_vector(evecs[:,0])

            self.base_vecs.append(repr)
            print("Adding vector for parameter set",params)
            #print("Check: energy should be",evals[0])
            #print(np.conjugate(np.transpose(evecs[:,0])) @ ham @ evecs[:,0])
            #print("")
        print("")

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
            overlap_matrix = np.zeros_like(smaller_ham)

        for i in range(nvecs):
            ui = self.ortho_vecs[i]
            for j in range(i,nvecs):
                uj = self.ortho_vecs[j]
                smaller_ham[i,j] = self.operator_evaluator(ui,ham,uj)

                if not i == j:
                    smaller_ham[j,i] = np.conjugate(smaller_ham[i,j])

                if not ortho:
                    overlap_matrix[i,j] = self.dot(ui,uj)

                    if not i == j:
                        overlap_matrix[j,i] = np.conjugate(overlap_matrix[i,j])


        if ortho:   # Solve the straight up eigenvalue problem
            evals, evecs = np.linalg.eigh(smaller_ham)
        else:       # Solve the generalized eigenvalue problem
            evals, evecs = scipy.linalg.eigh(smaller_ham,overlap_matrix)


        return evecs

    def get_target_eigenvectors(self,ortho):

        nvec = 0
        basis = None
        if ortho:
            nvec = len(self.ortho_vecs)
            basis = self.ortho_vecs
        else:
            nvec = len(self.base_vecs)
            basis = self.base_vecs



        new_evals = np.zeros([len(self.target_paramsets),nvec],dtype=complex)

        for ip,param in enumerate(self.target_paramsets):
            ham = self.hamiltonian_function(*param)

            evecs = self.do_continuation(ham,ortho=ortho)

            for k in range(nvec):
                fullvec = np.zeros_like(basis[0])
                for l in range(nvec):
                    fullvec += evecs[l,k] * basis[l]
                energy = self.operator_evaluator(fullvec,ham,fullvec)
                new_evals[ip,k] = energy

        return new_evals









