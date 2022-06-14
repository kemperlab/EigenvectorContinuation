import numpy as np
import matplotlib.pyplot as plt
from continuers import *
from qiskit.opflow import I, X, Y, Z

# pauli_x = np.array([[0,1],[1,0]],dtype=complex)
# pauli_y = np.array([[0,-1.j],[1.j,0]],dtype=complex)
# pauli_z = np.array([[1,0],[0,-1]],dtype=complex)
# pauli_I = np.array([[1,0],[0,1]],dtype=complex)

paulis = {}
paulis['X'] = np.array([[0,1],[1,0]],dtype=complex)
paulis['Y'] = np.array([[0,-1.j],[1.j,0]],dtype=complex)
paulis['Z'] = np.array([[1,0],[0,-1]],dtype=complex)
paulis['I'] = np.array([[1,0],[0,1]],dtype=complex)

def many_kron(ops):
    # Takes an array of Pauli characters and produces the tensor product
    op = paulis[ops[0]]
    if len(ops) == 1:
        return op

    for opj in ops[1:]:
        op = np.kron(op,paulis[opj])

    return op


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

def XY_hamiltonian_Qiskit(J, Bx, Bz, N, pbc):
    assert(N==2)
    hamiltonian = J*((X^X) + (Y^Y)) + Bz*((I^Z) + (Z^I)) + Bx*((I^X) + (X^I))
    return hamiltonian

def show_XY_spectrum(N,Bzmin,Bzmax,Bx):

    Bzlist = np.linspace(Bzmin,Bzmax,100)
    eval_stor = np.zeros([len(Bzlist),2**N])
    for iBz, Bz in enumerate(Bzlist):
        ham = XY_hamiltonian(J=-1,Bz=Bz,Bx=Bx,N=N,pbc=False)
        eval_stor[iBz,:] = np.linalg.eigvalsh(ham)

    fig, ax = plt.subplots()
    ax.set_xlabel("$B_z$")
    ax.set_ylabel("Energy")
    for j in range(2**N):
        ax.plot(Bzlist,eval_stor[:,j],'k-')
    return fig, ax



def dot_vectors(A,B):
    return np.conjugate(np.transpose(A)) @ B

def evaluate_op_vectors(A,B,C):
    return np.conjugate(np.transpose(A)) @ B @ C

def characterize_eigenspectrum(J=-1,Bz=0,Bx=0,N=8,pbc=True):

    ham = XY_hamiltonian(J,Bz,Bx,N,False)

    bcterm = ['Z']*N
    bcterm1 = bcterm
    bcterm1[0] = 'X'
    bcterm1[-1] = 'X'
    print(bcterm1)

    bcterm2 = bcterm
    bcterm1[0] = 'Y'
    bcterm1[-1] = 'Y'
    print(bcterm2)

    ham += many_kron(bcterm1)
    ham += many_kron(bcterm2)

    evals, evecs = np.linalg.eigh(ham)

    mztot_oplist = ['Z']*N
    mztot_op = many_kron(mztot_oplist)

    for k in range(2**N):
        energy = evals[k]
        evec = evecs[:,k]
        mz = np.real( np.conj(np.transpose(evec)) @ mztot_op @ evec )
        if abs(mz) < 0.01:
            print(mz,energy)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    fig, ax = show_XY_spectrum(2,0,2,0.05)

    #characterize_eigenspectrum(J=-1,Bz=0.000001,Bx=0,N=8,pbc=True)
    #bork()

    J = -1
    Bx = 0.05
    N = 2
    pbc = False

    # Set up training parameter sets for eigenvector continuer
    Bzlist = [0,0.2,0.75]
    training_paramlist = [[J,Bx,Bz,N,pbc] for Bz in Bzlist]

    if 'ax' in locals():
        for b in Bzlist:
            ax.axvline(b)

    # Set up target parameter sets for eigenvector continuer
    Bzlist = np.linspace(0,2,20)
    #Bzlist = [0.6]
    target_paramlist = [[J,Bx,Bz,N,pbc] for Bz in Bzlist]


    # Object that knows how to deal with the various operations needed
    vectorspace = vector_methods(XY_hamiltonian)

    # Reference vector is internal for now
    #vectorspace = unitary_methods(N, XY_hamiltonian)

    #vectorspace = circuit_methods(N,XY_hamiltonian_Qiskit)

    EVcontinuer = vector_continuer(vectorspace,
                                   XY_hamiltonian,
                                   training_paramlist,
                                   target_paramlist,
                                   N)

    EVcontinuer.get_base_eigenvectors()
    #EVcontinuer.form_orthogonal_basis()

    #added_evals = EVcontinuer.get_target_eigenvectors(ortho=True)
    added_evals = EVcontinuer.get_target_eigenvectors(ortho=False)

    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            ax.plot(Bzlist,np.real(added_evals[:,ip]),'o')


    plt.show()

