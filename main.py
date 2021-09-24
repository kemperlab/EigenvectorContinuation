import numpy as np
import matplotlib.pyplot as plt
from continuers import *

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

def evaluate_op(A,B,C):
    return np.conjugate(np.transpose(A)) @ B @ C

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    fig, ax = show_XY_spectrum(4,0,2,0.05)

    J = -1
    Bx = 0.05
    N = 4
    pbc = False

    # Set up training parameter sets for eigenvector continuer
    Bzlist = [0,0.2,0.5,0.75]
    training_paramlist = [[J,Bx,Bz,N,pbc] for Bz in Bzlist]

    if 'ax' in locals():
        for b in Bzlist:
            ax.axvline(b)

    # Set up target parameter sets for eigenvector continuer
    Bzlist = np.linspace(0,2,20)
    #Bzlist = [0.6]
    target_paramlist = [[J,Bx,Bz,N,pbc] for Bz in Bzlist]





    EVcontinuer = vector_continuer(dot_vectors,
                                   evaluate_op,
                                   XY_hamiltonian,
                                   training_paramlist,
                                   target_paramlist,
                                   4)

    EVcontinuer.get_base_eigenvectors()
    EVcontinuer.form_orthogonal_basis()

    #added_evals = EVcontinuer.get_target_eigenvectors(ortho=True)

    added_evals = EVcontinuer.get_target_eigenvectors(ortho=False)

    if 'ax' in locals():
        for ip in range(len(training_paramlist)):
            ax.plot(Bzlist,np.real(added_evals[:,ip]),'o')


    plt.show()


