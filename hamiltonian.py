import numpy as np
import matplotlib.pyplot as plt


# pauli_x = np.array([[0,1],[1,0]],dtype=complex)
# pauli_y = np.array([[0,-1.j],[1.j,0]],dtype=complex)
# pauli_z = np.array([[1,0],[0,-1]],dtype=complex)
# pauli_I = np.array([[1,0],[0,1]],dtype=complex)

paulis = {}
paulis['X'] = np.array([[0,1],[1,0]],dtype=complex)
paulis['Y'] = np.array([[0,-1.j],[1.j,0]],dtype=complex)
paulis['Z'] = np.array([[1,0],[0,-1]],dtype=complex)
paulis['I'] = np.array([[1,0],[0,1]],dtype=complex)

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

# def XY_hamiltonian_Qiskit(J, Bx, Bz, N, pbc):
#     assert(N==2)
#     hamiltonian = J*((X^X) + (Y^Y)) + Bz*((I^Z) + (Z^I)) + Bx*((I^X) + (X^I))
#     return hamiltonian

def show_XY_spectrum(N,Bzmin,Bzmax,Bx,J,pbc):

    Bzlist = np.linspace(Bzmin,Bzmax,100)
    eval_stor = np.zeros([len(Bzlist),2**N])
    for iBz, Bz in enumerate(Bzlist):
        ham = XY_hamiltonian(J=J,Bz=Bz,Bx=Bx,N=N,pbc=pbc)
        eval_stor[iBz,:] = np.linalg.eigvalsh(ham)

    fig, ax = plt.subplots()
    ax.set_xlabel("$B_z$")
    ax.set_ylabel("Energy")
    for j in range(2**N):
        ax.plot(Bzlist,eval_stor[:,j],'k-')
    return fig, ax, Bzlist, eval_stor

def show_XY_magnetization(N,Bzmin,Bzmax,Bx,J,pbc):

    Bzlist = np.linspace(Bzmin,Bzmax,100)
    mag_stor = np.zeros([len(Bzlist),2**N])
    # mztot_oplist = ['Z'] * N
    # mztot_op = many_kron(mztot_oplist)
    mztot_op =  Mag_op(N=2)
    for iBz, Bz in enumerate(Bzlist):
        ham = XY_hamiltonian(J=J,Bz=Bz,Bx=Bx,N=N,pbc=pbc)
        evals, evecs = np.linalg.eigh(ham)
        mzarray = np.zeros(2**N)
        for i in range(2**N):
            mz = np.real(np.conj(np.transpose(evecs[:,i])) @ mztot_op @ evecs[:,i])
            mzarray[i] = mz
        mag_stor[iBz,:] = mzarray

    fig, ax = plt.subplots()
    ax.set_xlabel("$B_z$")
    ax.set_ylabel("Magnetization")
    for j in range(2**N):
        ax.plot(Bzlist,mag_stor[:,j],'k-')
    return fig, ax, Bzlist, mag_stor
#
# def dot_vectors(A,B):
#     return np.conjugate(np.transpose(A)) @ B
#
# def evaluate_op_vectors(A,B,C):
#     return np.conjugate(np.transpose(A)) @ B @ C
#
def Mag_op(N):

    Mag = np.zeros([2**N,2**N],dtype=complex)

    # Build hamiltonian matrix
    for isite in range(N):

        # BZ
        oplist = ['I']*N
        oplist[isite] = 'Z'
        #print("".join(oplist))
        Mag +=many_kron(oplist)

    return Mag

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




def many_kron(ops):
    # Takes an array of Pauli characters and produces the tensor product
    op = paulis[ops[0]]
    if len(ops) == 1:
        return op

    for opj in ops[1:]:
        op = np.kron(op,paulis[opj])

    return op


def XXZ_hamiltonian(J, Jz, Bx, Bz, N, pbc):
    ham = np.zeros([2 ** N, 2 ** N], dtype=complex)

    # Build hamiltonian matrix
    for isite in range(N):

        # BZ
        oplist = ['I'] * N
        oplist[isite] = 'Z'
        # print("".join(oplist))
        ham += Bz * many_kron(oplist)

        # BX
        oplist = ['I'] * N
        oplist[isite] = 'X'
        # print("".join(oplist))
        ham += Bx * many_kron(oplist)

        jsite = (isite + 1) % N
        if not (jsite == isite + 1) and not pbc:
            continue

        # XX
        oplist = ['I'] * N
        oplist[isite] = 'X'
        oplist[jsite] = 'X'
        # print("".join(oplist))
        ham += J * many_kron(oplist)

        # YY
        oplist = ['I'] * N
        oplist[isite] = 'Y'
        oplist[jsite] = 'Y'
        # print("".join(oplist))
        ham += J * many_kron(oplist)

        # ZZ
        oplist = ['I'] * N
        oplist[isite] = 'Z'
        oplist[jsite] = 'Z'
        # print("".join(oplist))
        ham += -1*Jz * many_kron(oplist)

    return ham
def show_XXZ_spectrumwithJ(Jmin, Jmax, Jz=-1, Bx=0.0, Bz=0.0, pbc=False, N=2):
    Jlist = np.linspace(Jmin, Jmax, 50)
    # N=2
    # pbc= False
    eval_stor = np.zeros([len(Jlist), 2 ** N])

    for iJ, J in enumerate(Jlist):
        # ham = XXZ_hamiltonian(J=J, Jz=-1, Bx=0.0, Bz=0.0, N=N, pbc=pbc)
        ham = XXZ_hamiltonian(J=J, Jz=Jz, Bx=Bx, Bz=Bz, N=N, pbc=pbc)
        eval_stor[iJ, :] = np.linalg.eigvalsh(ham)

    fig, ax = plt.subplots()
    ax.set_xlabel("$J$")
    ax.set_ylabel("Energy")
    for j in range(2 ** N):
        ax.plot(Jlist, eval_stor[:, j], 'k-')
    return fig, ax, Jlist, eval_stor

def show_XXZ_spectrumwithJz(Jzmin, Jzmax, J=-1, Bx=0.0, Bz=0.0, pbc=False, N=2):
    Jzlist = np.linspace(Jzmin, Jzmax, 100)
    # N=2
    # pbc= False
    eval_stor = np.zeros([len(Jzlist), 2 ** N])

    for iJ, Jz in enumerate(Jzlist):
        # ham = XXZ_hamiltonian(J=J, Jz=-1, Bx=0.0, Bz=0.0, N=N, pbc=pbc)
        ham = XXZ_hamiltonian(J=J, Jz=Jz, Bx=Bx, Bz=Bz, N=N, pbc=pbc)
        eval_stor[iJ, :] = np.linalg.eigvalsh(ham)

    fig, ax = plt.subplots()
    ax.set_xlabel("$J_z$")
    ax.set_ylabel("Energy")
    for j in range(2 ** N):
        ax.plot(Jzlist, eval_stor[:, j], 'k-')
    return fig, ax, Jzlist, eval_stor
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

# def show_XY_spectrum(N,Bzmin,Bzmax,Bx):

#     Bzlist = np.linspace(Bzmin,Bzmax,100)
#     eval_stor = np.zeros([len(Bzlist),2**N])
#     for iBz, Bz in enumerate(Bzlist):
#         ham = XY_hamiltonian(J=-1,Bz=Bz,Bx=Bx,N=N,pbc=False)
#         eval_stor[iBz,:] = np.linalg.eigvalsh(ham)

#     fig, ax = plt.subplots()
#     ax.set_xlabel("$B_z$")
#     ax.set_ylabel("Energy")
#     for j in range(2**N):
#         ax.plot(Bzlist,eval_stor[:,j],'k-')
#     return fig, ax

# ####################

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