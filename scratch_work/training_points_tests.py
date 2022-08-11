#%%

"""creates a plot and matrix that correspond to a simplified model of a hamiltonian
    found from eigenvector continuation.
    @author Lex Kemper, Akhil Francis, & Jack Howard
    North Carolina State University -- Kemper Lab
"""
import random
import math
from collections import namedtuple
import numpy as np
# from numpy import ndarray
from matplotlib import pyplot as plt




class HamiltonianInitializer:
    """ initializes the hamiltonian """

    PAULIS = {}
    """ defines dict of Paulis to use below """

    DATA_POINTS = 100
    """ determines fineness of curves """

    BzCoord = namedtuple("BzCoord", "j_x j_z b_x b_z")
    """" useful tuple when dealing with coordinates in this space """

    COMP_TOLERANCE = 1e-9
    """ tolernace when comparing two floats """

    def __init__(self):
        """ initializes class instance and Paulis dict """
        self.PAULIS['X'] = np.array([[0,1],[1,0]], dtype=complex)
        self.PAULIS['Y'] = np.array([[0,-1.j],[1.j,0]], dtype=complex)
        self.PAULIS['Z'] = np.array([[1,0],[0,-1]], dtype=complex)
        self.PAULIS['I'] = np.array([[1,0], [0,1]], dtype=complex)

    def many_kron(self, ops):
        """ produces Kronecker (Tensor) product from list of Pauli charaters """
        result = self.PAULIS[ops[0]]    # set result equal to the first pauli given by the parameter
        if len(ops) == 1:
            return result

        for opj in ops[1:]:             # for all the operations in the parameter
            result = np.kron(result, self.PAULIS[opj])  # tensor product the current matrix with
                                                        # the next pauli in the parameter list
        return result

    def xxztype_hamiltonian(self, coord, n, pbc):
        """ produces the hamiltonian for a system where j_x = j_y and b_x = b_y
            :param j_x:     the x and y component of j
            :param j_z:     the z component of j
            :param b_x:     he x and y component of b
            :param b_z:     the z component of b
            :param n:       the number of quibits
            :param pbc:     whether or not to include periodic boundary condition wrap around logic
            :returns:       hamiltonian of the system
        """

        j_x = coord.j_x
        j_z = coord.j_z
        b_x = coord.b_x
        b_z = coord.b_z

        ham = np.zeros([2**n, 2**n], dtype=complex) # initializes the hamiltonian

        # build hamiltonian matrix
        for isite in range(n):

            # Apply the Bz information to the hamiltonian matrix
            oplist = ['I']*n                # makes list of operators (default = identity matrix)
            oplist[isite] = 'Z'             # sets the isite-th entry to Z
            ham += b_z * self.many_kron(oplist)  # applies the operations specified to the ham

            # Apply the Bx information to the hamiltonian matrix
            oplist = ['I']*n                # makes list of operators (default = identity matrix)
            oplist[isite] = 'X'             # sets the isite-th entry to X
            ham += b_x * self.many_kron(oplist)  # applies the operations specified to the ham

            # checks whether to apply wrap-around rules
            jsite = (isite + 1) % n 
            if (jsite != isite + 1 ) and not pbc:
                continue                            # skips the XX, YY, ZZ

            # Apply the XX information to the hamiltonian
            oplist = ['I']*n                # makes list of operators (default = identity matrix)
            oplist[isite] = 'X'             # sets the isite-th entry to X
            oplist[jsite] = 'X'             # sets the jsite-th entry to X
            ham += j_x * self.many_kron(oplist)  # applies the operations specified to ham

            # Apply the YY information to the hamiltonian
            oplist = ['I']*n                # makes list of operators (default = identity matrix)
            oplist[isite] = 'Y'             # sets the isite-th entry to Y
            oplist[jsite] = 'Y'             # sets the jsite-th entry to Y
            ham += j_x * self.many_kron(oplist)  # applies the operations specified to ham

            # Apply the Z information to the hamiltonian
            oplist = ['I']*n                # makes list of operators (default = identity matrix)
            oplist[isite] = 'Z'             # sets the isite-th entry to Z
            oplist[jsite] = 'Z'             # sets the jsite-th entry to Z
            ham += j_z * self.many_kron(oplist)  # applies the operations specified to ham

        return ham

    def get_random_training_points(self, bzlist, evals, num_points):
        """ returns random training points for the system
            NOTE: This is a simplified case of "get training points".
            It only accounts for Bz, but later updates will be more robust by exploring more
            degrees of freedom
        """
            # TODO This is a simplified case of "get training points".
            # It only accounts for Bz, but later updates will be more robust by exploring more
            # degrees of freedom

        # initialize keys to keep track of randomness
        keys = [0] * len(evals)

        # named tuple used to organize points
        Point = namedtuple("Point", "b_z energies")

        points = [None] * num_points
        for i in range(num_points):

            keys[i] = random.randrange(0, len(bzlist) - 1)

            point = Point(b_z=bzlist[keys[i]], energies=evals[keys[i]])

            points[i] = point
            # first index is Bz values; second index is energy values of different states 

        return points

    def get_eigenpairs(self, coord, domain, n, pbc):
        """ gets the eigenpairs for a given coordinate in a system"""

        evals = np.zeros([self.DATA_POINTS, 2**n])

        for bz_index, b_z in enumerate(domain): # populate evals
            b_zcoord = self.BzCoord(j_x=coord.j_x, j_z=coord.j_z, b_x=coord.b_x, b_z=b_z)

            # creates the hamiltonian for the current system
            ham = self.xxztype_hamiltonian(b_zcoord, n=n, pbc=pbc)

            evals[bz_index,:], evecs = np.linalg.eigh(ham)

        return evals, evecs

    def get_interaction_matrix(self, evecs, num_points):
        """ gets the interaction matrix for a given system """

        # set up interaction matrix
        s = np.zeros([num_points, num_points], dtype=complex)
        # simple case: 2 qubits

        # TODO Ask Kemper how to make this work for more qubits
        # TODO also, is this even right? 
        #       this iterates over all evecs but don't I only want the ground state?
        for i in range(num_points):
            for j in range(num_points):
                vector1 = np.matrix(evecs[:,i])
                vector2 = np.matrix(evecs[:,j])

                vector2 = vector2.conj().T
                s[i,j] = vector1 @ vector2
                # print(s)

        return s

    def compare_sets_of_points(self, set_a, set_b):
        """ return true if any overlap in b_z values"""
        # test for redundancy
        for val_a in set_a:
            for val_b in set_b:
                if math.isclose(val_a.b_z,val_b.b_z,rel_tol=self.COMP_TOLERANCE):
                    return True
        return False

    def calc_subspace_ham(self, hams, evecs, num_points): #call evecs basis vectors TODO
        """ calculates the hamiltonian for the subspace of the system """

        new_hams = [None] * len(hams)
        # print(num_points)

        for idx_k, ham_k in enumerate(hams):
            new_ham = np.zeros([num_points, num_points], dtype=complex)
            for i in range(num_points):
                for j in range(num_points):
                    vector1 = np.matrix(evecs[:,i])
                    vector2 = np.matrix(evecs[:,j])

                    vector2 = vector2.conj().T
                    new_ham[i,j] = vector1 @ ham_k @ vector2
            new_hams[idx_k] = new_ham
            # print(new_ham)

        return new_hams

    def diagonalize_ham(self, ham):
        """ returns the eigenvalues of the diagonalized hamoltonian 
            
            this is its own function because while this is a very simple return statement here,
            future abstractions of this funtion will be less simple, but still follow this template
        """
        
        return np.linalg.eigvalsh(ham) # TODO output evecs as well

    def plot_xxz_spectrum(self, bzlist, evals, points, phats, n):
        """ plots the spectrum along with training points and phats """

        ax = plt.subplots()[1]    # initializes axes
        ax.set_xlabel("$B_Z$")
        ax.set_ylabel("Energy")

        for idx in range(2**n):                       # prepares plot
            ax.plot(bzlist, evals[:,idx], 'k-')

        ax.axvline(1.0, ls = "--", color="blue")    # shows vertical line that represents [unsure] TODO

        # plot training points. i[1][0] corresponds to the lowest energy 
        for point in points:
            plt.plot(point.b_z, point.energies[0], marker="o", color="blue")

        # plot phats
        for phat in phats:
            plt.plot(phat.b_z, phat.energies[0], marker="o", color="orange")
            plt.plot(phat.b_z, phat.energies[1], marker="o", color="orange")
            # TODO only does lowest 2 energy states. can make this more flexible later          

        plt.show()

    def generate_xxz_type_spectrum(self, coord, n=2, pbc=False):
        """ calculates the different hamiltonians, eigenvalues, and interaction matrix for the system
            and plots the spectrum on a plot"""

        # j_x = coord.j_x
        # j_z = coord.j_z
        # b_x = coord.b_x
        # bzmin = coord.bzmin
        # bzmax = coord.bzmax

        # Initialize:
        # plotting tool: array of evenly spaced numbers between bzmin and bzmax
        bzlist = np.linspace(coord.bzmin, coord.bzmax, self.DATA_POINTS)
        evals, evecs = self.get_eigenpairs(coord, bzlist, n, pbc)

        # Training Points:
        # getting n random training points
        num_points = n
        points = self.get_random_training_points(bzlist=bzlist, evals=evals, num_points=num_points)

        # Phats:
        # set up new points to diagonalize from: denoted phat
        # num_phats = n # Instead, make all point selection done in main
        while True:
            phats = self.get_random_training_points(bzlist=bzlist, evals=evals, num_points=n)

            if not self.compare_sets_of_points(points, phats):
                break
        # sort phats in ascending order
        phats.sort()

        # Calculate Hams:
        # list of hamiltonians, one for each phat_k in phats
        hams = [None] * len(phats)

        # find the hamiltonian for each phat_k
        for idx_k, phat_k in enumerate(phats):
            b_zcoord = self.BzCoord(j_x=coord.j_x, j_z=coord.j_z, b_x=coord.b_x, b_z=phat_k.b_z)
            hams[idx_k] = self.xxztype_hamiltonian(coord=b_zcoord, n=n, pbc=pbc)

        # Interaction Matrix:
        # get the S interaction matrix
        s = self.get_interaction_matrix(evecs=evecs, num_points=num_points)

        # Subspace Hams:
        # get the subspace hamiltonian for each phat_k value
        new_hams = self.calc_subspace_ham(hams=hams, evecs=evecs, num_points=num_points)
        # print(new_hams)

        # Diagonalization:
        energy_lists = [None] * len(hams) # use np.array
        # get eigenvals corresponding to each energy level for every hamiltonian
        for idx_k, new_ham_k in enumerate(new_hams):
            energy_lists[idx_k] = self.diagonalize_ham(new_ham_k)
            print("Iteraction",idx_k)
            print("New ham:")
            print(new_ham_k)
            print("Diagonalized:")
            print(energy_lists[idx_k]) # TODO ask Dr. Kemper what to do with this info
            print()

            # Check:
            # checks to see if Inverse_Interaction • New_Ham = Eigenvals
            print("S_inv • New ham (should correspond to Diagonalized value")
            print(np.linalg.inv(s) @ new_ham_k)

        self.plot_xxz_spectrum(bzlist, evals, points, phats, n)
        

    
def main():
    """ generates the image, hamiltonian, and overlap matrix """
    
    bzmin = 0.0
    bzmax = 3
    b_x = 0
    j_x = -1.0
    j_z = 0
    n = 4
    pbc = False

    Coord = namedtuple("Coord", "j_x j_z b_x bzmin bzmax") # TODO think of how this could expand with general coordinates
    coord = Coord(j_x, j_z, b_x, bzmin, bzmax)
    

    hamiltonian_initializer = HamiltonianInitializer()
    hamiltonian_initializer.generate_xxz_type_spectrum(coord, n=n, pbc=pbc)

if __name__ == "__main__":
    main()
# %%
# shouldn't need n, evals, 
#  input: evecs, some ham of some osrt