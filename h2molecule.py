from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureMoleculeDriver, ElectronicStructureDriverType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
# from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
import numpy as np
from matplotlib import pyplot as plt
def h2_hamiltonian(dist):
    h2temp = h2molecule(dist=dist)
    h2temp.make_ham_matrix()
    return h2temp.ham

def show_h2_spectrum(distzmin=0.1, distmax=2.0,npoints=20):
    nuclear_repulsion_energires = []
    N=2
    distlist = np.linspace(distzmin,distmax,npoints)
    eval_stor = np.zeros([len(distlist),2**N])
    eval_stor_tot = np.zeros([len(distlist), 2 ** N])
    for ir, r in enumerate(distlist):
        h2temp = h2molecule(dist=r)
        h2temp.make_ham_matrix()
        eval_stor[ir,:] = np.linalg.eigvalsh(h2temp.ham)
        h2temp.solve_for_energies()
        nuclear_repulsion_energires.append(h2temp.nuclear_repulsion_energy)
        eval_stor_tot[ir,:] = eval_stor[ir,:] + h2temp.nuclear_repulsion_energy
    groundeneries = eval_stor_tot[:,0]
    equ_index = np.argmin(groundeneries)
    # for ir, r in enumerate(distlist):
    #     print("diff_energires in fn ")
    #     print(eval_stor_tot[ir,:]-eval_stor[ir,:])
    # print("nuclear_repulsion_energires in fn ", nuclear_repulsion_energires)
    fig, ax = plt.subplots(2,sharex=True)
    fig.subplots_adjust(hspace=.0)
    ax[1].set_xlabel("$distance$")
    ax[0].set_ylabel("Electronic Energy")
    ax[1].set_ylabel("Toatal Energy")
    for j in range(2**N):
        ax[0].plot(distlist, eval_stor[:,j],'k-')
        ax[1].plot(distlist, eval_stor_tot[:, j] , 'r-')
    ax[1].plot(distlist, groundeneries, 'b-')
    # ax[0].axvline(distlist[equ_index],color = "brown",ls="--")
    # ax[1].axvline(distlist[equ_index], color="brown", ls="--")
    print("Equillibrium dist ~ ",distlist[equ_index])
    return fig, ax, distlist, eval_stor, eval_stor_tot


class h2molecule:
    def __init__(self, dist):
        self.dist = dist

    def make_ham_matrix(self):
        molecule = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, self.dist]]], charge=0,
                            multiplicity=1)
        driver = ElectronicStructureMoleculeDriver(molecule, basis="sto-3g",
                                                   driver_type=ElectronicStructureDriverType.PYSCF)

        es_problem = ElectronicStructureProblem(driver)
        second_q_op = es_problem.second_q_ops()

        qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
        qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles)
        self.ham = qubit_op.to_matrix()

    def get_ham_coeffs_opslist(self):
        coeffs = []
        opslist = []
        for it in qubit_op.__dict__['_primitive'].to_list():
            # print(it[0],it[1])
            coeffs.append(it[1])
            op = []
            optog = it[0]
            for i in range(len(it[0])):
                op.append(str(optog[i]))
            opslist.append(op)
        self.coeffs = coeffs
        self.opslist = opslist

    def solve_for_energies(self):
        molecule = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, self.dist]]], charge=0,
                            multiplicity=1)
        driver = ElectronicStructureMoleculeDriver(molecule, basis="sto-3g",
                                                   driver_type=ElectronicStructureDriverType.PYSCF)

        es_problem = ElectronicStructureProblem(driver)
        second_q_op = es_problem.second_q_ops()

        qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
        qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles)
        numpyfactory = NumPyMinimumEigensolverFactory()
        numpysolver = GroundStateEigensolver(qubit_converter, numpyfactory)
        result = numpysolver.solve(es_problem)
        # print(result)
        energies = result.eigenenergies
        # print(energies)
        totalenergies = result.total_energies
        # print(totalenergies)
        # print(result.nuclear_repulsion_energy)
        # print(result.electronic_energies)
        self.electronic_energies = result.electronic_energies
        self.nuclear_repulsion_energy = result.nuclear_repulsion_energy
        self.total_energies = result.total_energies

    # def digonalize():
