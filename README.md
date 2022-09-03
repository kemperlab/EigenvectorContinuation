# EigenvectorContinuation

This code implements the eigenvector continuation (EC) process found in the paper linked below. The program contains a class `EigenvectorContinuer` that runs eignevector continuation for a user-implemented vector class of type `HilbertSpaceAbstract`, along with a few example implementations.


[TODO] [insert: paper title and authors, arxiv link]

*Abstract:* [TODO] [insert final abstract used in paper]

Authors:

- Jack H. Howard
- Akhil Francis
- Alexander F. Kemper

## Contents

#### EigenvectorContinuer class:
Instances of this class run EC for a given hilbert space and set of target points. An instance can:
- calculate the overlap matrix for the current system
- calculate the subspace hamiltonian for the current system
- refresh the overlap matrix property to correspond to the current hilbert space's training points property
- refresh the subspace hamiltonian to match the current target point property
- solve the generalized eigenvalue problem for all the properties of the current instance

#### HilbertSpaceAbstract (HSA) interface/abstract class:
Each implementation (subclass) of this class must include methods that perform the following:
- calculate basis vectors using the space's properties
- define and calculate the behavior of an inner product
- define and calulate the behavior of an expectation value
- calculate the overlap matrix according to the space's properties
- calculate the subspace hamiltonian according to the space's properties and a given full-size hamiltonian
- select which vectors are relevant in performing EC (ground state vs. a selected excited state)
Every implementation must also include an inner class, `HamiltonInitializer`, that constructs a hamiltonian for the system given a target point.

#### NumPyVectorSpace concrete/subclass of HSA:
Has all the functionality required for an HSA, implemented using the NumPy library and using NumPy matrices for representations of all matrices and vectors. Also contains:
- HamiltonianInitializer inner class:
    - provides needed functions for creating hamiltonians using NumPy arrays

#### UnitarySpace concrete/subclass of HSA:
Has all the functionality required for an HSA, implemented using the NumPy library and using NumPy matrices as unitary matrices for representations of all matrices and vectors. Also contains:
- HamiltonianInitializer inner class:
    - provides needed functions for creating hamiltonians using NumPy arrays
Instances of this class also can:
- calculate a unitary for a given vector
- calculate a set of unitaries for the instance's basis vectors

#### methods.py module:
Includes useful methods such as: 
- plot_xxz_spectrum: produces a plot for a given EigenvectorContinuer, a min B_z value, and max B_z value

More details can be found in the **Doucumentation** section below

#### ParamSet tuple:
Convenient tuple to use for calculations throughout the program


## Documentation
Find detailed documentation of modules [here](https://github.com/kemperlab/EigenvectorContinuation/tree/main/docs/_build/html/index.html). 
// [TODO] verify this


## General Notes
[backgound info on math @Akhil-Francis]

## Installation
Ensure Python 3.8.0 or higher is installed using
```
    python --version
```

Clone the git repository:
```
git clone https://github.com/kemperlab/EigenvectorContinuation
```

Change directories into the newly created repository
```
cd EigenvectorContinuation
```

Run the EigenvectorContinuation setup script
```
python3 setup.py install
```

## Usage
This program is intended to be used on its own or as a supplement to existing programs. This means that one can:
    A. run EC straight out of the box using the included implementations of HSA (NumPyVectorSpace and UnitarySpace), and 
    B. implement and use a custom implementation of HSA, adhering to the requirements outlined in the HSA interface itself. As long as the implementation is valid, an EigenvectorContinuer will take it as a parameter, and any EC calculations should run seamlessly.

In order to use:
1. Ensure that installation and setup (above) has completed successfully
2. Import relevant modules from this repository and/or from custom implementations into the desired python module/script

*Examples of how to import and use the modules properly are found [in the examples directory](https://github.com/kemperlab/EigenvectorContinuation/tree/main/examples).*
