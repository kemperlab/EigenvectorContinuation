Folders used:
=================
1. calibration -> This is the folder where IBM calbration files downloaded on the day the experiment are saved.

2. circuits -> This is where circuits are saved. EVC folder is for getting the matrix component experiments of EC method. LCU folder is for the experiments measuring energies using LCU method. These circuits are generated uing the qsearch compiler after passing the desired unitaries. then they are traspiled using qiskit. The ones before transpiling are saved with the name 'notrans' ans the transpiled ones which are run are inside the folder trans.

3. log -> This is the folder where log of some of the experiments and tests are saved incase if we want to refer them in future.

4. matrix_data -> This is the folder where the matrices (Hamiltonian pieces and overlap matrices) are written. This is after combining each of the Pauli matrix components measurements.

5. paper_figures -> This is where figure for the paper and the associated data for plotting is saved. schematic plots folder is where the machine layouts, circuit diagrams and other external figures are saved.

6. plots -> This is where plots during coding and experimenting are saved.

7. qsearch_dir -> This is where files made which while running the qsearch (search compiler) is saved. These files are used by the search compiler to make the circuit.

8. results -> This is the folder where the results of the quantum machine runs are saved. These are used later to recreate the matrix components and the plots

9. summary -> This is the folder where some computed results like eigen values are saved after the machine run. This is also like the log folder to see if these values were needed incase.
 
10. XYIntraExtraFig -> This is the folder that contains the data and the notebook that makes the Interpolation Extrapolation Figure for the 2 site XY model.

Jupyter notebooks used:
=======================

1. writing_data_for_paper_plot.ipynb -> This is notebook that writes data for the paper plot. After experimenting scripts in other notebooks, they are being saved this notebook so that most of the paper plots data writing are in one notebook.

2. plotsforpaper.ipynb -> This is the notebook that uses values from the written data and plots figures for the paper.

3. 5site_experiments.ipynb -> This is the notebook that currently writes data and plots figures for the 5 and 8 site basis completeness plots for the paper.

4. threebasisexperimentsinQC.ipynb -> This is the notebook where three basis experiments are run for the qasm simulator for 2 site XY model. The errors are not well understood so they are not used at the moment.

5. test_fidelity.ipynb -> This is the notebook where some basic fidelity tests are run 2 site XY model

python scripts used:
=======================

1. main.py -> This is the main function. This was used to run all of the quantum machine experiments.

2. functions_in_main.py -> This is most of the experiments run using main collected to a separate file to avoid clutttering.

3. continuers.py -> This is where the code for EC is in.

4. hamiltonian.py -> This is where the code defining different spin Hamiltonians are in.

5. h2molecule.py -> This is where H2 molecule code that uses qiskit_nature is in.

6. qsearch_bundle_circuit.py -> This is where the code that creates EC circuits that measures matrix compoments in quantum machine using qsearch/search compiler is in.

7. qsearch_LCU.py -> This is where the code that creates circuits for LCU energy measurement using qsearch/search compiler is in.

8. quantum_circuit_mimic.py -> This is where the mimic version of quantum circuits are run which basically uses unitaries rather than gates. 

others:
===============
1. matplotlibrc -> This has the matplotlib settings for the paper plots. 


