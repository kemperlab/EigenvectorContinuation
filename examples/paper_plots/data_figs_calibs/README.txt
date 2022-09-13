Figure 3: Continuing Eigen Spectrum for 2 site XY model. 
============================================================

1)  Fig3_ContinuingEigenspectrum_EC_Energy&Bz.dat stores the eigen energies of ground and first excited state respectively for each subfigure: 
column 1-2 corresponds to energies for subfigure (a) Nsite=2, Bx=0.1, Bztraining = [0.1, 0.9] ; 
column 3-4 corresponds to energies for subfigure (b)Nsite=2, Bx0.1, Bztraining = [0.1, 1.6] ; 
column 5-6 corresponds to energies for subfigure (c) Nsite=2, Bx=0.0, Bztraining = [0.9] (where ground and first excited were the two training points); 
column 7-8 corresponds to subfigure (d) Nsite=2, Bx0.0, Bztraining = [0.1, 1.6];
column 9 has the  Bz values for the x axis in all subfigures.

2)  Fig3_ContinuingEigenspectrum_Exact_Energy&Bz.txt has exact eigen energies and the corresponding Bz values for both Bx values in increasing order of energy. 
Column 1-4 has exact energy for Bx=0.1, column 5-8 has exact energy for Bx=0.0, column 9 has the Bz values

3) Fig3_ContinuingEigenspectrum_TrainingPt.dat has the data for the training points for all subfigures. Column 1 has the the Bz value and column 2 has the corresponding energy for the subfigure (a) ; column 3 has the the Bz value and column 4 has the corresponding energy for the subfigure (b) ; column 5 has the the Bz value and column 6 has the corresponding energy for the subfigure (c) ; column 7 has the the Bz value and column 8 has the corresponding energy for the subfigure (d).  



Figure 4: Basis set completeness plot for 5 site XY model. 
============================================================

4)  Fig4_BasisComp_Exact_energyN=5_npoints=100.dat has the data of eigenenergies vs Bz from exact diagonalization for 5 site system. 1st column has Bz values; 2nd to 1+2^5 th columns has the eigen energies.

5)  Fig4_BasisComp_ed_energyN=5[0.5, 1.3]lentarget=10.dat has the data of eigenenergies vs Bz using the EC method for 5 site system with training points Bz=0.5 and 1.3. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

6)  Fig4_BasisComp_ed_energyN=5[0.5, 1.3, 1.8]lentarget=10.dat has the data of eigenenergies vs Bz using the EC method for 5 site system with training points Bz=0.5, 1.3 and 1.8. 1st column has Bz values; 2nd, 3rd and 4th columns has the eigen energies.

7)  Fig4_BasisComp_crossingN=5.dat has Bz values where the ground state cross over occurs for 5 site XY model.

8)  Fig4_BasisComp_trainingN=5[0.5, 1.3].dat has the values for training points for the top panel. 1st column has the Bz values and 2nd cloumn has the corresponding ground state energy.

9)  Fig4_BasisComp_trainingN=5[0.5, 1.3, 1.8].dat has the values for training points for the bottom panel. 1st column has the Bz values and 2nd column has the corresponding ground state energy.

 
Figure 5: Basis set completeness plot for 8 site XY model. 
============================================================

10)  Fig5_BasisComp_Exact_energyN=8_npoints=100.dat has the data of eigenenergies vs Bz from exact diagonalization for 8 site system. 1st column has Bz values; 2nd to 1+2^8 th columns has the eigen energies.

11)  Fig5_BasisComp_ed_energyN=8[0.2, 0.5, 1.3]lentarget=10.dat has the data of eigenenergies vs Bz using the EC method for 8 site system with training points Bz=0.2, 0.5, 1.3. 1st column has Bz values; 2nd to 4th columns has the eigen energies.

12)  Fig5_BasisComp_ed_energyN=8[0.2, 0.5, 1.3, 1.7]lentarget=10.dat has the data of eigenenergies vs Bz using the EC method for 8 site system with training points Bz=0.2, 0.5, 1.3, 1.7. 1st column has Bz values; 2nd to 4th columns has the eigen energies.

13)  Fig5_BasisComp_ed_energyN=8[0.2, 0.5, 1.3, 1.7, 1.9]lentarget=10.dat has the data of eigenenergies vs Bz using the EC method for 8 site system with training points Bz=0.2, 0.5, 1.3, 1.7, 1.9. 1st column has Bz values; 2nd to 4th columns has the eigen energies.

14)  Fig5_BasisComp_crossingN=8.dat has Bz values where the ground state cross over occurs for 8 site XY model.

15)  Fig5_BasisComp_trainingN=8[0.2, 0.5, 1.3].dat has the values for training points for the top panel. 1st column has the Bz values and 2nd cloumn has the corresponding ground state energy.

16)  Fig5_BasisComp_trainingN=8[0.2, 0.5, 1.3,1.7].dat has the values for training points for the middle panel. 1st column has the Bz values and 2nd cloumn has the corresponding ground state energy.

17)  Fig5_BasisComp_trainingN=8[0.2, 0.5, 1.3,1.7,1.9].dat has the values for training points for the bottom panel. 1st column has the Bz values and 2nd cloumn has the corresponding ground state energy.

Figure 6: Eigenvector Continuation plot for 2 site XY model.
==============================================================

18)  Fig6_Exact_BzvsE_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1.dat has the data of eigenenergies vs Bz from exact diagonalization for 2 site system. 1st column has Bz values; 2nd to 1+2^2 th columns has the eigen energies.

19)  Fig6_ed_BzvsE_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1_Bzlist_training=[0.1, 1.3]moretargets=True.dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.3 for the top panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

20)  Fig6_ibmqsim_BzvsEm_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1_Bzlist_training=[0.1, 1.3]moretargets=True.dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.3 obtained using qasm simulator for the top panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

21)  Fig6_bogota_BzvsEm_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1_Bzlist_training=[0.1, 1.3]moretargets=True.dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.3 obtained using ibmq_bogota backend for the top panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

22)  Fig6_manila_BzvsEm_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1_Bzlist_training=[0.1, 1.3]moretargets=True.dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.3 obtained using ibmq_manila backend for the top panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

23)  Fig6_training[0.1, 1.3]BzvsE_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1.dat has the values for training points for the top panel. 1st column has the Bz values and 2nd column has the corresponding ground state energy.

24)  Fig6_ed_BzvsE_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1_Bzlist_training=[0.1, 1.9]moretargets=True.dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.9 for the bottom panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

25)  Fig6_ibmqsim_BzvsEm_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1_Bzlist_training=[0.1, 1.9]moretargets=True.dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.9 obtained using qasm simulator for the bottom panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

26)  Fig6_manila_BzvsEm_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1_Bzlist_training=[0.1, 1.9]moretargets=True.dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.3 obtained using ibmq_manila backend for the bottom panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

27)  Fig6_montreal_BzvsEm_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1_Bzlist_training=[0.1, 1.9]moretargets=True.dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.3 obtained using ibmq_manila backend for the bottom panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

28)  Fig6_training[0.1, 1.9]BzvsE_EVCN_2siteXY_J=-1_Bzmax=2_Bzmin=0.0Bx=0.1.dat has the values for training points for the bottom panel. 1st column has the Bz values and 2nd column has the corresponding ground state energy.

Figure 7: Linear Combination of Unitaries plot for 2 site XY model.
=====================================================================

29)  Exact_BzvsE_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1.dat has the data of eigenenergies vs Bz from exact diagonalization for 2 site system. 1st column has Bz values; 2nd to 1+2^2 th columns has the eigen energies.

30)  ed_BzvsE_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1_Bzlist_training=[0.1, 1.3].dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.3 for the top panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

31)  ibmqsim_BzvsEm_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1_Bzlist_training=[0.1, 1.3].dat has the data of eigenenergies vs Bz using the LCU method for 2 site system with training points Bz=0.1, 1.3 obtained using qasm simulator for the top panel. 1st column has Bz values; 2nd column has the measured energy.

32)  bogota_BzvsEm_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1_Bzlist_training=[0.1, 1.3].dat has the data of eigenenergies vs Bz using the LCU method for 2 site system with training points Bz=0.1, 1.3 obtained using ibmq_bogota backend for the top panel. 1st column has Bz values; 2nd column has the measured energy.

33)  montreal_BzvsEm_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1_Bzlist_training=[0.1, 1.3].dat has the data of eigenenergies vs Bz using the LCU method for 2 site system with training points Bz=0.1, 1.3 obtained using the ibmq_montreal for the top panel. 1st column has Bz values; 2nd column has the measured energy.

34)  training[0.1, 1.3]BzvsE_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1.dat has the values for training points for the bottom panel. 1st column has the Bz values and 2nd column has the corresponding ground state energy.

35)  ed_BzvsE_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1_Bzlist_training=[0.1, 1.9].dat has the data of eigenenergies vs Bz using the EC method for 2 site system with training points Bz=0.1, 1.9 for the bottom panel. 1st column has Bz values; 2nd and 3rd columns has the eigen energies.

36)  ibmqsim_BzvsEm_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1_Bzlist_training=[0.1, 1.9].dat has the data of eigenenergies vs Bz using the LCU method for 2 site system with training points Bz=0.1, 1.3 obtained using qasm simulator for the top panel. 1st column has Bz values; 2nd column has the measured energy.

37)  montreal_BzvsEm_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1_Bzlist_training=[0.1, 1.9].dat has the data of eigenenergies vs Bz using the LCU method for 2 site system with training points Bz=0.1, 1.3 obtained using the ibmq_montreal backend for the top panel. 1st column has Bz values; 2nd column has the measured energy.

38)  manila_BzvsEm_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1_Bzlist_training=[0.1, 1.9].dat has the data of eigenenergies vs Bz using the LCU method for 2 site system with training points Bz=0.1, 1.9 obtained using the ibmq_manila backend for the top panel. 1st column has Bz values; 2nd column has the measured energy.

39)  training[0.1, 1.9]BzvsE_LCUN_2siteXY_J=-1_Bzmax=2_Bzmin=0Bx=0.1.dat has the values for training points for the bottom panel. 1st column has the Bz values and 2nd column has the corresponding ground state energy.

Figure 8: Eigenvector Continuation in XXZ model.
==================================================
40)  Fig8_XXZ_Exact_energy[0.2, 1.2]_npoints=100.dat has the data of eigenenergies vs Jz from exact diagonalization for the four site XXZ system. 1st column has Jz values; 2nd to 1+2^4 th columns has the eigen energies.

41)  Fig8_XXZ_ed_energy=[0.2, 1.2]lentarget=30.dat has the data of eigenenergies vs Jz using the EC method for the four site XXZ system with training points Jz = 0.2, 1.2. 1st column has Jz values; 2nd and 3rd columns has the eigen energies.

42)  Fig8_XXZ_training[0.2, 1.2].dat has the values for training points. 1st column has the Jz values and 2nd column has the corresponding ground state energy.

Figure 9: Hydrogen molecule.
==============================
43)  Fig9_hydrogen_Exact_distvstotE_dist0.2to4.0_npoints=200.dat has the data of total energies (electronic energies + nuclear repulsion energy) vs interatomic distance (R) from exact diagonalization for the two qubit hydrogen problem. 1st column has R values; 2nd to 1+2^2 th columns has the eigen energies.

44)  Fig9_hydrogen_ed_distvstotE_disttraining=[0.4, 1.7]lentarget=200.dat has the data of eigenenergies vs R using the EC method for the the two qubit hydrogen problem with training points R = 0.4,1.7 for the top panel. 1st column has R values; 2nd and 3rd columns has the total energies.

45)  Fig9_hydrogen_training[0.4, 1.7].dat has the values for training points for the top panel. 1st column has the R values and 2nd column has the corresponding ground state total energy.

46)  Fig9_hydrogen_ed_distvstotE_disttraining=[1.6, 2.0]lentarget=200.dat has the data of eigenenergies vs R using the EC method for the the two qubit hydrogen problem with training points R = 1.6,2.0 for the topbottom panel. 1st column has R values; 2nd and 3rd columns has the total energies.

47)  Fig9_hydrogen_training[1.6, 2.0].da thas the values for training points for the bottom panel. 1st column has the R values and 2nd column has the corresponding ground state total energy.

IBMQ_backend calibration data.
===============================

48)  ibmq_bogota_calibrations_Feb25.csv has the calibration data for ibmq_bogota backend on February 25th 2022.
49)  ibmq_manila_calibrations_Feb25.csv has the calibration data for ibmq_manila backend on February 25th 2022.
50)  ibmq_montreal_calibrationsFeb26.csv has the calibration data for ibmq_montreal backend on February 26th 2022.
51)  ibmq_montreal_calibrations_Feb25.csv has the calibration data for ibmq_montreal backend on February 25th 2022.
52)  ibmq_montreal_calibrations_march28.csv has the calibration data for ibmq_montreal backend on March 28th 2022.









