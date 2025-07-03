# Connectome-based encoding of subjective drug responses to acute oral methamphetamine 

## Contents

- [Description](#Description)
- [Reference](#Reference)
- [Code](#Code)
- [Files](#Files)
- [Atlas](#atlas)
- [Contact](#Contact)
------------------------------------------------------------------------

## Description

The repository contain files and MATLAB code to run connectome-based predictive modelling (CPM) and different network lesioning analyses. These can be used to create brain-behavior models. 

------------------------------------------------------------------------

## Reference

Please cite:

- **Rodriguez-Santos et al., 2025**: Connectome-based encoding of subjective drug responses to acute oral methamphetamine

------------------------------------------------------------------------

## Code

Analyses were conducted using MATLAB version R2021a.

1. CPM_KfoldCV.mat - Runs k-fold CPM.
2. CPM_KfoldCV_Permutation.mat - Generates null distribution for CPM.
3. Leave_One_In_Lesioning.mat - Performs leave-one-in network lesioning.
4. Network_Pair_Lesioning.mat - Peforms network pair lesioning. 

------------------------------------------------------------------------

## Files

1. network_definition.txt - Contains node numbers and network affiliations for Shen 268 atlas.
2. Positive_Network_Weight_Mask.txt - Weight mask of positive network edges for replication.
3. Negative_Network_Weight_Mask.txt - Weight mask of negative network edges for replication.

------------------------------------------------------------------------

## Atlas

------------------------------------------------------------------------

## Contact

For any questions, contact Lester Rodriguez Santos (first author) at lester.rodriguez@yale.edu
