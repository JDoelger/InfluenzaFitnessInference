# Overview

"Inferring the intrinsic mutational fitness landscape of influenza-like evolving antigens from temporally ordered sequence data".
Study by Julia Doelger, Mehran Kardar, and Arup K. Chakraborty

A pre-print describing this study is available at ...

## Organization

This repository is organized into the following subdirectories:

- [./code/notebooks/fitnessinference/](./code/notebooks/fitnessinference/) contains python code for stochastic simulations of immune-driven pathogen sequences and for analysis and fitness inference based on such artificially created data sets
    - the file simulations.py contains code to run stochastic simulations of a population of antigenic sequences evolving under immune pressure and in a given intrinsic fitness landscape
    - the file [run_multiple_simus.py](./code/notebooks/fitnessinference/run_multiple_simus.py) contains code to run several parallel repetitions of a simulation batche with different rng initializations on a slurm-managed cluster
    - the file [analysis.py](./code/notebooks/fitnessinference/analysis.py) contains code to analyze each simulated data set and to save the analyzed data and figures (Figs. 2-8 in manuscript) in the respective result files
    - the file [run_multiple_anas.py](./code/notebooks/fitnessinference/analysis.py) contains code to analyze several simulation batches parallely on a slurm-managed cluster (in particular for creating analysis data for each simulation, which can take some time, plotting afterwards is fast)
    - the file [HA_analysis.py](./code/notebooks/fitnessinference) contains code to postprocess and plot the HA sequence data from the fasta file in the same folder. The sequences were retrieved from IRD (fludb.org) as all full protein sequences for HA(H3N2) between 1968-2020. The strain succession plots for HA (Fig. 1 in the manuscript) is saved under [./figures/](./figures/)

- [./results/simulations/](./results/simulations/) contains the simulation results, where the directories are named after the date of simulation and the simulation parameter, which was varied in the respective batch of simulations
