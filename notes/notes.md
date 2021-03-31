# log 25 March 2021 (simulation of flu evolution, summary of method, coding practices)

## simulation of artificial data for flu antigenic evolution (inputs and outputs, best coding practice)

### input parameters:
- N_pop: population size
- N_site: sequence length
- N_nonepi: number of sites that are not accessible by immune response (default: 0)
- N_state: number of states per site (default: 2)
- mu: mutation probability per site per time step (default: $10^{-4}$)
- sigmah: coefficient or immune-dependent fitness component (default: 1)
- D0: cross-immunity coefficient (default: 5)
- h_model: intrinsic fitness coefficients for single mutations (numpy array with dimensions (N_sites, N_state-1))
- J_model : intrinsic fitness coefficients for double mutations (numpy array with dimensions (num_pairs,N_state-1,N_state-1))
- seed_simu: seed for random number generator for reproducible simulation (for example use date 210325)
- N_simu : number of time steps to simulate (default: 200)

### simulation output:
- running and final pickled .data file (one file with intermediate output that is updated/newly created every or every few time steps + one file with final output):
	- strain_yearly: list [[list of unique sequences (strains)] for each time step] with strains from most to least prevalent at each time
	- strain_frequency_yearly:[[list of frequencies of strains] for each time step] in same order as in strain_yearly  
	- tree_yearly: list [[[parent strain index, child strain index] for each ancestrial pair] for each time step], each unique parent-child strain combination is added once
	- branch_weight_yearly: [[list of counts for each occurrence of a specific parent-child combi] for each time step], the index of the branch weight is the same as the index of the [parent,child] in tree_yearly for the same time step
	- input parameter values as dictionary
- .txt file for running output (log):
	- give some initial description of the simulation at start of file
	- print computation time for different parts of operation in each time step (is there a better way to check computation times, maybe use unit tests?)
	- print summaries of outputs such as total number of strains etc. to easily follow progress

### coding practice (see Jake Vanderplas youtube series for reproducible data analysis in jupyter):
- define simulation as function with each of the input parameters as input, setting default values for most of the input params in function definition
so that I don't have to fill each parameter every time
- define various modular functions for various calculations in the simulation
- save small functions in separate .py file(s)
- for each function include comprehensive documentation with function description, parameters (description + type/dimension), return values (description + type/dimension)
- for each function write a test function for unit testing, for example using pytest like J Vanderplas
- regularly test updated functions with pytest and update test functions if new functionalities are added or errors are identified and fixed (make sure that same error does not happen twice and if it does, let it be detected by test function)
- regularly update git and github repository by using git add/commit/push
- keep file overview: 
	- automatic naming of simulation output files: each file with at least current date and simulation file name that it is based on, maybe additional identifiers to find data later?
	- simulation input parameter values saved in .data output file and partially also in .txt log file
	- maybe save different sets of results for several simulations that for example relate to the same type of simulation or parameter sweep in different directories (but how do I do that consistently with the cluster?)
	- keep daily notes about simulations that I am running and about results
	- create one separate notebook for postprocessing and analysis/inference and save postprocessed data in file whose name connects to the simulation output files and to the code file that was used for postprocessing

# log 29 March 2021 (coding/simulation good practices)

- I could base my simulation on a dictionary of strains/sequences that are instances of selfdefined classes, where the dictionary gets updated every time step, losing some members and adding others
	- but I have to read more and think more, how much I would have to change in my current simu and if this makes the simulation better/more efficient
	- for now I will keep more closely to my previous simulations

##  data and file management for simulation results
- python package h5py to create and manage HDF5 file structures 
- python package pypet (uses h5py and sumatra (electronic lab notebook)) for parameter exploration and comprehensible simulation data storage in hdf5 format [Meyer & Obermayer 2016, Frontiers in Neuroinformatics]

# log 30 March 2021 (pypet test and simulation set up)
- test simulations for pypet (cellular automata etc.) are saved in new miscellaneous_code_experiments directory
- used pypet user manual https://pypet.readthedocs.io/en/latest/manual/manual_toc.html to find test code that I could try out
- downloaded HDFView to inspect HDF5 files (desktop shortcut for .bat batch file, since .exe file doesn't work correctly)
- didn't find so far how to store and retrieve intermediate results of simulation
	- but I could maybe add the intermediate results as additional results in the hdf5 file
	- or I could write out intermediate results (into the log file or as separate files like plots etc.) separately from the hdf5 output, which I then can easily delete or overwrite afterwards








# To Do (March 29 - April 2):
[x] look up good coding practices for simulations (filenames, log files, retrieving results from simu with specific parameters etc.) -> use pypet

[x] test pypet with cellular automata simulation [Meyer & Obermayer 2016, Frontiers in Neuroinformatics]

[ ] write basic code for simulation

[ ] write basic code for simulation postprocessing and analysis

[ ] test simulation and analysis on small example (on my pc)

[ ] submit simulation to cluster (lymphocyte or engaging)

[ ] think about which parameters to vary for sensitivity analysis

[ ] submit several simulations to do parameter analysis









