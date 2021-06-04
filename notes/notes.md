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

# log 31 March 2021 (simulation coding)
- for my sdelf-defined package fitnessinference, in which I added an __init__.py file I can for now only import it from the same directory, where the package folder is located

# log April 6 (simulation coding)
- I am able to use unit tests for my different functions by typing 'python -m pytest fitnessinference' in anaconda prompt when in same directory, where package fitnessinference is located
- I can test individual function with 'python -m pytest fitnessinference/tests/test_simulation.py::test_fitness_int' (in this example test function fitness_int)
- finished basic set up of simulation and ran minimal test

# log April 7 (simulation and inference)
- finished simulation set up and started first trial simulation,
in which I set parameter values as standard from last year
and vary population size
- storing of results in pypet file didn't work yet,
so far I am working with the stored intermediate results

# log April 9 (inference)
- when using each observed sequence at each time step as independent sample,
the inference gives poorer results (correlations) than when using each observed unique sequence (strain) at each time as a sample (as before)
- when weighting the feature matrix of each strain with its observation count, the inference also gives poorer results (correlations) than when using the previous approach without weighting
- therefore I have more confidence in the inference approach that I used so far

# log April 12 (analysis pipeline)
- I am still not very comfortable with pypet
- But made progress with analysis pipeline

# log April 13 (analysis pipeline)
- worked on reproducible simu and analysis pipeline
	- for each simu and each analysis created info_file with settings and description
	- moved away from pypet data, decided to only used it for parameter sweep for now
- need to still complete implementing the simu and analysis pipeline neatly

# log April 14 (manuscript outline and example figures)
- thought of types of figures for manuscript, filled some example figures
- wrote some rough comparison with MPL method in discussion of manuscript

# log April 15 (manuscript outline and meeting w Arup and Mehran)
- produced manuscript figure with plots of intrinsic, host, total fitness distributions for one example analysis
- discussed manuscript outline and figures with A & M

# log April 16 (manuscript plan and simulations)
- implemented simulation which runs parameter sweep
- submitted simulation (on my pc) for varying sequence length
- so far I use no multiprocessing, so simus are run one after the other
- logging output is automatically also saved in log directory under date and time folder names which is located in the current working directory, from where I run the simulation

# log April 22 (manuscript outline and figures)
- produced all figures that I have thought of so far for manuscript 
(example plots for one simulated data set, parameter explorations)
- wrote down some things to discuss with A and M in next meeting

# log April 27 (manuscript outline)
- Arup and Mehran are fine with figures that I have so far, want to wrap up study
- added inference performance and error measure definitions on rocketbook page

# log May 3 (manuscript)
- read manuscript draft and make small edits
- sent draft with intro, model, inference, discussion sections to A and M

# log May 4 (coding and meet Mehran)
- worked on tidying up code putting blocks of code into functions
- met Mehran shortly, where he gave me comments on the manuscript

# log May 6 (coding and edit ms results sections)
- ran simulation of Mt Fuji model with varying h
- tidied up plotting of results from single analysis

# log May 7 (simu analysis and references for manuscript)
- ran analysis on Mt Fuji model simulations
- finding references for traveling-wave influenza-like evolution modeling

# log May 10 (Mt Fuji discussion)
- meeting with Mehran discussing about Mt fuji simulation results

# log May 11 (Mt Fuji stringency analysis)
- calculate and plot number of mutations in selected seqences in each year: number of mutations increases with time but less than linearly (far from exponentially)
	- since the number of strains with k mutations increases with k (up to L/2) as L choose k, and if many fitness-equivalent strains are created parallelly then the immunity-dependent fitness cost for each individual strain accumulates slower with increasing number of fitness-equivalent sequences, i.e., it accumulates slower with increasing k

# log May 18 (code for nice figures)
- I wrote functions for each plot of the manuscript, so that I also have better overview and easily change layout later

# log May 19 (Supplementary)
- I wrote some supplementary with a short description of the simulations and reaults for the simple Mount Fuji model

# log May 20 (Supplementary)
- making plots of fitness distributions for all simulated seasons (instead of only inferred ones) for Mount Fuji model
- write supplementary description of simulation and fuji model results

# log May 24 (Supplementary)
- found that I am calculating the std for inferred parameters wrongly:
	- for the parameters which have no information from the observed data, the std should be sqrt(1/lambda) instead of lambda
	- I am correcting that in analysis.py (reran single simu plots) 
- also I found that I replaced the max and min of h and J not with the max and min of the long (L=105) sequence but with wrong values
	- Ideally I should run the simus again with corrected values or with even better (or no) scaling of fitness coefficient distribution between different sequence lengths

# log May 25 (check out engaging cluster again for repeat simus)
- connect to cluster node "eofe5.mit.edu" or eofe7 via putty as described in folder EngagingCluster under README, set up file transfer via securecrt as used before

# log June 2 (rerun simulations for manuscript)
- rerun simulations for manuscript

# log June 3 (rerun analyses for manuscript)
- rerun analysis producing analyzed data

# log June 4 (make manuscript plots and run new simus)
- make manuscript plots based on new simus and analyses (sampling size plot looks less convincing)
- run new simulations with same settings but different random seed for simulation

# To Do :

[ ] write abstract

[ ] run correct simulations with no replacement of min/max of h,J (ideally several runs w. different rng initialization for each parameter combo)

[ ] run correct analysis and plots with correct calc of std (single-simu plots done)

[ ] think about and write down (analytical) justification for selection stringency condition (with use of Mt Fuji model)

[ ] write down (and discuss with A and M) derivation of F_host functional form -> I don't understand why [Luksza and Laessig 2014] (eq. 11-14 in their methods) don't discuss validity of their expression 1-sum compared to more correct mean-field expression exp(-sum) (see also [Yan et al. 2019, eq. 3-5] and [Gog and Grenfell 2002, ])

[ ] tidy up code for figures -> write functions

[ ] write up full details of simulation, inference and analysis in Methods or SI

[ ] do replicate simulations (with different RNG seeds) to get robust inference performance (see Fig. 2 in Barton paper who did 100 replicate simus and calculated mean AUC)

[ ] think about/ explore appropriate indicator from sequence data (without additional info from simu) that correlates with selection stringency/inference performance, e.g. something about strain succession (avg./max. lifetime of strains, log(x/x'),...)
(might have to do few extra simus varying the selection regime like decreasing sigmah 1 to 0 in few steps, if sigmah=0 can I use MPL to infer exp(F)? only if F<<1 -> exp(F) approx 1+F)

[ ] submit simulation to cluster (lymphocyte or engaging), before try out multiprocessing in pypet to make use of parallel computing

[ ] submit simulation to sweep sigma_h=[0, 0.01, 0.1]

[ ] submit simu to sweep fitness coefficients, 3 diff. h (most del., some neutral, some beneficial), 3 diff. J (few del.,few ben., most neutral), vary ben. J, ben. h, e.g. start with two simus with larger/smaller ben. J, two simus w larger/smaller ben. h

[ ] make sure that I get some inference result for each analysis,
even where data are lacking, e.g. by small regularization coeffs for each param (corresponds to some wide gaussian prior)

[ ] find out, why the strain labels in succession plots seem to increase linearly in simulation and exponentially in data

[ ] implement reproducible simulation and analysis pipeline, using pypet mainly (only) for parameter sweep

[x] check sampling size analysis (Arup's comment on Fig. 6). How much does number of seasons matter compared to B?

[x] look up good coding practices for simulations (filenames, log files, retrieving results from simu with specific parameters etc.) -> use pypet

[x] test pypet with cellular automata simulation [Meyer & Obermayer 2016, Frontiers in Neuroinformatics]

[x] write basic code for simulation

[x] test simulation and analysis on small example (on my pc)

[x] write basic code for simulation postprocessing and analysis

[x] add weights x(Si, t) into inference (features and response) and think more about correct linear regression (normalization etc.) 
- book by Hastie et al. 2009, p. 45 and p. 64p. 
- tried different inference but previous methods without strain weighting gives best results

[x] plan for figures/sections for paper to better decide, how I do different analyses and what is missing

[x] write discussion in manuscript about challenges for adaptation to real flu data, indications for stringent selection regime, comparison with Barton inference method 

[x] slightly modify discussion of MPL emphasizing disussed important points

[x] remove mean(std) from fitness dist plot

[x] submit simulation to sweep sequence length L=[5, 10, 20, 30, 50, 100]

[x] vary sampling size and inf_end for inference param exploration (sensitivity analysis)

[x] make manuscript figures that A, M and I agreed on (update manuscript)

[x] write complete first manuscript draft and send to A and M

[x] run simulations for Mt Fuji model with equal h at all sites and J=0, for various h -> simu_name=2021May06_temp

[x] rewrite manuscript results sections based on M's suggestions

[x] do inference and fitness distribution analysis on Mt Fuji simulations

[x] plot number of accumulated mutations as function of time for fuji simulations










