from pypet import Environment, Parameter, cartesian_product, progressbar, Parameter
import numpy as np
import csv
import os
import copy
import pickle
import logging
from datetime import date
import time

def add_parameters(traj):
    """ 
    add parameters to the trajectory with descriptions and default values
    
    Parameters:
    
    traj: pypet.trajectory.Trajectory 
            trajectory container, which manages the parameters
    
    Returns:
    
    None
    
    Dependencies:
    
    from pypet import Parameter
    import numpy as np
    """
    traj.par.N_pop = Parameter('N_pop', 10**5, 'population size')
    traj.par.N_site = Parameter('N_site', 20, 'sequence length')
    traj.par.N_state = Parameter('N_state', 2, 'number of states per site')
    traj.par.mu = Parameter('mu', 10**(-4), 'mutation prob. per site per time step')
    traj.par.sigma_h = Parameter('sigma_h', 1, 'host fitness coefficient')
    traj.par.D0 = Parameter('D0', 5, 'cross-immunity distance')
    traj.par.h_0 = Parameter('h_0', -7, 'typical single mutation fitness cost')
    traj.par.J_0 = Parameter('J_0', 0, 'typical mutation coupling coefficient')
    traj.par.hJ_coeffs = Parameter('hJ_coeffs', 'p24',
                                   'fitness coefficients')
    # traj.par.seed = Parameter('seed', 123456, 'RNG seed')
    # randomly choose rng seed and save it as parameter
    seed = np.random.randint(10**6)
    traj.par.seed = Parameter('seed', seed, 'RNG seed')
    traj.par.N_simu = Parameter('N_simu', 200, 'number of time steps to simulate')
    
def fitness_coeff_constant(N_site,N_state,h_0,J_0):
    """
    creating the mutational fitness coefficients for the simulated sequences
    in the case of constant fields and constant couplings
    
    Parameters:
    
    N_site: int
            sequence length
    N_state: int
            number of states per site
    h_0: int or float
            single-mutation fitness cost
    J_0: int or float
            fitness coupling coefficient for double mutations
    
    Returns:
    
    h_list: numpy.ndarray
            mutational fitness change for mutation to each mutated state at each site      
    J_list: numpy.ndarray
            added fitness change due to couplings of 
            two specific mutations to each state at each site
    
    Dependencies:
    
    import numpy as np
    """
    numparam_J=int(N_site*(N_site-1)/2)
    J_list=np.ones((numparam_J, N_state-1, N_state-1))*J_0
    h_list=np.ones((N_site, N_state-1))*h_0
    
    return h_list, J_list

def fitness_coeff_p24(N_site, N_state, filepath='C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/NewApproachFromMarch2021/'
                'InfluenzaFitnessInference/code/notebooks/fitnessinference/p24-B-S0.90-Ising-reduced-out-learn.j', seed=12345, h_min=-9., h_max=-0.5, J_min=-2., J_max=3.):
    """
    creating the mutational fitness coefficients for the simulated sequences in the case
    of fitness coeffs sampled from coefficients inferred for the HIV protein p24
    
    Parameters:
    
    N_site: int
            sequence length (<=105)
    N_state: int
            number of states per site
    filepath (optional): str
            filepath to a .j file that was created by the ACE inference of
            p24 fitness coefficients
    seed (optional): int 
            seed for random sanpling from the given coefficients
    h_min, h_max, J_min, J_max (optional): float
            maximum and minimum mutational fitness coefficients 
            (fixed for various sequence lengths)
            
    Returns:
    
    h_list: numpy.ndarray
            mutational fitness change for mutation to each mutated state at each site
    J_list: numpy.ndarray
            added fitness change due to couplings of 
            a mutation to a specific state at any site i 
            with a mutation to a specific state at any other site j
            
    Dependencies:
    
    import os
    import numpy as np
    import csv
    """
    filepath = os.path.normpath(filepath)
    if not os.path.exists(filepath):
        filepath = os.path.join(os.getcwd(), 'code', 'notebooks', 'fitnessinference', 'p24-B-S0.90-Ising-reduced-out-learn.j')

    # np.random.seed(seed)
    # get coefficients from file
    with open(filepath) as f:
        reader = csv.reader(f, delimiter = '\t')
        param_list = list(reader)
    # calculate sequence length from the coeff data
    seq_length = int((np.sqrt(1 + 8 * len(param_list)) - 1) / 2)
    # separate h and J list
    h_list = [[float(param_list[i][j]) for j in range(len(param_list[i]))]
             for i in range(seq_length)]
    J_list = [[float(param_list[i][j]) for j in range(len(param_list[i]))]
              for i in range(seq_length, len(param_list))]
    # calculate matrix from J_list
    k=0
    J_mat=[[[] for j in range(seq_length)] for i in range(seq_length)]
    for i in range(seq_length):
        for j in range(i):
            J_mat[i][j]=J_list[k]
            J_mat[j][i]=J_list[k]
            k+=1
    # reduce J and h lists to sequence of length N_site
    J_list_red = []
    for i in range(N_site):
        for j in range(i):
            J_list_red.append(J_mat[i][j])
    h_list_red = h_list[:N_site]
    # sample from h and J parameters to get coefficient lists for only N_state states at each site
    J_list_final = np.array([np.random.choice(J_list_red[i], size=(N_state-1, N_state-1))
                            for i in range(len(J_list_red))])
    h_list_final = np.array([np.random.choice(h_list_red[i], size=N_state-1)
                            for i in range(len(h_list_red))])

    # # replace max and min of coefficients by specific value, comment out if no modification to sampled coefficients
    # J_list_final = np.where(J_list_final==np.max(J_list_final), J_max, J_list_final)
    # J_list_final = np.where(J_list_final==np.min(J_list_final), J_min, J_list_final)
    # h_list_final = np.where(h_list_final==np.max(h_list_final), h_max, h_list_final)
    # h_list_final = np.where(h_list_final==np.min(h_list_final), h_min, h_list_final)
    
    return h_list_final, J_list_final

def mutate_seqs(seqs, N_state, mu):
    """
    mutate list of sequences according to given mutation probability and number of states,
    
    Parameters:
    
    seqs: numpy.ndarray
            list of sequences
    
    N_state: int
            number of states per site
            
    mu: float
        probability to mutate from the current state to any one of the other states <<1
            
    Returns:
    
    seqs_m: numpy.ndarray
            list of mutated sequences
    
    Dependencies:
    
    import numpy as np
    """
    # first choose randomly how far in the state space each site is shifted
    shift_ind = np.random.choice(N_state, size=seqs.shape, replace=True, p=[1-mu*(N_state-1)]+[mu]*(N_state-1))
    # from this calculate the new state index (which can be negative)
    new_ind = np.array(- N_state + shift_ind + seqs, dtype=int)
    # set the new state
    state_list = np.arange(N_state)
    seqs_m = state_list[new_ind]
    
    return seqs_m

def fitness_int(seq, N_state, h_model, J_model, statevec_list):
    """
    calculate the intrinsic fitness for one sequence
    
    Parameters:
    
    seq: numpy.ndarray
            sequence
            
    N_state: int
            number of states per site
            
    h_model: numpy.ndarray
            mutational fitness change for mutation to each mutated state at each site
            
    J_model: numpy.ndarray
            added fitness change due to couplings of 
            two specific mutations to each state at each site
            
    statevec_list: numpy.ndarray
            list of vectors that represent the states of a sequence site
    
    Returns:
    
    f_int: float
            intrinsic fitness for the sequence
    
    Dependencies:
    
    import numpy as np
    
    """ 
    f_int = 0
    k = 0
    for i in range(len(seq)): # for each state 1
        # state at site i
        s1 = statevec_list[seq[i]]
        # fitness contribution from state at i
        f_int += np.dot(s1, h_model[i])
        for j in range(i): # for each state 2<state 1
            # state at other site j
            s2 = statevec_list[seq[j]]
            # fitness contribution from coupling of state at i with state at j
            f_int += np.matmul(np.matmul(s1, J_model[k]), s2.T)
            k += 1
            
    return f_int

def fitness_int_list(strain_current, N_state, h_model, J_model):
    """
    calculate the intrinsic fitness for each current strain
    
    Parameters:
    
    strain_current: numpy.ndarray
            list of current strains (=unique sequences)
            
    N_state: int
            number of states per site
            
    h_model: numpy.ndarray
            mutational fitness change for mutation to each mutated state at each site
            
    J_model: numpy.ndarray
            added fitness change due to couplings of 
            two specific mutations to each state at each site
    
    Returns:
    
    f_int_list: numpy.ndarray
            intrinsic fitness for each strain
    
    Dependencies:
    
    import numpy as np
    """
    statevec_list=np.array([[int(i==j) for j in range(1,N_state)] 
                        for i in range(N_state)])
    
    f_int_list = np.array([fitness_int(seq, N_state, h_model, J_model, statevec_list) 
                           for seq in strain_current])
        
    return f_int_list

def fitness_host(seq, st_yearly, st_freq_yearly, sigma_h, D0):
    """
    calculate the host population-dependent fitness contribution for one sequence
    at the current time
    
    Parameters:
    
    seq: numpy.ndarray
            sequence
            
    st_yearly: list
            list of strains for each time step up to t-1
            
    st_freq_yearly: list
            list of strain frequencies for each time step up to t-1
            
    sigma_h: float
            coefficient modulating f_host
            
    D0: float
            cross-immunity distance
    
    Returns:
    
    f_host: float
            host-dependent fitness for the sequence at the current time
    
    Dependencies:
    
    import numpy as np
    """
    f_host_noSig = 0 # initialize host fitness without sigma_h factor
    
    for t in range(len(st_yearly)): # iterate through all prev. time steps
        strains = st_yearly[t]
        # create array of same dimension as strain list at t
        seq_arr = np.repeat([seq], len(strains), axis=0)
        # calculate mutational distances between seq_arr and strains
        mut_dist = np.sum(seq_arr!=strains, axis=1)
        f_host_noSig += -np.dot(st_freq_yearly[t], np.exp(-mut_dist/D0))
    
    f_host = sigma_h*f_host_noSig
        
    return f_host

def fitness_host_list(strain_current, st_yearly, st_freq_yearly, sigma_h, D0):
    """
    calculate the host population-dependent fitness contribution for all strains
    at the current time
    
    Parameters:
    
    strain_current: numpy.ndarray
            list of current strains (=unique sequences)
            
    st_yearly: list
            list of strains for each time step up to t-1
            
    st_freq_yearly: list
            list of strain frequencies for each time step up to t-1
            
    sigma_h: float
            coefficient modulating f_host
            
    D0: float
            cross-immunity distance
    
    Returns:
    
    f_host_list: numpy.ndarray
            host-dependent fitness for each strain at the current time
    
    Dependencies:
    
    import numpy as np
    """

    f_host_list = np.array([fitness_host(seq, st_yearly, st_freq_yearly, sigma_h, D0) 
                            for seq in strain_current])
        
    return f_host_list

def flu_antigen_simulation(traj, filepath, varied_simu_params):
    """
    simulate the evolution of flu-like antigenic sequences
    
    Parameters:
    
    traj: pypet.trajectory.Trajectory
            trajectory container, which manages the parameters
            
    filepath: str
            path to folder 
            where results should be stored 
            
    varied_simu_params: list
            list of names of parameters that are varied
            in the parameter sweep
    
    Results:
    
    strain_yearly: list 
            [[list of unique sequences (strains)] 
            for each time step] 
            with strains from most to least prevalent at each time
            
    strain_frequency_yearly: list
            [[list of frequencies of strains] 
            for each time step] 
            in same order as in strain_yearly
            
    pickled .data files with intermediate simulation results (uodated at each simulated time step)
            
    Returns:
    
    run_name: str
            name of file without path or extension
            in which the results of the single run are saved
              
    Dependencies:
    
    other functions in this module
    import numpy as np
    from pypet import Environment, Parameter
    import os
    import pickle
    import copy
    
    """
    # initializations:
    
    # set RNG seed:
    np.random.seed(traj.seed)
    # current sequences, numpy array, initialized with all zeros
    seqs = np.zeros((traj.N_pop, traj.N_site)) 
    # current strains
    strain_current, strain_count_current =\
            np.unique(seqs, return_counts=True, axis=0)
    # strains at each time, list, initialized with initial strain
    strain_yearly = [strain_current]
    # strain frequencies at each time, list, initialized with 1
    strain_frequency_yearly = [strain_count_current/np.sum(strain_count_current)] 
    # set fitness coefficients according to the selected rule
    if traj.hJ_coeffs=='constant':
        h_model, J_model = fitness_coeff_constant(traj.N_site, traj.N_state, traj.h_0, traj.J_0)
    elif traj.hJ_coeffs=='p24':
        h_model, J_model = fitness_coeff_p24(traj.N_site, traj.N_state)

    # filenames for intermediate results:
    name_drop = len('parameters.') # from parameter names length of first part
    params = ''
    # add each of the parameters that are varied in the parameter sweep simulation 
    # with name and value into filename:
    for key, value in traj.f_get_parameters(1).items():
        if key[name_drop:] in varied_simu_params:
            if isinstance(value, int) and (value<100 or key[name_drop:]=='seed'):
                params += key[name_drop:] + '_%.i' % value 
            elif isinstance(value, float) or (isinstance(value, int) and value>=100):
                params += key[name_drop:] + '_%.e' % value
            elif isinstance(value, str):
                params += key[name_drop:] + '_' + value + '_'
            
    filename = os.path.join(filepath, params + '.data')
    
    # simulation of sequence evolution:
    for t in range(traj.N_simu):
        
        # mutate sequences
        
        seqs_m = mutate_seqs(seqs, traj.N_state, traj.mu)
        
        # determine fitnesses
        
        # update strains and strain counts/frequencies
        strain_current, strain_count_current =\
            np.unique(seqs_m, return_counts=True, axis=0) 
        strain_frequency_current = strain_count_current/np.sum(strain_count_current)

        # intrinsic fitness
        f_int_list =\
            fitness_int_list(strain_current, traj.N_state, h_model, J_model)
        # host-dependent fitness
        f_host_list =\
            fitness_host_list(strain_current, strain_yearly, 
                                   strain_frequency_yearly, traj.sigma_h, traj.D0)
        
        # select surviving seqs
        
        # exp(Fint + Fhost) for each strain
        pfit_strain_list = np.exp(f_int_list + f_host_list)
        pfitweighted_list = pfit_strain_list*strain_frequency_current
        # exp(Fi)*xi/(sum(exp(Fj) xj))
        pfitnorm_list = pfitweighted_list/np.sum(pfitweighted_list)
        # randomly select surviving seqs (expressed as strain indices)
        selected_ids = np.random.choice(len(strain_current), size=traj.N_pop, replace=True, 
                                            p=pfitnorm_list) 
        # new list of sequences (from selected strain ids)
        seqs = strain_current[selected_ids]
        
        # update and save data
        
        # update strains and strain frequencies
        strain_current, strain_count_current =\
            np.unique(seqs, return_counts=True, axis=0) 
        strain_frequency_current = strain_count_current/np.sum(strain_count_current)
        # rank strains before saving
        merge_list=list(zip(strain_frequency_current.tolist(), strain_current.tolist()))
        merge_list.sort(reverse=True) # sort coarse strain list according to count
        strain_frequency_current=np.array([x1 for x1,x2 in merge_list])
        strain_current=np.array([x2 for x1,x2 in merge_list])
        # store current strains
        strain_yearly.append(strain_current)
        strain_frequency_yearly.append(strain_frequency_current)
        
        # save intermediate data at time step
        resulting_data = {'strain_yearly': strain_yearly,
                  'strain_frequency_yearly': strain_frequency_yearly}
        with open(filename, 'wb') as filehandle:
            pickle.dump(resulting_data, filehandle)
    
    # add simulation results to the trajectory         
#     traj.f_add_result('test_result', {'list': [[1,2,3,4]]}, comment='test result for testing pypet results')
    # name of saved result file without file extension
    run_name = params
    
    return run_name

def exe_simu(exp_dict, t0):
    """
    run flu evolutionary simulation with varying params given in exp_dict
    """
    # if I want to run all parameter combinations, run cartesian product
    exp_dict = cartesian_product(exp_dict)
    # the entries in the final dictionary need to all have equal lengths
    # to tell the simulation which specific param combos to test
    varied_simu_params = [key for key, val in exp_dict.items()]

    # this file, with which simulation is executed
    simu_code_file = os.path.basename(__file__)

    today = date.today()
    strdate_today = today.strftime("%Y%b%d")
    repository_path = os.path.normpath(
        'C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/NewApproachFromMarch2021/'
        'InfluenzaFitnessInference')
    # for cluster simus use current working directory (and make sure to run python files from the repository directory)
    if not os.path.exists(repository_path):
        repository_path = os.getcwd()

    # determine directories for storage of results

    result_directory = repository_path
    # result folder:
    folder = os.path.join(result_directory, 'results', 'simulations')

    # subfolder to store results
    simu_name1 = strdate_today
    for p in varied_simu_params:
        simu_name1 += "_var" + p

    simu_name = simu_name1
    temp_folder = os.path.join(folder, simu_name)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    simu_num = 1
    while os.path.isdir(temp_folder):
        simu_num += 1
        simu_name = simu_name1 + '_' + str(simu_num)
        temp_folder = os.path.join(folder, simu_name)
    os.makedirs(temp_folder)

    t_makeFile = time.time()
    print('time until file creation: ', t_makeFile-t0, flush=True)

    # filename for final pypet results of the experiment
    simu_file = os.path.join(folder, simu_name + '.hdf5')
    # filepath for logs and storage of intermediate files
    filepath = temp_folder

    # create environment and run the simulation using pypet

    # make use of logging
    logger = logging.getLogger()

    # Create an environment
    env = Environment(trajectory=simu_name,
                      filename=simu_file,
                      overwrite_file=True)

    # Extract the trajectory
    traj = env.traj

    # use the add_parameter function to add the default parameters
    add_parameters(traj)

    simu_param_dict = dict(N_pop=traj.N_pop,
                           N_site=traj.N_site,
                           N_state=traj.N_state,
                           mu=traj.mu,
                           sigma_h=traj.sigma_h,
                           D0=traj.D0,
                           h_0=traj.h_0,
                           J_0=traj.J_0,
                           hJ_coeffs=traj.hJ_coeffs,
                           seed=traj.seed,
                           N_simu=traj.N_simu
                           )

    simu_comment = 'simulation with varying'
    for p in varied_simu_params:
        simu_comment += ' ' + p

    # add the exploration to the trajectory
    traj.f_explore(exp_dict)

    # store simulation info as dictionary

    simu_dict = {}
    # store the various params, which describe the simu, in the dictionary
    simu_dict['simu_name'] = simu_name
    simu_dict['simu_code_file'] = simu_code_file
    simu_dict['simu_comment'] = simu_comment
    simu_dict['varied_simu_params'] = varied_simu_params
    simu_dict['simu_param_dict'] = simu_param_dict
    simu_dict['exp_dict'] = exp_dict

    # save simu info before running simulations
    simu_info_filename = 'simu_info.data'
    simu_info_filepath = os.path.join(temp_folder, simu_info_filename)
    with open(simu_info_filepath, 'wb') as f:
        pickle.dump(simu_dict, f)

    # Run the simulation
    logger.info('Starting Simulation')
    run_names = env.run(flu_antigen_simulation, filepath, varied_simu_params)

    run_names = dict(run_names)
    run_list = [key for key, val in run_names.items()]

    # update simu info with list and names of runs and overwrite prev. simu info file with complete info dict
    simu_dict['run_names'] = run_names
    simu_dict['run_list'] = run_list
    with open(simu_info_filepath, 'wb') as f:
        pickle.dump(simu_dict, f)

def main():
    """
    main function to execute simulation,
    can be executed by running this file simulation.py as a python script,
    """
    t0 = time.time() # starting time of simulation
    # define the parameter exploration for this experiment via exp_dict and run simulation batch
    # exp_dict = {'N_pop': [10, 100, 10**3, 10**4, 10**5, 10**6]}
    exp_dict = {'N_site': [5, 10, 20, 30, 50, 100]}
    # # exp_dict = {'hJ_coeffs': ['constant'], 'h_0': [-15, -10, -7, -5, -1, 0, 1, 5]}
    exe_simu(exp_dict, t0)

# if this file is run from the console, the function main will be executed
if __name__ == '__main__':
    main()
