import numpy as np
import copy
import os 
from pypet import Trajectory
import pickle
import scipy
from fitnessinference import simulation as simu


def load_simu_data(single_simu_filename, simu_name, exp_idx,
                   result_directory='C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/'
                                    'NewApproachFromMarch2021/InfluenzaFitnessInference'):
    """
    load simulation results from a single simulation
    
    Parameters:
    
    single_simu_filename: str
            name including extension .data of the result file for the
            single param combo I want to analyze        
    simu_name: str
            name of the whole pypet based simulation
            usually as date like '2021Apr07'
    exp_idx: int
            run index = index in exp_dict (dictionary)
            for the specific run (parameter combo) that I want to analyze
    result_directory (optional): str
            path to the directory 
            where results are stored 
            (in general: path to the InfluenzaFitnessInference repository)
            
    Returns:
    
    strain_yearly: list
            [[list of unique sequences (strains)] 
            for each time step] 
            with strains from most to least prevalent at each time
    strain_frequency_yearly: list
            [[list of frequencies of strains] 
            for each time step] 
            in same order as in strain_yearly
    traj: Trajectory (pypet.trajectory.Trajectory)
            trajectory with the parameters of this
            single simulation loaded
    
    Dependencies:
    
    import os
    from pypet import Trajectory
    import pickle
    """
    # make sure that file path is in norm path format:
    result_directory = os.path.normpath(result_directory)
    # load parameters from the pypet file
    simu_file = os.path.join(result_directory, 'results', 'simulations', simu_name + '.hdf5')
    # only need trajectory, not environment to look at parameters and results:
    traj = Trajectory(simu_name, add_time=False)
    # load the trajectory from the file with only parameters but not results loaded
    traj.f_load(filename=simu_file, load_parameters=2,
                load_results=2, load_derived_parameters=0)
    # load the parameter values of this specific run
    traj.v_idx = exp_idx

    # load data from the pickled files
    temp_folder = os.path.join(result_directory, 'results', 'simulations', simu_name + '_temp')
    test_filepath = os.path.join(temp_folder, single_simu_filename)

    with open(test_filepath, 'rb') as f:
        resulting_data = pickle.load(f)

    # load information from pypet trajectory or from stored result files:
    strain_yearly = resulting_data['strain_yearly']
    strain_frequency_yearly = resulting_data['strain_frequency_yearly']
                   
    return strain_yearly, strain_frequency_yearly, traj

def sample_seqs(strain_yearly, strain_freq_yearly, seed, B, inf_end):
    """
    create subsamples of the simulated sequences for each time step with given size
    
    Parameters:
    
    strain_yearly: list 
            [[list of unique sequences (strains)] 
            for each time step] 
            with strains from most to least prevalent at each time
            
    strain_frequency_yearly: list
            [[list of frequencies of strains] 
            for each time step] 
            in same order as in strain_yearly
            
    seed: int
            RNG seed for sampling
            
    B: int
            sample size
            
    inf_end: int
            last time step used for inference
    
    Returns:
    
    strain_samp_yearly: list
            subsampling of strain_yearly
            
    strain_samp_count_yearly: list
            strain counts that go with strain_samp_yearly
            
    strain_samp_freq_yearly: list
            strain frequencies that go with strain_samp_yearly
            
    
    Dependencies:
    
    import numpy as np
    """
    np.random.seed(seed)
    strain_samp_yearly = []
    strain_samp_count_yearly = []
    strain_samp_freq_yearly = []
    for t in range(inf_end):
        
        # randomly sample strain indices
        sampled_ids = np.random.choice(len(strain_yearly[t]), size=B, replace=True, p=strain_freq_yearly[t])
        
        # create new strain list with corresponding frequencies
        seqs = strain_yearly[t][sampled_ids]
        strain_samp_current, strain_samp_count_current =\
            np.unique(seqs, return_counts=True, axis=0)
        strain_samp_freq_current = strain_samp_count_current/np.sum(strain_samp_count_current)
        # rank sampled strains
        merge_list = list(zip(strain_samp_count_current.tolist(), strain_samp_freq_current.tolist(), strain_samp_current.tolist()))
        merge_list.sort(reverse=True)
        strain_samp_count_current = np.array([x1 for x1,x2,x3 in merge_list])
        strain_samp_freq_current = np.array([x2 for x1,x2,x3 in merge_list])
        strain_samp_current = np.array([x3 for x1,x2,x3 in merge_list])
        
        # store strains and frequencies
        strain_samp_yearly.append(strain_samp_current)
        strain_samp_count_yearly.append(strain_samp_count_current)
        strain_samp_freq_yearly.append(strain_samp_freq_current)
        
    return strain_samp_yearly, strain_samp_count_yearly, strain_samp_freq_yearly

def inference_features_Ising(strain_samp_yearly):
    """
    calculate the feature matrix for inference (for Ising strains)
    
    Parameters:
    
    strain_samp_yearly: list
            list of strains for each inference time step (between inf_start and inf_end)
    
    Returns: 
    
    X: numpy.ndarray
            feature matrix for inference of {h,J,f} from -F_host
    
    Dependencies:
    
    import numpy as np
    """
    X = []
    for t in range(len(strain_samp_yearly)-1):
        strains_next = strain_samp_yearly[t+1]
        # features (for time-dependent coefficient f)
        gen_features = [0]*(len(strain_samp_yearly)-1)
        gen_features[t] = 1
        # sequence features (for h and J)
        X_next = []
        for strain in strains_next:
            X_sample = strain.tolist()
            for i in range(len(strain)):
                for j in range(i):
                    X_sample.append(strain[i]*strain[j])
            X_sample = np.concatenate((X_sample, gen_features))
            X_next.append(X_sample)
        if len(X) != 0:
            X = np.concatenate((X, X_next), axis=0)
        else:
            X = copy.deepcopy(X_next)
    X = np.array(X)

    return X

def inference_response_FhostPrediction(minus_fhost_yearly):
    """
    calculate response function from -F_host
    
    Parameters:
    
    minus_fhost_yearly: list
            list of -F_host for each strain at each time step between inf_start and inf_end
    
    Returns:
    
    Y: numpy.ndarray
            response function for the inference of {h,J,f} from -F_host
    
    Dependencies:
    
    import numpy as np
    """
    Y = []
    for t in range(len(minus_fhost_yearly)-1):
        minus_fhosts_next = minus_fhost_yearly[t+1]
        Y_next = minus_fhosts_next
        Y = np.concatenate((Y, Y_next))  

    Y = np.array(Y)
    
    return Y
        
def infer_ridge(X, Y, lambda_h, lambda_J, lambda_f, inf_start, inf_end):
    """
    infer the parameters {h,J,f} with ridge regression (Gaussian prior for regularized params)
    
    Parameters:
    
    X: numpy.ndarray
            feature matrix
    Y: numpy.ndarray
            response vector
    lambda_h, lambda_J, lambda_f: int (or float)
            regularization coefficients, if 0 no regularization
    inf_start, inf_end: start and end generation for inference
        
    Returns:
    
    M: numpy.ndarray
            list of inferred coefficients
    M_std: numpy.ndarray
            list of standard deviation for inferred coefficients
            
    Dependencies:
    
    import numpy as np
    import copy
    """
    # number of features
    num_param = len(X[0])
    num_f = int(inf_end - inf_start - 1)
    num_h = int((np.sqrt(1 + 8 * (num_param - num_f)) - 1) / 2)
    num_J = int(num_h * (num_h - 1) / 2) 
    # regularization matrix
    reg_mat = np.zeros((num_param, num_param))
    for i in range(num_h):
        reg_mat[i, i] = lambda_h
    for i in range(num_h, num_h + num_J):
        reg_mat[i, i] = lambda_J
    for i in range(num_h + num_J, num_param):
        reg_mat[i, i] = lambda_f
        
    # standard deviation of features
    X_std = np.std(X, axis=0)
    std_nonzero = np.where(X_std!=0)[0] # use only features where std is nonzero
    param_included = std_nonzero
    X_inf = copy.deepcopy(X[:, param_included])
    reg_mat_reduced = reg_mat[param_included, :]
    reg_mat_reduced = reg_mat_reduced[:, param_included]
    
    # inference by solving X*M = Y for M
    XT = np.transpose(X_inf)
    XTX = np.matmul(XT, X_inf) # covariance
    XTX_reg_inv = np.linalg.inv(XTX + reg_mat_reduced)
    XTY = np.matmul(XT, Y)
    M_inf = np.matmul(XTX_reg_inv, XTY)
    
    M_full = np.zeros(num_param)
    M_full[param_included] = M_inf
    
    # unbiased estimator of variance
    sigma_res = np.sqrt(len(Y)/(len(Y) - len(M_inf))*np.mean([(Y - np.matmul(X_inf, M_inf))**2]))
    v_vec = np.diag(XTX_reg_inv)
    # use std of prior distribution 
    #for parameters that are not informed by model
    M_std = copy.deepcopy(np.diag(reg_mat)) 
    # standard deviation of the parameter distribution 
    # from diagonal of the covariance matrix 
    M_std[param_included] = np.sqrt(v_vec) * sigma_res
    
    return M_full, M_std
    
def hJ_model_lists(h_model, J_model):
    """
    calculate the model fitness coefficients as simple unnested lists
    from h_model, J_model used in simulation (for Ising model)
    
    Parameters:
    
    h_model: numpy.ndarray
            single-mutation fitness coefficients
    J_model: numpy.ndarray
            double-mutation fitness couplings
            
    Returns:
    
    h_list: numpy.ndarray
            simple list of single-mut. fitness coeffs
    J_list: numpy.ndarray
            simple list of fitness coupling coeffs
    hJ_list: numpy.ndarray
            simple list of total fitness effects
            due to double mutations
            
    Dependencies:
    
    import numpy as np
    """
    h_list = h_model.flatten()
    J_list = J_model.flatten()
    hJ_list = []
    k=0
    for i in range(len(h_model)):
        for j in range(i):
            hJ_list.append(h_list[i]+h_list[j]+J_list[k])
            k+=1
    hJ_list = np.array(hJ_list)
    
    return h_list, J_list, hJ_list

def hJ_inf_lists(M, N_site):
    """
    calculate the inferred fitness coefficients as simple unnested lists
    from the inferred parameter vector M (for Ising model)
    
    Parameters:
    
    M: numpy.ndarray
            list of inferred parameter values
    N_site: int
            sequence length
            
    Returns: 
    
    h_list: numpy.ndarray
            simple list of single-mut. fitness coeffs
    J_list: numpy.ndarray
            simple list of fitness coupling coeffs
    hJ_list: numpy.ndarray
            simple list of total fitness effects
            due to double mutations
            
    Dependencies:
    
    import numpy as np       
    """
    num_h = N_site
    num_J = int(N_site*(N_site - 1)/2)
    h_list = M[:num_h]
    J_list = M[num_h: num_h+num_J]
    hJ_list = []
    k=0
    for i in range(N_site):
        for j in range(i):
            hJ_list.append(h_list[i]+h_list[j]+J_list[k])
            k+=1
    hJ_list = np.array(hJ_list)
    
    return h_list, J_list, hJ_list

def hJ_inf_std_lists(M_std, N_site):
    """
    calculate the stds of the inferred fitness coefficients as simple unnested lists
    from the inferred parameter std vector M_std (for Ising model)
    
    Parameters:
    
    M_std: numpy.ndarray
            list of stds for inferred parameter values
    N_site: int
            sequence length
            
    Returns: 
    
    std_h_list: numpy.ndarray
            simple list of stds for single-mut. fitness coeffs
    std_J_list: numpy.ndarray
            simple list of stds for fitness coupling coeffs
    std_hJ_list: numpy.ndarray
            simple list of stds for total fitness effects
            due to double mutations
            
    Dependencies:
    
    import numpy as np       
    """
    num_h = N_site
    num_J = int(N_site*(N_site - 1)/2)
    std_h_list = M_std[:num_h]
    std_J_list = M_std[num_h: num_h+num_J]
    std_hJ_list = []
    k=0
    for i in range(N_site):
        for j in range(i):
            std_hJ_list.append(np.sqrt(std_h_list[i]**2+std_h_list[j]**2+std_J_list[k]**2))
            k+=1
    std_hJ_list = np.array(std_hJ_list)
    
    return std_h_list, std_J_list, std_hJ_list

def this_filepath():
    """
    prints the absolute path to this file
    
    Returns:
    
    filepath: str
            absolute path to file
    
    Dependencies:
    
    import os
    """
    filepath = os.path.normpath(__file__)
    
    return filepath
    

def single_simu_analysis(single_simu_filename, simu_name, exp_idx, ana_param_dict,
                        result_directory='C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/'
                                    'NewApproachFromMarch2021/InfluenzaFitnessInference'):
    """
    run a single inference and analysis on an individual simulation 
    with specific postprocessing parameters
    
    Parameters:
    
    
    single_simu_filename: str
            name including extension .data of the result file for the
            single param combo I want to analyze        
    simu_name: str
            name of the whole pypet based simulation
            usually as date like '2021Apr07'
            (giving the folder in which the simu file is located) 
    exp_idx: int
            index of single simu run
    ana_param_dict: dict
            parameters for inference/analysis
    result_directory (optional): str
            path to the directory 
            where results are stored 
            (in general: path to the InfluenzaFitnessInference repository)
    
    Results:
    
    strain_sample_yearly, strain_sample_count_yearly, strain_sample_frequency_yearly: lists
            randomly sampled strains with respective counts and frequencies
    
    minus_fhost_yearly: list
            list of -F_host for each sampled strain in each time step
            calculated from all sampled data up to inf_end (incl. before inf_start) 
            
    fint_yearly: list
            list of F_int for each sampled strain in each time step
            calculated with fitness coeffs that were used in the simulation
            
    ftot_yearly: list
            list of total fitness F_int + F_host for each sampled strain
            in each time step
    
    M, M_std:  numpy.ndarray
            inferred parameters
            
    summary_stats: dict
            summary statistics for this analysis
            mean stds of fitnesses w. standard errors
            correlation coefficients w. standard errors
    
    analysis_results: dict
            all the above results combined
    
    Returns:
    
    analysis_results: dict
    
    Dependencies:
    
    import pickle
    import numpy as np
    import scipy
    from fitnessinference import simulation as simu
    other functions in this module
    """
    # unpack analysis params
    seed = ana_param_dict['seed'] # RNG seed
    B = ana_param_dict['B'] # sampling size
    inf_start = ana_param_dict['inf_start'] # first time step for inference
    inf_end = ana_param_dict['inf_end']
    # regularization coeffs, if 0 no regularization
    # if !=0, number gives width of gaussian prior around 0
    lambda_h = ana_param_dict['lambda_h']
    lambda_J = ana_param_dict['lambda_J']
    lambda_f = ana_param_dict['lambda_f']
    
    # load data from the simulation
    
    strain_yearly, strain_frequency_yearly, traj =\
            load_simu_data(single_simu_filename, simu_name, exp_idx, result_directory=result_directory)

    # fitness landscape (model input):
    if traj.hJ_coeffs=='constant':
        h_model, J_model = simu.fitness_coeff_constant(traj.N_site, traj.N_state, traj.h_0, traj.J_0)
    elif traj.hJ_coeffs=='p24':
        h_model, J_model = simu.fitness_coeff_p24(traj.N_site, traj.N_state)
        
    
    # take random samples from each time step up to inf_end:
    strain_sample_yearly, strain_sample_count_yearly, strain_sample_frequency_yearly=\
            sample_seqs(strain_yearly, strain_frequency_yearly, seed, B, inf_end)

    # calculate -F_host for each sampled strain at each inference time step
    minus_fhost_yearly = [-simu.fitness_host_list(strain_sample_yearly[t], strain_sample_yearly[:t], 
                                                  strain_sample_frequency_yearly[:t], traj.sigma_h, traj.D0)
                        for t in range(inf_start, inf_end)]

    # calculate F_int for each sampled strain at each inference time step
    fint_yearly = [simu.fitness_int_list(strain_sample_yearly[t], traj.N_state, h_model, J_model)
                        for t in range(inf_start, inf_end)]

    # calculate F_tot= F_int + F_host for each sampled strain at each inference time step
    ftot_yearly = [[fint_yearly[t][i] - minus_fhost_yearly[t][i] for i in range(len(minus_fhost_yearly[t]))] 
                   for t in range(len(minus_fhost_yearly))]
    
    # process data for inference:

    # calculation of features matrix X
    X = inference_features_Ising(strain_sample_yearly[inf_start:inf_end])

    # calculation of response vector Y
    Y = inference_response_FhostPrediction(minus_fhost_yearly)

    # inference (parameter vector M and standard error M_std of inferred params):
    M, M_std = infer_ridge(X, Y, lambda_h, lambda_J, lambda_f, inf_start, inf_end)
    
    # calculate summary statistics for observed fitnesses
    
    f_host_std = np.mean([np.std(fs) for fs in minus_fhost_yearly])
    f_int_std = np.mean([np.std(fs) for fs in fint_yearly])
    f_tot_std = np.mean([np.std(fs) for fs in ftot_yearly])
    # mean selection stringency
    mean_string = np.mean([(np.std(ftot)/np.std(mfhost)) for mfhost, ftot in zip(minus_fhost_yearly, ftot_yearly)])
    # standard error of selection stringency
    SE_string = np.std([(np.std(ftot)/np.std(mfhost)) for mfhost, ftot in zip(minus_fhost_yearly, ftot_yearly)])/len(ftot_yearly)
    
    # compare model fitness coeffs against inferred coeffs

    # model coefficients
    h_model_list, J_model_list, hJ_model_list = hJ_model_lists(h_model, J_model)
    # inferred coefficients
    h_inf_list, J_inf_list, hJ_inf_list = hJ_inf_lists(M, traj.N_site)
    # std of inferred coefficients
    std_h_inf_list, std_J_inf_list, std_hJ_inf_list = hJ_inf_std_lists(M_std, traj.N_site)
    
    # pearson linear correlations:
    
    r_h, pr_h = scipy.stats.pearsonr(h_model_list, h_inf_list)
    # standard error of corr. coeff.
    SE_r_h = np.sqrt(1 - r_h**2)/np.sqrt(len(h_model_list) - 2)
    r_J, pr_J = scipy.stats.pearsonr(J_model_list, J_inf_list)
    SE_r_J = np.sqrt(1-r_J**2)/np.sqrt(len(J_model_list)-2)
    r_hJ, pr_hJ = scipy.stats.pearsonr(hJ_model_list, hJ_inf_list)
    SE_r_hJ = np.sqrt(1-r_hJ**2)/np.sqrt(len(hJ_model_list)-2)
    
    # spearman rank correlations:
    
    rho_h, prho_h = scipy.stats.spearmanr(h_model_list, h_inf_list)
    # standard error of corr. coeff.
    SE_rho_h = np.sqrt(1 - rho_h**2)/np.sqrt(len(h_model_list) - 2)
    rho_J, prho_J = scipy.stats.spearmanr(J_model_list, J_inf_list)
    SE_rho_J = np.sqrt(1 - rho_J**2)/np.sqrt(len(J_model_list) - 2)
    rho_hJ, prho_hJ = scipy.stats.spearmanr(hJ_model_list, hJ_inf_list)
    SE_rho_hJ = np.sqrt(1 - rho_hJ**2)/np.sqrt(len(hJ_model_list) - 2)
    
    summary_stats = {
        'r_h': r_h,
        'pr_h': pr_h,
        'SE_r_h': SE_r_h,
        'r_J': r_J,
        'pr_J': pr_J,
        'SE_r_J': SE_r_J,
        'r_hJ': r_hJ,
        'pr_hJ': pr_hJ,
        'SE_r_hJ': SE_r_hJ,
        'rho_h': rho_h,
        'prho_h': prho_h,
        'SE_rho_h': SE_rho_h,
        'rho_J': rho_J,
        'prho_J': prho_J,
        'SE_rho_J': SE_rho_J,
        'rho_hJ': rho_hJ,
        'prho_hJ': prho_hJ,
        'SE_rho_hJ': SE_rho_hJ,
        'f_host_std': f_host_std,
        'f_int_std': f_int_std,
        'f_tot_std': f_tot_std,
        'mean_string': mean_string,
        'SE_string': SE_string
    }
    
    analysis_results = {
        'strain_sample_yearly': strain_sample_yearly,
        'strain_sample_count_yearly': strain_sample_count_yearly,
        'strain_sample_frequency_yearly': strain_sample_frequency_yearly,
        'minus_fhost_yearly': minus_fhost_yearly,
        'fint_yearly': fint_yearly,
        'ftot_yearly': ftot_yearly,
        'M': M,
        'M_std': M_std,
        'summary_stats': summary_stats
    }
    
#     # save inference/analysis results in pickled file
#     analysis_filename = 'ana_B_%.e' %B + single_simu_filename
#     temp_folder = os.path.join(result_directory, 'results', 'simulations', simu_name + '_temp')
#     test_filepath = os.path.join(temp_folder, analysis_filename)
    
#     with open(test_filepath, 'wb') as f:
#         pickle.dump(analysis_results, f)
        
#     result_path = 'idx_%.i.B_%.i' % (exp_idx, B) + '.summary_stats'
#     traj.f_add_result(result_path, summary_stats,
#                       comment='summary statistics')
#     traj.f_store()
        
    return analysis_results

def multi_simu_analysis(inf_dict,
                        result_directory='C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/'
                                    'NewApproachFromMarch2021/InfluenzaFitnessInference'):
    """
    run inference and analysis for several individual simulations
    with specific postprocessing parameters
    
    Parameters:
    
    inf_dict: dict
            contains values for simu_name, exp_dict, 
            seed, B, inf_start, inf_end, lambda_h, lambda_J, lambda_f
            
    result_directory (optional): str
            path to the directory 
            where results are stored 
            (in general: path to the InfluenzaFitnessInference repository)
    
    Results:
    
    summary_stats_all: dict
            lists of summary statistics for each single analysis
            mean stds of fitnesses w. standard errors
            correlation coefficients w. standard errors
            summary stat lists are in the same order as runs in exp_dict
    pickle .data file
            stores dictionary
            dict = {
            'summary stats' = summary_stats,
            'inf_dict' = inf_dict}
    
    Returns:
    
    summary_stats_all: dict
            lists of summary statistics for each single analysis
            mean stds of fitnesses w. standard errors
            correlation coefficients w. standard errors
            summary stat lists are in the same order as runs in exp_dict
    
    Dependencies:
    
    import pickle
    import numpy as np
    import scipy
    from fitnessinference import simulation as simu
    other functions in this module
    """
#     for key, val in exp_dict.items():
        
        