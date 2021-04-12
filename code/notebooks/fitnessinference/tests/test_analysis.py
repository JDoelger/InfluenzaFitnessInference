import numpy as np
from fitnessinference import analysis as ana
from pypet import Trajectory

def test_load_simu_data():
    """ test load_simu_data
    """
    # test parameters
    single_simu_filename = ('running_N_pop_1e+06N_site_20N_state_2mu_'
                            '1e-04sigma_h_1D0_5h_0_-7J_0_0seed_123456N_simu_2e+02.data')
    simu_name = '2021Apr07' 
    exp_idx = 3
    
    # use function to load simulation results
    strain_yearly, strain_frequency_yearly, traj = ana.load_simu_data(single_simu_filename, simu_name, exp_idx)
    
    # assert various things
    assert isinstance(strain_yearly, list)
    assert len(strain_yearly)==len(strain_frequency_yearly)
    assert isinstance(traj, Trajectory)
    assert isinstance(traj.N_pop, int)

def test_sample_seqs():
    """ test sample_seqs
    """
    # test parameters:
    N_strain = 100 # strains per year
    N_gen = 100 # number of time steps
    N_site = 20 # length of seqs
    N_state = 2 # num of states
    max_count = 100 # max counts for each strain
    strain_yearly = [np.random.randint(0, N_state, size=(N_strain, N_site)) for t in range(N_gen)]
    strain_count_yearly = [np.random.randint(1, max_count+1, size=N_strain) for t in range(N_gen)]
    strain_frequency_yearly = [strain_count_yearly[t]/np.sum(strain_count_yearly[t]) for t in range(N_gen)]
    
    seed = 20390
    B = 100 # sample size
    inf_end = N_gen # num of sampled time steps
     
    # use function to randomly sample sequences
    st_samp_yearly, st_samp_c_yearly, st_samp_f_yearly =\
            ana.sample_seqs(strain_yearly, strain_frequency_yearly, seed, B, inf_end)
    
    # assert various things
    assert isinstance(st_samp_yearly[0], np.ndarray)
    assert len(st_samp_yearly)==inf_end
    assert np.sum(st_samp_f_yearly[0])==1
    
def test_inference_features_Ising():
    """ test inference_features_Ising
    """
    # test parameters
    
    N_steps = 10
    N_seqs = 10**3
    N_site = 20
    N_state = 2
    # sequences
    seq_samp_yearly = [np.random.randint(0, N_state, size=(N_seqs, N_site)) for t in range(N_steps)]
    # unique sequences (strains)
    strain_samp_yearly = [np.unique(seqs, axis=0) for seqs in seq_samp_yearly]
    
    # use function to calculate the feature matrix
    
    X = ana.inference_features_Ising(strain_samp_yearly)
    
    # assert various things
    
    assert isinstance(X, np.ndarray)
    
def test_inference_response_FhostPrediction():
    """test inference_response_FhostPrediction
    """
    # test parameters
    N_strains = 10**2
    N_steps = 10
    minus_fhost_yearly = [np.random.uniform(0, 100, size=N_strains) for t in range(N_steps)]
    
    # use the function to calculate the response vector
    Y = ana.inference_response_FhostPrediction(minus_fhost_yearly)
    
    # assert various things
    assert isinstance(Y, np.ndarray)
    assert isinstance(Y[0], float)
    
def test_infer_ridge():
    """ test infer_ridge
    """
    # test parameters
    inf_start = 0
    inf_end = 50
    num_h = 10
    num_J = int(num_h*(num_h-1)/2)
    num_f = int(inf_end - inf_start - 1)
    num_parameters = num_h + num_J + num_f
    num_samples = 10**4
    lambda_h = 0
    lambda_J = 1
    lambda_f = 0
    X = np.random.randint(0, 2, size=(num_samples, num_parameters))
    Y = np.random.uniform(0, 100, size=num_samples)
    
    # use the function to calculate the infered parameters 
    # from linear regression with regularization (ridge regression)
    M, M_std = ana.infer_ridge(X, Y, lambda_h, lambda_J, lambda_f, inf_start, inf_end)
    
    # assert various things
    assert isinstance(M, np.ndarray)
    assert M.shape==M_std.shape
    assert len(M)==num_parameters
    
# def test_hJ_model_lists():
    
#     # ana.hJ_model_lists(h_model, J_model)
    
# def test_hJ_inf_lists():
    
#     # ana.hJ_inf_lists(M, N_site)
    
# def test_hJ_inf_std_lists():
    
#     # ana.hJ_inf_std_lists(M_std, N_site)

# def test_single_simu_analysis():
    
# #     ana.single_simu_analysis(single_simu_filename, simu_name, exp_idx,
# #                         seed, B, inf_start, inf_end, lambda_h, lambda_J, lambda_f,
# #                         result_directory='C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/'
# #                                     'NewApproachFromMarch2021/InfluenzaFitnessInference')