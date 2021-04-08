import numpy as np
from fitnessinference import analysis as ana

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
    # ana.inference_features_Ising(strain_samp_yearly, strain_samp_count_yearly)
    
def test_inference_response_FhostPrediction():
    """test inference_response_FhostPrediction
    """
    # ana.inference_response_FhostPrediction(minus_fhost_yearly, strain_samp_count_yearly)