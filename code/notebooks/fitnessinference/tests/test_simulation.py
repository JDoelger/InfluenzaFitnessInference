import fitnessinference.simulation as simu
from pypet import Environment
import pytest
import numpy as np

def test_add_parameters():
    """ test add_parameter function 
    """ 
    # create environment and call its trajectory traj
    env = Environment()
    traj = env.traj

    # use the add_parameter function to add parameters
    simu.add_parameters(traj)
    
    # assert that some parameters have the expected type
    assert isinstance(traj.N_pop, int)
    assert isinstance(traj.mu, float)
    # assert that calling some random parameter raises AttributeError
    with pytest.raises(AttributeError) as e_info:
        somepar = traj.someparameter
        
def test_fitness_coeff_constant():
    """ test fitness_coeff_constant
    """
    # define test parameters
    N_site = 5
    N_state = 2
    h_0 = -7
    J_0 = -0.5
    
    # use the function to create h_list and J_list
    h_list, J_list = simu.fitness_coeff_constant(N_site,N_state,h_0,J_0)
    
    # assert the expected data type and the value of some coeffs
    assert isinstance(h_list, np.ndarray)
    assert isinstance(J_list, np.ndarray)
    assert h_list[0][0]==h_0
    assert J_list[0][0][0]==J_0
    
def test_mutate_seqs():
    """ test mutate_seqs
    """
    # define test parameters
    seqs = np.random.randint(0, 2, size=(10**6, 5))
    N_state = 2
    mu = 10**(-3)
    
    # use the function to create mutated sequences
    seqs_m = simu.mutate_seqs(seqs, N_state, mu)
    
    # assert various things
    assert isinstance(seqs_m, np.ndarray) # is the type preserved?
    assert seqs_m.shape==seqs.shape # is the shape of the array unchanged?
    assert np.any(seqs_m!=seqs) # are there any mutations?
    
def test_fitness_int():
    """ test fitness_int
    """
    # define test parameters
    seq = np.random.randint(0, 2, size=(20, ))
    N_state = 2
    h_model, J_model = simu.fitness_coeff_constant(20, N_state, -7, -1)
    statevec_list = np.array([[int(i==j) for j in range(1,N_state)] 
                        for i in range(N_state)])
    
    # use the function to calculate intrinsic fitness
    f_int = simu.fitness_int(seq, N_state, h_model, J_model, statevec_list)
    
    # assert various things
    assert isinstance(f_int, float) # datatype?
    assert f_int <= 0 # not positive, since we choose coeffs negative
    
def test_fitness_int_list():
    """ test fitness_int_list
    """
    # define test parameters
    N_state = 2
    N_site = 20
    strain_current = np.random.randint(0, N_state, size=(10**3, N_site))
    h_model, J_model = simu.fitness_coeff_constant(N_site, N_state, -7, -1)
    
    # use the function to calculate intrinsic fitness list
    f_int_list = simu.fitness_int_list(strain_current, N_state, h_model, J_model)
    
    # assert various things
    assert isinstance(f_int_list, np.ndarray) # type?
    assert np.any(f_int_list < 0) # check that some elements are negative
    assert np.all(f_int_list <= 0) # check that all elements are not positive 
    
def test_fitness_host():
    """ test fitness_host
    """
    # define test parameters
    N_state = 2
    N_site = 20
    N_steps = 100
    seq = np.random.randint(0, N_state, size=(N_site, ))
    st_yearly = [np.random.randint(0, N_state, size=(np.random.randint(500,2*10**3), N_site))
                                   for t in range(N_steps)]
    st_count_yearly = [np.random.randint(0, 100, size=(len(st_yearly[t]), )) 
                     for t in range(N_steps)]
    st_freq_yearly = [counts/np.sum(counts) for counts in st_count_yearly]
    sigma_h = 1
    D0 = 1
    
    # use the function to calculate the host fitness
    f_host = simu.fitness_host(seq, st_yearly, st_freq_yearly, sigma_h, D0)
    
    # assert various things
    assert isinstance(f_host, float) # type?
    assert f_host<0 # negative values?
    
def test_fitness_host_list():
    """test fitness_host_list
    """
    # define test parameters
    N_state = 2
    N_site = 20
    N_steps = 100
    strain_current = np.random.randint(0, N_state, size=(10**3, N_site))
    # choose varying numbers of strains for each prev time step
    st_yearly = [np.random.randint(0, N_state, size=(np.random.randint(500,2*10**3), N_site))
                                   for t in range(N_steps)]
    st_count_yearly = [np.random.randint(0, 100, size=(len(st_yearly[t]), )) 
                     for t in range(N_steps)]
    st_freq_yearly = [counts/np.sum(counts) for counts in st_count_yearly]
    sigma_h = 1
    D0 = 1
    
    # use the function to calculate the host fitness list
    f_host_list = simu.fitness_host_list(strain_current, st_yearly, 
                                         st_freq_yearly, sigma_h, D0)
    
    # assert various things
    assert isinstance(f_host_list, np.ndarray) # type?
    assert np.all(f_host_list <= 0) # values non-positive?
    

    