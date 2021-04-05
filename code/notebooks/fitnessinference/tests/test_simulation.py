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
    # define test parameters so that it is very likely to have at least a few mutations
    seqs = np.random.randint(0, 2, size=(10**6, 5))
    N_state = 2
    mu = 10**(-3)
    
    # use the function to create mutated sequences
    seqs_m = simu.mutate_seqs(seqs, N_state, mu)
    
    # assert various things
    assert isinstance(seqs_m, np.ndarray) # is the data type preserved?
    assert seqs_m.shape==seqs.shape # is the shape of the array unchanged?
    assert np.any(seqs_m!=seqs) # are there any mutations?
    