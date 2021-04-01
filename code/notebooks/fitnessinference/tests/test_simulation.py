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
    # define which parameters I want to test
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