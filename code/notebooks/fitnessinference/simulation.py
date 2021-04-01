from pypet import Parameter
import numpy as np


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
    """
    traj.par.N_pop = Parameter('N_pop', 10**2, 'population size')
    traj.par.N_site = Parameter('N_site', 5, 'sequence length')
    traj.par.N_state = Parameter('N_state', 2, 'number of states per site')
    traj.par.mu = Parameter('mu', 10**(-3), 'mutation prob. per site per time step')
    traj.par.sigma_h = Parameter('sigma_h', 1, 'host fitness coefficient')
    traj.par.D0 = Parameter('D0', 5, 'cross-immunity distance')
    traj.par.h_0 = Parameter('h_0', -7, 'typical single mutation fitness cost')
    traj.par.J_0 = Parameter('J_0', 0, 'typical mutation coupling coefficient')
    traj.par.hJ_coeffs = Parameter('hJ_coeffs', 'constant', 'if constant,'
                                   'create all coefficients from h_0 and J_0')
    traj.par.seed = Parameter('seed', 123456, 'RNG seed')
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
            added fitness change due to couplings of two specific mutations to each state
            at each site
    
    Dependencies:
    
    import numpy as np
    """
    numparam_J=int(N_site*(N_site-1)/2)
    J_list=np.ones((numparam_J,N_state-1,N_state-1))*J_0
    h_list=np.ones((N_site,N_state-1))*h_0
    
    return h_list, J_list