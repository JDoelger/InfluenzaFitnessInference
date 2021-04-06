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
            added fitness change due to couplings of 
            two specific mutations to each state at each site
    
    Dependencies:
    
    import numpy as np
    """
    numparam_J=int(N_site*(N_site-1)/2)
    J_list=np.ones((numparam_J,N_state-1,N_state-1))*J_0
    h_list=np.ones((N_site,N_state-1))*h_0
    
    return h_list, J_list

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

def fitness_int(seq, N_state, h_model, J_model):
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
    
    Returns:
    
    f_int: float
            intrinsic fitness for the sequence
    
    Dependencies:
    
    import numpy as np
    
    """
    statevec_list=np.array([[int(i==j) for j in range(1,N_state)] 
                        for i in range(N_state)]) 
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
    f_int_list = np.array([fitness_int(seq, N_state, h_model, J_model) 
                           for seq in strain_current])
        
    return f_int_list