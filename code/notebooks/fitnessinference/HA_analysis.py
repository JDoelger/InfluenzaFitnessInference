import numpy as np
import copy
import os
import pickle
import scipy
try:
    import simulation as simu
    import analysis as ana
except ModuleNotFoundError:
    from fitnessinference import simulation as simu
    from fitnessinference import analysis as ana
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from datetime import date
import matplotlib as mpl
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from math import log10, floor
import pandas as pd
import os

def retrieve_seqs(fastafile='HA(H3N2)1968-2020_Accessed210418.fasta'):
    """
    extract yearly sequences from fasta file
    """
    repo_path = os.getcwd()
    fastafilepath = os.path.join(repo_path, 'figures', fastafile)

    protein_list = list(SeqIO.parse(fastafilepath,
                                    'fasta'))  # HA (H3N2) protein records from IRD (fludb.org) for 1968-2020, downloaded on 18th Apr. 2021, only date and season in description
    # protein_BI1619068 = list(SeqIO.parse('BI_16190_68_ProteinFasta.fasta',
    #                                      'fasta'))  # HA (H3N2) protein records from IRD (fludb.org) for strain BI/16190/68 (accession: KC296480)
    # seq_BI68 = protein_BI1619068[0].seq  # reference sequence for strain BI/68

    # use only seqs that are complete with no insertions/deletions
    complete_list = []
    for rec in protein_list:
        if len(rec) == 566:
            complete_list.append(rec)

    # remove all sequences with ambiguous amino acid codes
    amb_aa_list = ['B', 'J', 'Z', 'X']
    complete_unamb_list = []
    for rec in complete_list:
        amb_count = 0
        for aa in amb_aa_list:
            if aa in rec.seq:
                amb_count += 1
                break
        if amb_count == 0:
            complete_unamb_list.append(rec)

    # divide sequences into years:  as list of years, which contain list of sequences
    year1 = 1968
    yearend = 2020
    year_list = list(i for i in range(year1, yearend + 1))  # list of years
    yearly = list([] for i in range(0, yearend - year1 + 1))  # list of sequences for each year
    for rec in complete_unamb_list:
        for year in year_list:
            if str(year) in rec.id:
                yearly[year_list.index(year)].append(str(rec.seq))  # append only the sequence, not whole record

    return year_list, yearly

def add_reference_sequences_from_fasta(fastafile, seq_name, results_directory=('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                      '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures')):
    """
    add one reference sequence in dictionary of reference sequences that is saved in the figure directory
    """
    # load current seq_refs
    results_directory = os.path.normpath(results_directory)
    if not os.path.exists(results_directory):
        results_directory = os.path.join(os.getcwd(), 'figures')

    seq_ref_file = os.path.join(results_directory, 'reference_sequences.data')
    if os.path.exists(seq_ref_file):
        with open(seq_ref_file, 'rb') as f:
            seq_ref_dict = pickle.load(f)
    else:
        # if no previous reference sequences saved, initialize empty directory
        seq_ref_dict = {}

    # retrieve sequence from fasta file
    fasta_path = os.path.join(results_directory, fastafile)
    seq_rec_list = list(SeqIO.parse(fasta_path, 'fasta'))
    seq_ref = seq_rec_list[0].seq  # choose first entry of sequence list, although each should only have one entry

    # add the new reference sequence under its chosen name in the dictionary
    seq_ref_dict[seq_name] = seq_ref
    # save the dictionary back in the file
    with open(seq_ref_file, 'wb') as f:
        pickle.dump(seq_ref_dict, f)

def print_seq_refs(results_directory=('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                      '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures')):
    """
    print out the names of added reference sequences in the list
    """
    results_directory = os.path.normpath(results_directory)
    if not os.path.exists(results_directory):
        results_directory = os.path.join(os.getcwd(), 'figures')

    seq_ref_file = os.path.join(results_directory, 'reference_sequences.data')
    if os.path.exists(seq_ref_file):
        with open(seq_ref_file, 'rb') as f:
            seq_ref_dict = pickle.load(f)
    for key in seq_ref_dict.keys():
        print(key)

def strain_info(seqs_list):
    """
    calculate strains and frequencies from list of seq.s at different time points
    seqs_list: list of list of sequences for a number of time points
    returns lists of strains and strain frequencies for each time, total count at each time,
    strains and frequencies across all time points
    """
    total_count_list=[len(seqs) for seqs in seqs_list] # total number of sequences at each time
    strains_list=[[] for seqs in seqs_list]
    strains_freq_list=[[] for seqs in seqs_list]
    strain_All_list=[]
    strain_All_freq_list=[]
    for y in range(len(seqs_list)): # for each time point
        ## finding unique seqs in each time point
        strains_count=[] # counts for each strain before normalization
        for i in range(len(seqs_list[y])):
            if seqs_list[y][i] not in strains_list[y]:
                strains_list[y].append(seqs_list[y][i])
                strains_count.append(1)
            else:
                strains_count[strains_list[y].index(seqs_list[y][i])]+=1
        # rank strains of this year:
        merge_list=list(zip(strains_count,strains_list[y]))
        merge_list.sort(reverse=True) # sort coarse strain list according to count
        strains_count=[y for y,x in merge_list]
        strains_list[y]=[x for y,x in merge_list]
        strains_freq_list[y]=[c/total_count_list[y] for c in strains_count] # calculate strain frequency from count
        ## finding unique seqs across time points
        for sti in range(len(strains_list[y])): # for each strain at this time
            if strains_list[y][sti] not in strain_All_list:
                strain_All_list.append(strains_list[y][sti])
                strain_All_freq_list.append(strains_freq_list[y][sti]) # unnormalized (adding yearly freq)
            else:
                strain_All_freq_list[strain_All_list.index(strains_list[y][sti])]+=strains_freq_list[y][sti]
    merge_list=list(zip(strain_All_freq_list,strain_All_list))
    merge_list.sort(reverse=True) # sort coarse strain list according to count
    strain_All_freq_list=[y/len(seqs_list) for y,x in merge_list] # normalized by number of time points
    strain_All_list=[x for y,x in merge_list]
    return [strains_list, strains_freq_list, total_count_list, strain_All_list,strain_All_freq_list]

def exe_plot_strainSuccession_HA():
    """
    make and save plot of strain succession since 1968 of HA (H3N2) as collected from
    the influenza research database (fludb.org)

    Results:

    plot file: .pdf
                name: HA_strain_succession

    Returns:

    None

    Dependencies:

    import os
    import pickle
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from Bio import SeqIO
    from Bio.Seq import Seq
    other functions in this module
    """
    # plot settings
    plt_set = ana.set_plot_settings()

    fig = plt.figure(figsize=(plt_set['full_page_width'], 3))
    ax1 = fig.add_axes(plt_set['plot_dim_2pan'][0])
    ax2 = fig.add_axes(plt_set['plot_dim_2pan'][1])

    repo_directory = ('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/'
                      'NewApproachFromMarch2021/InfluenzaFitnessInference')
    repo_directory = os.path.normpath(repo_directory)
    if not os.path.exists(repo_directory):
        repo_directory = os.getcwd()

    figure_directory = os.path.join(repo_directory, 'figures')
    this_plot_filepath = os.path.join(figure_directory,
                                      'HA_strain_succession' + plt_set['file_extension'])

    # retrieve HA protein sequences from fasta file
    year_list, yearly = retrieve_seqs()

    # divide sequences into strains
    [strain_yearly, strain_frequency_yearly, tot_count_yearly,
     strain_All, strain_frequency_All] = strain_info(yearly)

    strain_All_timeOrdered = []  # all strains ordered in time (first observed with highest frequency listed first)
    strain_All_freq_timeOrdered = []  # frequency of all strains ordered in time
    # order strains
    for y in range(len(strain_yearly)):
        for sti in range(len(strain_yearly[y])):  # for each strain at this time
            if strain_yearly[y][sti] not in strain_All_timeOrdered:
                strain_All_timeOrdered.append(strain_yearly[y][sti])
                strain_All_freq_timeOrdered.append(strain_frequency_yearly[y][sti])  # unnormalized (adding yearly freq)
            else:
                strain_All_freq_timeOrdered[strain_All_timeOrdered.index(strain_yearly[y][sti])] += \
                strain_frequency_yearly[y][sti]
    # assign strain label to each strain in each year
    strain_All_freq_yearly = [[0 for i in range(len(strain_All_timeOrdered))] for y in
                              range(len(strain_yearly))]  # frequency of all ever observed strains in each year
    strain_index_yearly = [[0 for sti in range(len(strain_yearly[y]))] for y in
                           range(len(strain_yearly))]  # strain labels for strains that are observed in each year
    for y in range(len(strain_yearly)):
        for sti in range(len(strain_yearly[y])):
            label = strain_All_timeOrdered.index(strain_yearly[y][sti])  # strain label
            strain_All_freq_yearly[y][label] = strain_frequency_yearly[y][sti]  # strain frequency update
            strain_index_yearly[y][sti] = label  # save strain label

    strain_frequency_yearly_transpose = list(map(list, zip(*strain_All_freq_yearly)))

    cm = plt.get_cmap('rainbow')
    colorlist = [cm(1. * i / (len(strain_frequency_yearly_transpose)))
                 for i in range(len(strain_frequency_yearly_transpose))]

    for sti in range(len(strain_frequency_yearly_transpose)):
        ax1.plot(year_list, strain_frequency_yearly_transpose[sti], color=colorlist[sti])
    ax1.set_xlabel('year')
    ax1.set_ylabel('strain frequency')
    ax1.text(plt_set['plotlabel_shift_2pan'], 1, '(a)', transform=ax1.transAxes,
             fontsize=plt_set['label_font_size'], va='top', ha='right')

    for y in range(len(strain_index_yearly)):
        for sti in range(len(strain_index_yearly[y]) - 1, -1, -1):
            ax2.plot(y + year_list[0], strain_index_yearly[y][sti], '.',
                     markersize=plt_set['plot_marker_size_dot'], color='blue')
        ax2.plot(y + year_list[0], strain_index_yearly[y][0], '.',
                 markersize=plt_set['plot_marker_size_dot'], color='red')
    ax2.set_xlabel('year')
    ax2.set_ylabel('strain label')
    ax2.text(plt_set['plotlabel_shift_2pan'], 1, '(b)', transform=ax2.transAxes,
             fontsize=plt_set['label_font_size'], va='top', ha='right')

    plt.savefig(this_plot_filepath, bbox_inches='tight')
    plt.close()

def fitness_host(seq, st_yearly, st_freq_yearly, sigma_h, D0, res_targeted):
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

    Results:

    f_host: float
            host-dependent fitness for the sequence at the current time

    Dependencies:

    import numpy as np
    """
    seq = np.array(list(seq))[res_targeted]
    st_yearly = [np.array([np.array(list(seq))[res_targeted] for seq in st_current]) for st_current in st_yearly]
    st_freq_yearly = [np.array(stf_current) for stf_current in st_freq_yearly]

    f_host_noSig = 0  # initialize host fitness without sigma_h factor

    for t in range(len(st_yearly)):  # iterate through all prev. time steps
        strains = st_yearly[t]
        # create array of same dimension as strain list at t
        seq_arr = np.repeat([seq], len(strains), axis=0)
        # calculate mutational distances between seq_arr and strains
        mut_dist = np.sum(seq_arr != strains, axis=1)
        f_host_noSig += -np.dot(st_freq_yearly[t], np.exp(-mut_dist / D0))

    f_host = sigma_h * f_host_noSig

    return f_host

def minus_fhost_list(strain_current, st_yearly, st_freq_yearly, sigma_h, D0, res_targeted):
    """
    calculate minus the host population-dependent fitness contribution for all strains
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

    Mf_host_list = np.array([-fitness_host(seq, st_yearly, st_freq_yearly, sigma_h, D0, res_targeted)
                             for seq in strain_current])

    return Mf_host_list

def def_res_epitope_list():
    """
    stores list of residue indices (in my numbering) for HA epitopes A, B, C, D, E with residue positions taken
    and translated from (Suzuki 2006, Mol. Biol. Evol.)
    """

    res_epitope_list = [[137, 139, 141, 145, 146, 147, 148, 150, 152, 153, 155, 157, 158, 159, 160, 161, 165, 167, 183],
     [143, 144, 170, 171, 172, 173, 174, 175, 178, 179, 180, 201, 202, 203, 204, 205, 207, 208, 209, 211, 212, 213],
     [59, 60, 61, 62, 63, 65, 66, 68, 69, 288, 290, 291, 293, 294, 295, 309, 312, 314, 315, 319, 320, 322, 323, 324,
      325, 326, 327],
     [111, 117, 118, 132, 136, 182, 185, 186, 187, 188, 189, 190, 191, 192, 194, 197, 216, 218, 222, 223, 224, 227, 228,
      229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 253, 255, 257, 259, 261, 262, 263],
     [72, 74, 77, 78, 82, 90, 93, 95, 96, 97, 98, 101, 102, 103, 106, 107, 109, 124, 275, 276, 277, 280]]

    return res_epitope_list

def convert_my_ind_to_Lee_HA_numbering(my_indices):
    """
    convert list of indices in my numbering to HA numbering used by Lee et al. (PNAS 2018)
    """
    Lee_indices = []
    for ind in my_indices:
        if ind <= 15:
            Lee_ind = ind - 16
        else:
            Lee_ind = ind - 15
        Lee_indices.append(Lee_ind)

    return Lee_indices

def convert_Lee_HA_numbering_to_my_ind(Lee_indices):
    """
    convert list of indices in HA numbering used by Lee et al. (PNAS 2018) to my numbering
    """
    my_indices = []
    for ind in Lee_indices:
        if ind < 0:
            my_ind = ind + 16
        elif ind > 0:
            my_ind = ind + 15
        else:
            print('error: Lee index=0!!')
        my_indices.append(my_ind)

    return my_indices


def exe_minus_fhost_yearly(sigma_h, D0, results_directory=('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                      '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures')):
    """
    calculates -fhost for each strain in each given strain in each year
    and saves it in pickled file "HA_MinusFhost_yearly.data"
    """
    ## define res_targeted as all head epitope residues
    # list of residue indices (in my numbering) for epitopes A, B, C, D, E with residue positions taken
    # and translated from (Suzuki 2006, Mol. Biol. Evol.):
    res_epitope_list = def_res_epitope_list()
    res_allepitopes_list = [res for res_list in res_epitope_list for res in res_list]
    res_targeted = res_allepitopes_list

    # retrieve HA sequences
    year_list, yearly = retrieve_seqs()

    # divide sequences into strains
    [strain_yearly, strain_frequency_yearly, tot_count_yearly,
     strain_All, strain_frequency_All] = strain_info(yearly)

    # calculate -Fhost for each strain in each year
    MinusFhost_yearly = []
    for y in range(len(strain_yearly) - 1):
        MinusFhost_list = \
            minus_fhost_list(strain_yearly[y + 1], strain_yearly[:y + 1], strain_frequency_yearly[:y + 1], sigma_h,
                               D0, res_targeted)
        MinusFhost_yearly.append(MinusFhost_list)

    # save minus_fhost_yearly as pickle file in figures folder
    results_directory = os.path.normpath(results_directory)
    if not os.path.exists(results_directory):
        results_directory = os.path.join(os.getcwd(), 'figures')

    file_name = 'HA_MinusFhost_yearly' + 'sigma_h_'+ str(sigma_h) + '_D0_' + str(D0) + '.data'
    file_path = os.path.join(results_directory, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(MinusFhost_yearly, f)

def exe_plot_minus_fhost_yearly(sigma_h, D0,
                                results_directory=('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                      '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures'),
                                figure_directory=('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                        '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures')):
    # load minus_fhost_yearly from pickle file in figures folder
    results_directory = os.path.normpath(results_directory)
    if not os.path.exists(results_directory):
        results_directory = os.path.join(os.getcwd(), 'figures')

    file_name = 'HA_MinusFhost_yearly' + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + '.data'
    file_path = os.path.join(results_directory, file_name)
    with open(file_path, 'rb') as f:
        MinusFhost_yearly = pickle.load(f)

    figure_directory = os.path.normpath(figure_directory)
    if not os.path.exists(figure_directory):
        figure_directory = os.path.join(os.getcwd(), 'figures')

    plt_set = ana.set_plot_settings()

    fig_name = 'HA_MFhost_dist' + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + plt_set['file_extension']
    this_plot_filepath = os.path.join(figure_directory, fig_name)
    fig = plt.figure(figsize=(plt_set['full_page_width']/2, 3))
    ax1 = fig.add_axes(plt_set['plot_dim_1pan'][0])

    # retrieve HA sequences in order to get year_list
    year_list, yearly = retrieve_seqs()

    for y in range(len(MinusFhost_yearly)):
        ax1.plot([year_list[y]] * len(MinusFhost_yearly[y]), MinusFhost_yearly[y] - np.mean(MinusFhost_yearly[y]), '.',
                 color='black')
    ax1.set_xlabel('year')
    ax1.set_ylabel('$-F_{host}$ - $<-F_{host}>$')
    plt.savefig(this_plot_filepath, bbox_inches='tight')

def binary_strains(seq_ref, st_yearly, st_freq_yearly, minus_f_host_yearly, res_targeted):
    """
    translate strains into binary representation of head epitope region based on chosen reference sequence
    and update the respective response values minus_f_host_yearly for the respective binary strains
    """
    ## turn list of strings into arrays with sequences reduced to the HA head epitope sites
    seq_ref = np.array(list(seq_ref))[res_targeted]
    st_yearly = [np.array([np.array(list(seq))[res_targeted] for seq in st_current]) for st_current in st_yearly]
    st_freq_yearly = [np.array(stf_current) for stf_current in st_freq_yearly]

    ## compare each strain in each year to the reference seq and create lists of the sequence reps and frequencies of
    # the new binary strains
    st_bin_yearly = [] # binary strain list
    for t in range(len(st_yearly)):  # iterate through all prev. time steps
        strains = st_yearly[t]
        # create array of same dimension as strain list at t
        seq_arr = np.repeat([seq_ref], len(strains), axis=0)
        # calculate binary strains based on difference to reference seq
        binary_strains = (seq_arr!=strains).astype(int)
        st_bin_yearly.append(binary_strains)

    # update strain and strain frequency lists as well as minus_f_host_yearly for binary strains
    st_bin_yearly_new = [[] for t in range(len(st_yearly))] # new list of binary strains
    st_yearly_new = [[] for t in range(len(st_yearly))] # non-redundant lists of nonbin strains
    minus_f_host_yearly_new = [[] for t in range(len(minus_f_host_yearly))]
    st_bin_freq_yearly = [[] for t in range(len(st_yearly))]
    for t in range(len(st_bin_yearly)):
        for i in range(len(st_bin_yearly[t])):
            # if current binary strain saved already
            # print(type(st_bin_yearly[t][i]), type(st_bin_yearly_new[t]))
            if st_bin_yearly[t][i].tolist() in st_bin_yearly_new[t]:
                # if corresponding non-bin strain not saved yet
                if st_yearly[t][i].tolist() not in st_yearly_new[t]:
                    # add new strain to list and add its frequency to the frequency list
                    st_bin_yearly_new[t].append(st_bin_yearly[t][i].tolist())
                    st_bin_freq_yearly[t].append(st_freq_yearly[t][i])
                    if t != 0:
                        minus_f_host_yearly_new[t-1].append(minus_f_host_yearly[t-1][i])
                # if corresponding non-bin strain already saved
                else:
                    st_index = st_yearly_new[t].tolist().index(st_yearly[t][i])
                    st_bin_freq_yearly[t][st_index] += st_freq_yearly[t][i]
            # if current binary strain not saved already
            else:
                st_bin_yearly_new[t].append(st_bin_yearly[t][i].tolist())
                st_bin_freq_yearly[t].append(st_freq_yearly[t][i])
                if t != 0:
                    minus_f_host_yearly_new[t-1].append(minus_f_host_yearly[t-1][i])

    return st_bin_yearly_new, st_bin_freq_yearly, minus_f_host_yearly_new

def inference_features_Ising_noCouplings(strain_samp_yearly):
    """
    calculate the feature matrix for inference (for Ising strains)

    Parameters:

    strain_samp_yearly: list
            list of strains for each inference time step (between inf_start and inf_end)

    Returns:

    X: numpy.ndarray
            feature matrix for inference of {h,f} from -F_host

    Dependencies:

    import numpy as np
    """
    X = []
    for t in range(len(strain_samp_yearly)):
        strains_next = strain_samp_yearly[t]
        # features (for time-dependent coefficient f)
        gen_features = [0] * (len(strain_samp_yearly))
        gen_features[t] = 1
        # sequence features (for h and J)
        X_next = []
        for strain in strains_next:
            # X_sample = strain.tolist()
            X_sample = strain
            X_sample = np.concatenate((X_sample, gen_features))
            X_next.append(X_sample)
        if len(X) != 0:
            X = np.concatenate((X, X_next), axis=0)
        else:
            X = copy.deepcopy(X_next)
    X = np.array(X)

    return X

def inference_features_Ising_WithCouplings(strain_samp_yearly):
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
    for t in range(len(strain_samp_yearly)):
        strains_next = strain_samp_yearly[t]
        # features (for time-dependent coefficient f)
        gen_features = [0] * (len(strain_samp_yearly))
        gen_features[t] = 1
        # sequence features (for h and J)
        X_next = []
        for strain in strains_next:
            # X_sample = strain.tolist()
            X_sample = strain
            for i in range(len(strain)):
                for j in range(i):
                    X_sample = np.concatenate((X_sample, np.array([strain[i]*strain[j]])))
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
            response function for the inference of intrinsic fitness coeffs

    Dependencies:

    import numpy as np
    """
    Y = []
    for t in range(len(minus_fhost_yearly)):
        minus_fhosts_next = minus_fhost_yearly[t]
        Y_next = minus_fhosts_next
        Y = np.concatenate((Y, Y_next))

    Y = np.array(Y)

    return Y

def infer_ridge_noCouplings(X, Y, lambda_h, lambda_f, inf_start, inf_end):
    """
        infer the parameters {h,f} with ridge regression (Gaussian prior for regularized params)

        Parameters:

        X: numpy.ndarray
                feature matrix
        Y: numpy.ndarray
                response vector
        lambda_h, lambda_f: int (or float)
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
    num_h = int(num_param - num_f)
    # regularization matrix
    reg_mat = np.zeros((num_param, num_param))
    for i in range(num_h):
        reg_mat[i, i] = lambda_h
    for i in range(num_h, num_param):
        reg_mat[i, i] = lambda_f

    # standard deviation of features
    X_std = np.std(X, axis=0)
    std_nonzero = np.where(X_std != 0)[0]  # use only features where std is nonzero
    param_included = std_nonzero
    X_inf = copy.deepcopy(X[:, param_included])
    reg_mat_reduced = reg_mat[param_included, :]
    reg_mat_reduced = reg_mat_reduced[:, param_included]

    # inference by solving X*M = Y for M
    XT = np.transpose(X_inf)
    XTX = np.matmul(XT, X_inf)  # covariance
    try:
        XTX_reg_inv = np.linalg.inv(XTX + reg_mat_reduced)
        XTY = np.matmul(XT, Y)
        M_inf = np.matmul(XTX_reg_inv, XTY)

        M_full = np.zeros(num_param)
        M_full[param_included] = M_inf

        # unbiased estimator of variance
        sigma_res = np.sqrt(len(Y) / (len(Y) - len(M_inf)) * np.mean([(Y - np.matmul(X_inf, M_inf)) ** 2]))
        v_vec = np.diag(XTX_reg_inv)
        # use std of prior distribution (if <infinity, else use 0)
        # for parameters that are not informed by model
        # M_var_inv = copy.deepcopy(np.diag(reg_mat))
        M_std = np.zeros(M_full.shape)
        for i in range(len(M_std)):
            if reg_mat[i, i] != 0:
                M_std[i] = np.sqrt(1 / reg_mat[i, i])
        # standard deviation of the parameter distribution
        # from diagonal of the covariance matrix
        M_std[param_included] = np.sqrt(v_vec) * sigma_res
    except:
        print('exception error')
        M_full = np.zeros(num_param)
        M_std = np.zeros(num_param)

    return M_full, M_std

def infer_ridge_WithCouplings(X, Y, lambda_h, lambda_J, lambda_f, inf_start, inf_end):
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
    num_h = int(-1/2 + np.sqrt(1/4 + 2*(num_param - num_f))) # calculate num_h from num_hJ = num_h*(num_h + 1)/2
    num_J = num_param - (num_f + num_h)
    # regularization matrix
    reg_mat = np.zeros((num_param, num_param))
    for i in range(num_h):
        reg_mat[i, i] = lambda_h
    for i in range(num_h, num_h + num_J):
        reg_mat[i,i] = lambda_J
    for i in range(num_h + num_J, num_param):
        reg_mat[i, i] = lambda_f

    # standard deviation of features
    X_std = np.std(X, axis=0)
    std_nonzero = np.where(X_std != 0)[0]  # use only features where std is nonzero
    param_included = std_nonzero
    X_inf = copy.deepcopy(X[:, param_included])
    reg_mat_reduced = reg_mat[param_included, :]
    reg_mat_reduced = reg_mat_reduced[:, param_included]

    # inference by solving X*M = Y for M
    XT = np.transpose(X_inf)
    XTX = np.matmul(XT, X_inf)  # covariance
    try:
        XTX_reg_inv = np.linalg.inv(XTX + reg_mat_reduced)
        XTY = np.matmul(XT, Y)
        M_inf = np.matmul(XTX_reg_inv, XTY)

        M_full = np.zeros(num_param)
        M_full[param_included] = M_inf

        # unbiased estimator of variance
        sigma_res = np.sqrt(len(Y) / (len(Y) - len(M_inf)) * np.mean([(Y - np.matmul(X_inf, M_inf)) ** 2]))
        v_vec = np.diag(XTX_reg_inv)
        # use std of prior distribution (if <infinity, else use 0)
        # for parameters that are not informed by model
        # M_var_inv = copy.deepcopy(np.diag(reg_mat))
        M_std = np.zeros(M_full.shape)
        for i in range(len(M_std)):
            if reg_mat[i, i] != 0:
                M_std[i] = np.sqrt(1 / reg_mat[i, i])
        # standard deviation of the parameter distribution
        # from diagonal of the covariance matrix
        M_std[param_included] = np.sqrt(v_vec) * sigma_res
    except:
        print('exception error')
        M_full = np.zeros(num_param)
        M_std = np.zeros(num_param)

    return M_full, M_std


def exe_inference_noCouplings(seq_ref_name, sigma_h, D0, res_targeted,
                              lambda_h, lambda_f, inf_start, inf_end,
                              results_directory=('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                                                 '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures')
                              ):
    """
    infer single-mutation intrinsic fitness coefficients h (without couplings), together with temporal params F*
    based on specific reference sequence, from which other strains are mutated within the head epitope regions (given by res_targeted)
    """
    ## retrieve st_yearly and st_freq_yearly from collected HA strains (before dim reduction)
    # retrieve HA protein sequences from fasta file
    year_list, yearly = retrieve_seqs()
    print('start: ', year_list[inf_start], 'end: ', year_list[inf_end-1])
    # divide sequences into strains
    [st_yearly, st_freq_yearly, tot_count_yearly,
     strain_All, strain_frequency_All] = strain_info(yearly)

    # load minus_fhost_yearly from pickle file based on values of sigma_h and D0
    results_directory = os.path.normpath(results_directory)
    if not os.path.exists(results_directory):
        results_directory = os.path.join(os.getcwd(), 'figures')

    file_name = 'HA_MinusFhost_yearly' + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + '.data'
    file_path = os.path.join(results_directory, file_name)
    with open(file_path, 'rb') as f:
        minus_f_host_yearly = pickle.load(f)

    seq_ref_file = os.path.join(results_directory, 'reference_sequences.data')
    with open(seq_ref_file, 'rb') as f:
        seq_ref_dict = pickle.load(f)
    seq_ref = seq_ref_dict[seq_ref_name]
    # calculate binary strain rep. and update minus_f_host_yearly respectively
    st_bin_yearly_new, st_bin_freq_yearly, minus_f_host_yearly_new =\
        binary_strains(seq_ref, st_yearly, st_freq_yearly, minus_f_host_yearly, res_targeted)

    # calculate feature matrix and response vector
    strain_samp_yearly = st_bin_yearly_new[inf_start+1:inf_end]
    minus_f_host_yearly = minus_f_host_yearly_new[inf_start:inf_end-1]
    X = inference_features_Ising_noCouplings(strain_samp_yearly)
    Y = inference_response_FhostPrediction(minus_f_host_yearly)

    # do inference and extract h and h_std from inference
    M, M_std = infer_ridge_noCouplings(X, Y, lambda_h, lambda_f, inf_start, inf_end)
    num_h = len(M) - (inf_end - inf_start - 1)
    h_inf_list = M[:num_h]
    h_inf_std_list = M_std[:num_h]

    # print basic results:
    print('inferred h: ', h_inf_list)
    print('number of sites: ', len(h_inf_list))

    # save results from inference and used parameters in dictionary
    ana_result_dict = {
        'seq_ref_name': seq_ref_name,
        'seq_ref': seq_ref,
        'st_yearly': st_yearly,
        'st_freq_yearly': st_freq_yearly,
        'inf_start': inf_start,
        'inf_end': inf_end,
        'sigma_h': sigma_h,
        'D0': D0,
        'res_targeted': res_targeted,
        'lambda_h': lambda_h,
        'lambda_f': lambda_f,
        'h_inf_list': h_inf_list,
        'h_inf_std_list': h_inf_std_list,
        'M': M,
        'M_std': M_std
    }
    result_filename = 'HA_Inference_noCouplings' + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + '.data'
    # switch to results folder for specific reference seq
    seqref_results_folder = os.path.join(results_directory, seq_ref_name)
    if not os.path.exists(seqref_results_folder):
        os.mkdir(seqref_results_folder)
    result_filepath = os.path.join(seqref_results_folder, result_filename)
    with open(result_filepath, 'wb') as f:
        pickle.dump(ana_result_dict, f)

def exe_inference_WithCouplings(seq_ref_name, sigma_h, D0, res_targeted,
                              lambda_h, lambda_J, lambda_f, inf_start, inf_end,
                              results_directory=('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                                                 '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures')
                              ):
    """
    infer single-mutation intrinsic fitness coefficients h and J, together with temporal params F*
    based on specific reference sequence, from which other strains are mutated within the head epitope regions (given by res_targeted)
    """
    ## retrieve st_yearly and st_freq_yearly from collected HA strains (before dim reduction)
    # retrieve HA protein sequences from fasta file
    year_list, yearly = retrieve_seqs()
    # divide sequences into strains
    [st_yearly, st_freq_yearly, tot_count_yearly,
     strain_All, strain_frequency_All] = strain_info(yearly)

    # load minus_fhost_yearly from pickle file based on values of sigma_h and D0
    results_directory = os.path.normpath(results_directory)
    if not os.path.exists(results_directory):
        results_directory = os.path.join(os.getcwd(), 'figures')

    file_name = 'HA_MinusFhost_yearly' + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + '.data'
    file_path = os.path.join(results_directory, file_name)
    with open(file_path, 'rb') as f:
        minus_f_host_yearly = pickle.load(f)

    seq_ref_file = os.path.join(results_directory, 'reference_sequences.data')
    with open(seq_ref_file, 'rb') as f:
        seq_ref_dict = pickle.load(f)
    seq_ref = seq_ref_dict[seq_ref_name]
    # calculate binary strain rep. and update minus_f_host_yearly respectively
    st_bin_yearly_new, st_bin_freq_yearly, minus_f_host_yearly_new =\
        binary_strains(seq_ref, st_yearly, st_freq_yearly, minus_f_host_yearly, res_targeted)

    # calculate feature matrix and response vector
    strain_samp_yearly = st_bin_yearly_new[inf_start+1:inf_end]
    minus_f_host_yearly = minus_f_host_yearly_new[inf_start:inf_end-1]
    X = inference_features_Ising_WithCouplings(strain_samp_yearly)
    Y = inference_response_FhostPrediction(minus_f_host_yearly)

    # do inference and extract h and h_std from inference
    M, M_std = infer_ridge_WithCouplings(X, Y, lambda_h, lambda_J, lambda_f, inf_start, inf_end)
    num_h = int(-1/2 + np.sqrt(1/4 + 2*(len(M) - (inf_end - inf_start - 1)))) # calculate num_h from num_hJ=num_params-num_f
    h_inf_list = M[:num_h]
    h_inf_std_list = M_std[:num_h]

    # print basic results:
    print('inferred h: ', h_inf_list)
    print('number of sites: ', len(h_inf_list))

    # save results from inference and used parameters in dictionary
    ana_result_dict = {
        'seq_ref_name': seq_ref_name,
        'seq_ref': seq_ref,
        'st_yearly': st_yearly,
        'st_freq_yearly': st_freq_yearly,
        'inf_start': inf_start,
        'inf_end': inf_end,
        'sigma_h': sigma_h,
        'D0': D0,
        'res_targeted': res_targeted,
        'lambda_h': lambda_h,
        'lambda_f': lambda_f,
        'h_inf_list': h_inf_list,
        'h_inf_std_list': h_inf_std_list,
        'M': M,
        'M_std': M_std
    }
    result_filename = 'HA_Inference_WithCouplings' + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + '.data'
    # switch to results folder for specific reference seq
    seqref_results_folder = os.path.join(results_directory, seq_ref_name)
    if not os.path.exists(seqref_results_folder):
        os.mkdir(seqref_results_folder)
    result_filepath = os.path.join(seqref_results_folder, result_filename)
    with open(result_filepath, 'wb') as f:
        pickle.dump(ana_result_dict, f)


def round_to_1(x):
    """
    round to 1 significant digit
    """
    if x == 0:
        rounded_x = 0
    else:
        rounded_x = round(x, -int(floor(log10(abs(x)))))
    return rounded_x

def eval_inference_noCouplings(seq_ref_name, sigma_h, D0,
                               results_directory=('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                                                 '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures')):
    """
    retrieve inferred fitness parameters for specific reference seq and fitness params
    plot inferred param for each Lee HA residue index
    """
    results_directory = os.path.normpath(results_directory)
    if not os.path.exists(results_directory):
        results_directory = os.path.join(os.getcwd(), 'figures')

    result_filename = 'HA_Inference_noCouplings' + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + '.data'
    seqref_results_folder = os.path.join(results_directory, seq_ref_name)
    result_filepath = os.path.join(seqref_results_folder, result_filename)
    with open(result_filepath, 'rb') as f:
        ana_result_dict = pickle.load(f)

    ## inferred fitness params
    h_inf_list = ana_result_dict['h_inf_list']
    h_inf_std_list = ana_result_dict['h_inf_std_list']
    print('h_inf_list: ', h_inf_list)
    print('h_inf_std_list: ', h_inf_std_list)

    ## plot inferred params as function of residue numbers in Lee numbering

    res_epitope_list = def_res_epitope_list()
    res_allepitopes_list = [res for res_list in res_epitope_list for res in res_list]
    res_targeted = res_allepitopes_list
    Lee_indices = convert_my_ind_to_Lee_HA_numbering(res_targeted)

    plt_set = ana.set_plot_settings()

    # plot h inferred on y_axis against HA position (Lee numbering)
    fig_name = 'hInferred_vs_Lee_HAposition_' + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + plt_set['file_extension']
    this_plot_filepath = os.path.join(seqref_results_folder, fig_name)
    fig = plt.figure(figsize=(plt_set['full_page_width']*10, 2))
    ax1 = fig.add_axes(plt_set['plot_dim_1pan'][0])
    # label x-axis with each epitope position and label each point with rounded inferred h value
    h_inf_labels = [round_to_1(h) for h in h_inf_list] # round to 1 significant digit
    ax1.set_xticks(Lee_indices)
    for i, txt in enumerate(h_inf_labels):
        ax1.annotate(txt, (Lee_indices[i], h_inf_list[i]))

    ax1.errorbar(Lee_indices, h_inf_list, h_inf_std_list, marker='o', linestyle='none', zorder=1)
    ax1.set_ylim(-1.5,1.5)
    ax1.set_xlabel('HA position (Lee numbering scheme)')
    ax1.set_ylabel('inferred $h$')
    plt.savefig(this_plot_filepath, bbox_inches='tight')

def comparison_inference_LeeDeepMutScanning(sigma_h, D0, inf_scheme = 'noCouplings'):
    """
    plot inferred params, inferred w specific sigma_h and D0
    against mutational effects measured by Lee et al.
    calculate rank correlations, print them out and save those results in the result dictionary of the inference results
    """
    # get aa preference table (from csv file) as pandas dataframe
    data_filename = 'github_jbloomlab_Perth2009-DMS-Manuscript_summary_avgprefs.csv'
    data_folder = os.path.normpath('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                                   '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures/Perth_16_2009_G78D_T212I')
    if not os.path.exists(data_folder):
        data_folder = os.path.join(os.getcwd(), 'figures', 'Perth_16_2009_G78D_T212I')

    data_path = os.path.join(data_folder, data_filename)

    data = pd.read_csv(data_path)

    # get reference sequence for strain Perth_16_2009_G78D_T212I
    strain_name = 'Perth_16_2009_G78D_T212I'

    strain_list_folder = os.path.normpath('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape'
                                          '/NewApproachFromMarch2021/InfluenzaFitnessInference/figures')
    if not os.path.exists(strain_list_folder):
        strain_list_folder = os.path.join(os.getcwd(), 'figures')
    strain_list_filename = 'reference_sequences.data'
    strain_list_filepath = os.path.join(strain_list_folder, strain_list_filename)

    with open(strain_list_filepath, 'rb') as f:
        seq_ref_dict = pickle.load(f)

    seq_ref = seq_ref_dict[strain_name]

    # epitope sites (in my numbering) for which I did the inference
    res_epitope_list = def_res_epitope_list()
    res_allepitopes_list = [res for res_list in res_epitope_list for res in res_list]

    ## extract preferences and aa_list as list/array (sequence position in array has my numbering)

    # list of amino acids
    aa_list = list(data.columns)[1:]

    # transform preference table into array of shape N_site rows * num_aa=20 cols
    aa_pref_arr = data.to_numpy()[:, 1:]

    # extract preference array and ref sequence for epitope sites only (for which I did the inference)
    aa_pref_epi = aa_pref_arr[res_allepitopes_list, :]
    seq_ref_epi = np.array(seq_ref)[res_allepitopes_list]

    ## calculate measured mutational effects as log(max(p_mut(i))/p_ref(i)) as
    ## the intrinsic mutational effect for the easiest mutation at site i away from the aa of the reference seq
    ## or as  avg(log(p_mut(i)/p_ref(i))), i.e. the average mutational effect

    max_mut_effect_list = []
    avg_mut_effect_list = []
    for i in range(len(seq_ref_epi)):
        aa_ref = seq_ref_epi[i]  # reference state
        ref_index = aa_list.index(aa_ref)  # index for ref state in array
        p_ref_list = aa_pref_epi[i, :]
        p_ref = p_ref_list[ref_index]  # preference for ref state
        p_mut_list = np.delete(p_ref_list, ref_index)  # preference for mutated states
        p_max = np.amax(p_mut_list)  # maximum preference to another state
        max_mut_effect = np.log(p_max / p_ref)
        mut_effects = np.log(p_mut_list / p_ref)  # list of log preference ratios
        avg_mut_effect = np.mean(mut_effects)
        max_mut_effect_list.append(max_mut_effect)
        avg_mut_effect_list.append(avg_mut_effect)

    ## calculate shannon entropy from aa preferences
    shannon_e_list = []

    for i in range(len(seq_ref_epi)):
        p_list = aa_pref_epi[i, :]
        shannon_e = -np.sum(np.log(p_list) * p_list)
        shannon_e_list.append(shannon_e)

    ## get the inferred fitness coefficients for this reference sequence
    ## and the specified coefficients sigma_h, D0
    result_filename = 'HA_Inference_' + inf_scheme + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + '.data'
    seqref_results_folder = data_folder
    result_filepath = os.path.join(seqref_results_folder, result_filename)
    with open(result_filepath, 'rb') as f:
        ana_result_dict = pickle.load(f)

    # inferred fitness params (in same order as mut_effect_list)
    h_inf_list = ana_result_dict['h_inf_list']
    h_inf_std_list = ana_result_dict['h_inf_std_list']

    ## calculate the rank correlation between inferred and measured mutational effects and with measured shannon entropy
    rhoMaxEffect_pears, prho_MaxEffect_pears = scipy.stats.pearsonr(max_mut_effect_list, h_inf_list)
    rhoMaxEffect, prho_MaxEffect = scipy.stats.spearmanr(max_mut_effect_list, h_inf_list)
    rhoAvgEffect, prho_AvgEffect = scipy.stats.spearmanr(avg_mut_effect_list, h_inf_list)
    rho_shannon, prho_shannon = scipy.stats.spearmanr(shannon_e_list, h_inf_list)

    print('rhoMaxEffect=', rhoMaxEffect, 'p=', prho_MaxEffect)
    print('rhoMaxEffect_pears=', rhoMaxEffect_pears, 'p=', prho_MaxEffect_pears)
    print('rhoAvgEffect=', rhoAvgEffect, 'p=', prho_AvgEffect)
    print('rho_shannon=', rho_shannon, 'p=', prho_shannon)

    # save comparison measures in result_dict
    ana_result_dict['rho_MaxEffect'] = rhoMaxEffect
    ana_result_dict['prho_MaxEffect'] = prho_MaxEffect
    ana_result_dict['rho_AvgEffect'] = rhoAvgEffect
    ana_result_dict['prho_AvgEffect'] = prho_AvgEffect
    ana_result_dict['rho_shannon'] = rho_shannon
    ana_result_dict['prho_shannon'] = prho_shannon
    with open(result_filepath, 'wb') as f:
        pickle.dump(ana_result_dict, f)

    # plot comparison inferred vs measured coefficients
    plt_set = ana.set_plot_settings()

    fig_name = 'hInferred_vs_Exp_' + inf_scheme + 'sigma_h_' + str(sigma_h) + '_D0_' + str(D0) + plt_set['file_extension']
    this_plot_filepath = os.path.join(data_folder, fig_name)
    # fig = plt.figure(figsize=(plt_set['full_page_width'], 3))
    fig = plt.figure(figsize=(plt_set['single_pan_width'], 3))
    ax1= fig.add_axes(plt_set['plot_dim_1pan'][0])
    # ax2 = fig.add_axes(plt_set['plot_dim_3pan'][1])
    # ax3 = fig.add_axes(plt_set['plot_dim_3pan'][2])

    # inferred vs max mutational effects
    ax1.errorbar(max_mut_effect_list, h_inf_list, h_inf_std_list, marker='o', linestyle='none', zorder=1)
    ax1.set_xlabel('measured log preference ratios')
    ax1.set_ylabel('inferred $h$')
    ax1.set_ylim(-1.5, 1.5)
    text = '$r_{h}$ = %.2f, p = %.e' % (rhoMaxEffect_pears, prho_MaxEffect_pears)
    ax1.text(0.05, 0.95, text, ha='left', va='top', fontsize=12, transform=ax1.transAxes)
    # ax1.text(plt_set['plotlabel_shift_3pan'], plt_set['plotlabel_up_3pan'], '(a)', transform=ax1.transAxes,
    #          fontsize=plt_set['label_font_size'], va='top', ha='right')

    # # inferred vs avg. mutational effects
    # ax2.errorbar(avg_mut_effect_list, h_inf_list, h_inf_std_list, marker='o', linestyle='none', zorder=1)
    # ax2.set_xlabel('measured avg. log aa preference ratios')
    # ax2.set_ylabel('inferred $h$')
    # ax2.set_ylim(-1.5, 1.5)
    # text = '$r_{spearman}$ = %.2f, p = %.e' % (rhoAvgEffect, prho_AvgEffect)
    # ax2.text(0.05, 0.95, text, ha='left', va='top', fontsize=12, transform=ax2.transAxes)
    # ax2.text(plt_set['plotlabel_shift_3pan'], plt_set['plotlabel_up_3pan'], '(b)', transform=ax2.transAxes,
    #          fontsize=plt_set['label_font_size'], va='top', ha='right')
    #
    # ax3.errorbar(shannon_e_list, h_inf_list, h_inf_std_list, marker='o', linestyle='none', zorder=1)
    # ax3.set_xlabel('Shannon entropy of measured aa preferences')
    # ax3.set_ylabel('inferred $h$')
    # ax3.set_ylim(-1.5, 1.5)
    # text = '$r_{spearman}$ = %.2f, p = %.e' % (rho_shannon, prho_shannon)
    # ax3.text(0.05, 0.95, text, ha='left', va='top', fontsize=12, transform=ax3.transAxes)
    # ax3.text(plt_set['plotlabel_shift_3pan'], plt_set['plotlabel_up_3pan'], '(c)', transform=ax3.transAxes,
    #          fontsize=plt_set['label_font_size'], va='top', ha='right')

    plt.savefig(this_plot_filepath, bbox_inches='tight')
    plt.close()


def main():
    ## plot HA strain succession from 1968 to 2020
    # exe_plot_strainSuccession_HA()

    ## calculate and save minus_f_host_yearly
    # sigma_h = 1
    # D0 = 5
    # exe_minus_fhost_yearly(sigma_h, D0)

    ## plot distribution of minus_f_host_yearly
    # sigma_h = 1
    # D0 = 5
    # exe_plot_minus_fhost_yearly(sigma_h, D0)

    ## add reference sequence to dictionary
    # add_reference_sequences_from_fasta('BI_16190_68_ProteinFasta.fasta', 'BI_16190_68')
    # add_reference_sequences_from_fasta('Perth_16_2009_ProteinFasta.fasta', 'Perth_16_2009')
    # add_reference_sequences_from_fasta('Perth_16_2009_G78D_T212I_ProteinFasta.fasta', 'Perth_16_2009_G78D_T212I')
    # print_seq_refs() # print names of added reference sequences

    ## run inference
    # seq_ref_name = 'Perth_16_2009_G78D_T212I' # 'BI_16190_68'
    # sigma_h = 1
    # D0 = 5
    # # fixed params:
    # lambda_h = 10 ** (-4) # 10**(-4)
    # lambda_J = 1 # only needed for inference with couplings
    # lambda_f = 10 ** (-4)
    # inf_start = 0
    # inf_end = 43 # 53 (53 is length of year_list, 43 is 2010 as last year)
    # res_epitope_list = def_res_epitope_list()
    # res_allepitopes_list = [res for res_list in res_epitope_list for res in res_list]
    # res_targeted = res_allepitopes_list
    # # run inference with chosen params:
    # exe_inference_noCouplings(seq_ref_name, sigma_h, D0, res_targeted,
    #                           lambda_h, lambda_f, inf_start, inf_end)
    # exe_inference_WithCouplings(seq_ref_name, sigma_h, D0, res_targeted,
    #                             lambda_h, lambda_J, lambda_f, inf_start, inf_end)

    ## evaluate inference: print and plot inferred params
    # seq_ref_name = 'Perth_16_2009_G78D_T212I' # 'BI_16190_68'
    # sigma_h = 1
    # D0 = 5
    # eval_inference_noCouplings(seq_ref_name, sigma_h, D0)

    # compare inferred fitness coefficients to mutational fitness effects
    # measured by Lee et al. 2018 (PNAS)
    # save comparison figure and print/save rank correlations
    sigma_h = 1
    D0 = 5
    comparison_inference_LeeDeepMutScanning(sigma_h, D0, inf_scheme='noCouplings')
    # comparison_inference_LeeDeepMutScanning(sigma_h, D0, inf_scheme='WithCouplings')

# if this file is run from the console, the function main will be executed
if __name__ == '__main__':
    main()