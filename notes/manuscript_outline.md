# Introduction

- description of influenza evolutionary mechanisms/patterns and available 'raw' data (without tree inference)
    - __figure__: HA strain succession

- Approaches for using HA sequence data for prediction towards seasonal vaccine development:
    - prediction of sequence evolution:
        - Neher, Russell, Shraiman (2014), eLife
        - Luksza and Laessig (2014), Nature
    - mapping and prediction of antigenic evolution:
        - Neher et al. (2016), PNAS
    - only directly useful for seasonal vaccine development, not for long-term protective approaches

- Approaches for development of universal, cross-protective influenza vaccines
    - strategies that target conserved, but less accessible, sites (stem, receptor binding region)
    - key challenge: less accessible therefore hard to develop abs against and intrinsic fitness unknown (might be able to escape easily)

- approaches for inferring intrinsic fitness of other highly mutable viruses:
    - HIV, (polio) (papers by Chakraborty lab)
    - use equilibrium approach with assumption that human immune response acts as random force, this assumption is not valid for influenza that is by population-wide immune memory driven out of equilibrium
    - Cocco and Monasson review for standard approaches (adaptive cluster expansion)

- fitness inference method for time series data (MPL):
    - considers time-invariant fitness landscape (does not try to disentangle immune-dependent from intrinsic contribution)
    - does not take into account fitness coupling of pair mutations
    

# Model of influenza antigen evolution

- model mechanisms, assumptions, temporal coarse-graining (see model schematic in slides)
- mathematical model formulation (see slides) (take out details for methods)
    - essential model equations
    - fitness formulation
    - __table__: model parameters plus inference parameters
    - __figure__: strain succession from simulated data 


# Inference of intrinsic fitness from flu-like simulated sequence data

- inference method and stringency assumption:
    - essential equations
    - refer to parameter table
    - __figure__: widths of fitness dists from example simulation
    - __figure?__: log(x(t+1)/xm(t)) for various B/N

- test of inference performance
    - __figure__: inferred vs. simulated fitness params with correlation (and classification curves) for one example analysis
    - __figure__: performance measure as function of sample size/ number of seasons
    - __figure__: performance and stringency measure as function of sequence length
    - __figure__: performance and stringency measure as function of population size

# Discussion 

- summary of results, we were able to
    - infer intrinsic fitness landscape from influenza-like immune-driven sequence data
    - approach takes into account single and pair mutations
    - like MPL, controls for genetic linkage effects, by taking into account full sequence information for immune-dependent fitness estimate of each sample
    - simulation of influenza-like evolution allows to analyze performance under different conditions

- in comparison with previous studies (MPL, HIV studies):
    - were able to disentangle time-varying immune-dependent from intrinsic fitness in inference
    - were able to infer single and pairwise mutational effects

- outlook:
    - inference of intrinsic fitness from HA yearly sequence data
        - use HI assays for cross-immunity data
        - identify regions/sites/combos that are ost vulnerable
        - design vaccine or antibody treatment, which target those vulnerable regions
    - challenges
        - useful coarse-graining of HA data for fitness inference with the available data
        - test prediction with simulation/prediction on held-out test set/ predictions that can be verified in small-scale experiment
        - challenge of development of treatment/prevention strategy (vaccination, antibody treatment) based on intrinsic fitness information
        - will there be specific regions/sites/combinations of sites that are most vulnerable independent of the current sequence state and sequence background?
        - we don't want to/need to target epitopes that are already highly targeted by immune memory responses
        - we need an accurate cross-immunity measure for our model (based on Hamming distance? distance within each epitope?)
        - might have to take into account that different head epitopes are not equally targeted?
        might need to take into account epitope/site accessibility to not infer that instead of actual intrinsic fitness
- additional considerations for effective influenza treatments:
    - T-cell immunity
    - other antigens besides HA

